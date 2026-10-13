"""Minimal Gradio presentation and password-protected remote launcher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import logging
import os
from pathlib import Path
import time

from visual_retrieval.config import DEFAULT_BASE, MODEL_NAMES
from .config import DemoConfig
from .media import MediaCache, MediaError
from .service import SearchService

LABELS = {'clip': 'CLIP', 'clip4clip': 'CLIP4Clip', 'internvideo2': 'InternVideo2'}
CSS = """
.gradio-container {max-width:1120px !important; margin:auto !important; padding:44px 24px !important;}
#heading h1 {font-size:28px; font-weight:550; letter-spacing:-.8px; margin:0 0 18px;}
#search-row {gap:12px; align-items:stretch;}
#query textarea {font-size:16px; padding:15px 18px; border-radius:12px;}
#search-button {border-radius:12px; min-height:54px;}
#choices {max-width:470px; margin:10px 0 2px; gap:12px;}
#summary {color:#727873; font-size:13px; min-height:24px;}
#results {border:none; background:transparent;}
#results .caption {font-size:12px;}
#selected-preview {margin-top:20px;}
#selection-detail {font-size:13px; color:#727873;}
footer {display:none !important;}
@media(max-width:600px) {.gradio-container {padding:24px 14px !important;} #choices {max-width:100%;}}
"""


def auth_from_env(share, environ=None):
    environ = os.environ if environ is None else environ
    username, password = environ.get('VR_DEMO_USERNAME'), environ.get('VR_DEMO_PASSWORD')
    if bool(username) != bool(password):
        raise ValueError('Set both VR_DEMO_USERNAME and VR_DEMO_PASSWORD, or neither for local access.')
    if share and not (username and password):
        raise ValueError('Sharing requires VR_DEMO_USERNAME and VR_DEMO_PASSWORD environment variables.')
    if username and not username.strip():
        raise ValueError('VR_DEMO_USERNAME must be nonempty.')
    return (username, password) if username and password else None


class DemoController:
    def __init__(self, service, media, top_k=10):
        self.service, self.media, self.top_k = service, media, top_k

    def summary(self, model, dataset='all'):
        counts = self.service.gallery_counts(model)
        selected = {key: value for key, value in counts.items() if dataset == 'all' or key == dataset}
        omitted = self.service.omitted_counts(model)
        text = f'{sum(selected.values()):,} clips · {len(selected)} dataset' + ('s' if len(selected) != 1 else '')
        missing = sum(omitted[key] for key in selected)
        if missing:
            text += f' · {missing} omitted'
        return text

    def search(self, text, model, dataset):
        started = time.perf_counter()
        results = self.service.search(text, model, dataset, self.top_k)
        def thumbnail(item):
            try:
                return str(self.media.thumbnail(item)), False
            except (MediaError, OSError):
                return str(self.media.placeholder()), True
        with ThreadPoolExecutor(max_workers=2) as pool:
            thumbnails = list(pool.map(thumbnail, results))
        images = [(path, f'{row["rank"]:02d} · {row["dataset_label"]} · {row["item_id"]}')
                  for row, (path, _) in zip(results, thumbnails)]
        status = f'{self.summary(model, dataset)} · {time.perf_counter() - started:.2f}s'
        if any(failed for _, failed in thumbnails):
            status += ' · Some previews unavailable'
        return images, results, status

    def select(self, results, index):
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(results):
            raise ValueError('Select a result from the current search.')
        row = results[index]
        item = self.service.resolve_result(row)
        path = str(self.media.video(item))
        description = f'{html.escape(item["dataset_label"])} · {html.escape(item["item_id"])} · similarity {row["cosine_similarity"]:.3f}'
        if item['start'] is not None:
            description += f' · {item["start"]:.2f}–{item["end"]:.2f}s'
        return path, description


def build_demo(service, media, default_model='internvideo2', top_k=10):
    import gradio as gr
    controller = DemoController(service, media, top_k)
    initial = default_model if default_model in service.available_models else service.available_models[0]
    with gr.Blocks(title='Video search', analytics_enabled=False) as demo:
        gr.Markdown('# Video search', elem_id='heading')
        with gr.Row(elem_id='search-row'):
            query = gr.Textbox(placeholder='Describe a moment…', label='Search', show_label=False,
                               lines=1, max_lines=3, container=False, scale=7, elem_id='query')
            button = gr.Button('Search', variant='primary', scale=1, min_width=100, elem_id='search-button')
        with gr.Row(elem_id='choices'):
            model = gr.Dropdown([(LABELS[m],m) for m in service.available_models], value=initial,
                                label='Model', show_label=False, container=False, filterable=False, elem_id='model-choice')
            dataset = gr.Dropdown(service.dataset_choices(initial), value='all', label='Dataset',
                                  show_label=False, container=False, filterable=False, elem_id='dataset-choice')
        status = gr.Markdown(controller.summary(initial), elem_id='summary')
        results = gr.Gallery(show_label=False, columns=[5,5,3,2,2,2], height='auto', object_fit='contain',
                             allow_preview=False, interactive=False, buttons=[], elem_id='results')
        state = gr.State([])
        context = gr.State(dict(revision=0, selection=0, model=initial, dataset='all'))
        with gr.Column(visible=False, elem_id='selected-preview') as selected:
            player = gr.Video(show_label=False, interactive=False, autoplay=False, include_audio=False,
                              height=360, buttons=[])
            details = gr.Markdown(elem_id='selection-detail')

        def do_search(text, chosen_model, chosen_dataset, ctx):
            if (chosen_model, chosen_dataset) != (ctx['model'], ctx['dataset']):
                return (gr.skip(),) * 6
            ctx['revision'] += 1
            revision = ctx['revision']
            try:
                images, rows, message = controller.search(text, chosen_model, chosen_dataset)
                for row in rows:
                    row['view_revision'] = revision
                answer = images, rows, message, None, '', gr.update(visible=False)
            except ValueError as error:
                answer = [], [], str(error), None, '', gr.update(visible=False)
            except Exception:
                logging.exception('Search failed')
                answer = [], [], 'Search failed. Check the PC terminal.', None, '', gr.update(visible=False)
            return answer if ctx['revision'] == revision else (gr.skip(),) * 6

        outputs = [results, state, status, player, details, selected]
        event_args = dict(fn=do_search, inputs=[query,model,dataset,context], outputs=outputs,
                          concurrency_id='inference', concurrency_limit=1, show_progress='minimal',
                          trigger_mode='always_last', api_visibility='undocumented')
        click_event = button.click(api_name='search', **event_args)
        submit_event = query.submit(api_name=False, **event_args)

        def select_video(rows, ctx, evt: gr.EventData):
            revision = ctx['revision']
            if not rows or rows[0].get('view_revision') != revision or evt._data.get('index') is None:
                return (gr.skip(),) * 3
            ctx['selection'] += 1
            selection = ctx['selection']
            try:
                path, text = controller.select(rows, evt._data['index'])
                answer = path, text, gr.update(visible=True)
            except (ValueError, OSError):
                answer = None, 'This preview is unavailable.', gr.update(visible=True)
            except Exception:
                logging.exception('Playback failed')
                answer = None, 'This preview could not be prepared.', gr.update(visible=True)
            return answer if (revision, selection) == (ctx['revision'], ctx['selection']) else (gr.skip(),) * 3
        select_event = results.select(select_video, inputs=[state,context], outputs=[player,details,selected],
                       api_name=False, concurrency_id='media', concurrency_limit=2, show_progress='minimal')

        def change_model(chosen_model, ctx):
            ctx.update(revision=ctx['revision'] + 1, model=chosen_model, dataset='all')
            return gr.update(choices=service.dataset_choices(chosen_model), value='all'), [], [], None, '', gr.update(visible=False), controller.summary(chosen_model)
        def warm_model(chosen_model):
            try:
                service.warmup(chosen_model)
            except Exception:
                logging.exception('Model warmup failed; the next search will retry')
        cancel = [click_event, submit_event, select_event]
        model.change(change_model, inputs=[model,context], outputs=[dataset,results,state,player,details,selected,status],
                     queue=False, api_name=False, cancels=cancel).then(
            warm_model, inputs=[model], outputs=[],
            concurrency_id='inference', concurrency_limit=1, show_progress='minimal', api_name=False)
        def change_dataset(m, d, ctx):
            ctx.update(revision=ctx['revision'] + 1, model=m, dataset=d)
            return [], [], None, '', gr.update(visible=False), controller.summary(m,d)
        dataset.change(change_dataset, inputs=[model,dataset,context],
                       outputs=[results,state,player,details,selected,status], queue=False, api_name=False, cancels=cancel)
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Custom gallery registry YAML, including future datasets.')
    parser.add_argument('--base-root', type=Path, help='Default dataset base; cannot be combined with --config.')
    parser.add_argument('--cache-dir', type=Path, help='Generated thumbnails/previews; default BASE/demo_cache.')
    parser.add_argument('--model', choices=MODEL_NAMES, help='Initial model; otherwise use configuration default.')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='Password-protected temporary public Gradio link.')
    parser.add_argument('--no-warmup', action='store_true', help='Load the model on the first search instead of before launch.')
    parser.add_argument('--check', action='store_true', help='Check galleries and FFmpeg without loading models or starting a server.')
    args = parser.parse_args(argv)
    if args.config and args.base_root:
        parser.error('Use either --config or --base-root.')
    if not 1 <= args.port <= 65535 or (args.top_k is not None and not 1 <= args.top_k <= 100):
        parser.error('Use a valid port and top-k from 1 to 100.')
    base = args.base_root or DEFAULT_BASE
    try:
        auth = auth_from_env(args.share) if not args.check else None
        config = DemoConfig.from_yaml(args.config) if args.config else DemoConfig.defaults(base)
        service = SearchService(config.datasets, args.device)
        media = MediaCache(args.cache_dir or base / 'demo_cache')
        for spec in config.datasets:
            if media.root.is_relative_to((spec.root / 'raw_videos').resolve()):
                raise ValueError('Use a preview cache outside raw_videos.')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for issue in service.issues:
        print('Unavailable gallery:', issue)
    for model in service.available_models:
        for dataset, count in service.gallery_counts(model).items():
            print(f'{model} / {dataset}: {count} clips; {service.omitted_counts(model)[dataset]} omitted')
    if args.check:
        print('Demo preflight passed. No model loaded or server started.')
        return
    initial = args.model or config.default_model
    if initial not in service.available_models:
        if args.model:
            parser.error(f'No saved gallery for {initial}.')
        initial = service.available_models[0]
        print(f'Using available initial model: {initial}')
    try:
        if not args.no_warmup:
            print(f'Warming {initial} on {args.device}…', flush=True)
            service.warmup(initial)
        app = build_demo(service, media, initial, args.top_k or config.top_k)
        import gradio as gr
        theme = gr.themes.Base(primary_hue='gray', neutral_hue='gray', font=['ui-sans-serif','system-ui','sans-serif'])
        app.launch(server_name='127.0.0.1', server_port=args.port, share=args.share, auth=auth,
                   allowed_paths=[str(media.root)], blocked_paths=[str((spec.root / 'raw_videos').resolve()) for spec in config.datasets],
                   theme=theme, css=CSS, footer_links=[], show_error=False, enable_monitoring=False)
    finally:
        service.close()


if __name__ == '__main__':
    main()
