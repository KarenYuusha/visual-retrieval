"""Encode WebVid-CoVR with four model spaces; evaluate ONLY CoVR-2."""
import argparse
from pathlib import Path

from visual_retrieval.common import write_json
from visual_retrieval.config import DEFAULT_BASE,MODEL_NAMES
from visual_retrieval.data.webvid_covr import PROTOCOL,load_subset,check_media,serializable_queries
from visual_retrieval.features.webvid_covr import extract_baseline,release_gpu
from visual_retrieval.composed.webvid_covr import extract_queries
from visual_retrieval.evaluation.webvid_covr import evaluate,clear_reports


def prepare_gallery(*args,**kwargs):
    from visual_retrieval.composed.prepare import prepare
    try:
        return prepare(*args,**kwargs)
    finally:
        release_gpu()


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_BASE/'webvid_covr')
    parser.add_argument('--split',choices=['train','val','test'],default='test')
    parser.add_argument('--subset',type=Path,help='Downloader subset.json; default ROOT/samples/SPLIT/subset.json.')
    parser.add_argument('--limit',type=int,help='Only the first N downloaded samples; default all saved samples.')
    parser.add_argument('--stage',choices=['all','check','extract','evaluate'],default='all')
    parser.add_argument('--output-root',type=Path,help='Default ROOT/features/webvid_covr/SPLIT.')
    parser.add_argument('--models',choices=MODEL_NAMES,nargs='+',default=list(MODEL_NAMES),
                        help='Baseline ENCODERS to run; CoVR-2 is always encoded. None of these baselines is evaluated.')
    parser.add_argument('--clip-checkpoint',type=Path,help='Optional local HF CLIP directory; otherwise cached/downloaded.')
    parser.add_argument('--clip4clip-checkpoint',type=Path,help='Default ROOT-parent/models/clip4clip_webvid150k_hf.bin.')
    parser.add_argument('--internvideo2-checkpoint',type=Path,help='Optional local .pt; otherwise cached/downloaded.')
    parser.add_argument('--covr2-base-checkpoint',type=Path,help='Default ROOT-parent/weights/covr2/blip2_finetune_coco.pth.')
    parser.add_argument('--covr2-checkpoint',type=Path,help='Default ROOT-parent/weights/covr2/coir+covr.ckpt.')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default='auto',help='Baseline precision. CoVR-2 uses fp16 CUDA / fp32 CPU.')
    parser.add_argument('--text-batch-size',type=int,default=32)
    parser.add_argument('--decode-workers',type=int,default=2,help='CPU video decoding threads; 0 disables prefetch.')
    parser.add_argument('--video-batch-size',type=int,help='Default 4 for CLIP/CLIP4Clip, 1 for InternVideo2. InternVideo2 still infers each video separately.')
    parser.add_argument('--covr2-frame-batch-size',type=int,default=1,help='Gallery frames per inference; default 1. Try 2 if VRAM permits.')
    parser.add_argument('--validation-workers',type=int,default=4,help='Parallel gallery validation threads; 0 runs synchronously.')
    parser.add_argument('--gallery-cache-mb',type=int,default=512,help='RAM budget for validated gallery tokens during evaluation; 0 disables caching.')
    parser.add_argument('--gallery',choices=['targets','all'],default='targets',help='Unique targets (default), or all unique reference+target videos.')
    parser.add_argument('--top-k',type=int,default=10,help='Candidates to export per query; evaluation always ranks the FULL gallery.')
    parser.add_argument('--allow-incomplete',action='store_true',help='Explicitly permit partial downloads/extraction; omissions and denominators are reported.')
    return parser


def main(argv=None):
    parser=make_parser();args=parser.parse_args(argv)
    if args.text_batch_size<1 or args.top_k<1 or (args.limit is not None and args.limit<1):
        parser.error('Batch size, top-k and limit must be positive.')
    if args.decode_workers<0 or args.covr2_frame_batch_size<1 or (args.video_batch_size is not None and args.video_batch_size<1):
        parser.error('Decoder workers must be nonnegative and video/frame batch sizes positive.')
    if args.validation_workers<0 or args.gallery_cache_mb<0:
        parser.error('Validation workers and gallery cache MB must be nonnegative.')
    try:
        data=load_subset(args.root,args.split,args.subset,args.limit,args.allow_incomplete)
        output=(args.output_root or data.root/'features'/'webvid_covr'/args.split).resolve()
        results=output/'results'/'covr2'
        print(f'WebVid-CoVR {data.split}: {len(data.queries)} image/edit queries; {len(data.items)} unique videos.',flush=True)
        print(f'Evaluation model: CoVR-2 ONLY; gallery: {args.gallery}; exported top-k: {args.top_k}.',flush=True)
        if args.stage in ('check','extract','all'):
            missing=check_media(data)
            print(f'Missing source files: {len(missing)}',flush=True)
            if missing and not args.allow_incomplete:
                raise ValueError('Missing media: '+', '.join(missing[:5])+'. Restore files or use --allow-incomplete.')
        if args.stage=='check':
            print('Data check passed. No checkpoint downloaded, encoder loaded, features extracted or model evaluated.')
            return 0
        if args.stage in ('extract','all'):
            from visual_retrieval.models.base import resolve_precision
            resolve_precision(args.device,args.precision)
            base=args.covr2_base_checkpoint or data.root.parent/'weights'/'covr2'/'blip2_finetune_coco.pth'
            trained=args.covr2_checkpoint or data.root.parent/'weights'/'covr2'/'coir+covr.ckpt'
            clip4clip=args.clip4clip_checkpoint or data.root.parent/'models'/'clip4clip_webvid150k_hf.bin'
            for path in (base,trained):
                if not path.is_file():raise ValueError(f'Missing CoVR-2 checkpoint: {path}')
            if 'clip4clip' in args.models and not clip4clip.is_file():
                raise ValueError('Provide --clip4clip-checkpoint to the trained meanP/2d .bin file.')
            if args.internvideo2_checkpoint and not args.internvideo2_checkpoint.is_file():
                raise ValueError('Invalid --internvideo2-checkpoint file.')
            clear_reports(results)
            checkpoints=dict(clip=args.clip_checkpoint,clip4clip=clip4clip,internvideo2=args.internvideo2_checkpoint)
            for model in dict.fromkeys(args.models):
                print(f'Encoding {model} (no evaluation)…',flush=True)
                extract_baseline(data,model,output/model,checkpoint=checkpoints[model],device=args.device,
                    precision=args.precision,text_batch_size=args.text_batch_size,allow_incomplete=args.allow_incomplete,
                    decode_workers=args.decode_workers,video_batch_size=args.video_batch_size)
            source_manifest=output/'gallery_source.json'
            write_json(source_manifest,dict(protocol=PROTOCOL,dataset='webvid_covr',split=data.split,
                subset_sha256=data.subset_sha256,requested_video_ids=[item['item_id'] for item in data.items],
                gallery_items=[dict(item,path=str(item['path'])) for item in data.items],
                queries=serializable_queries(data.queries)))
            print('Encoding frozen CoVR-2 gallery tokens…',flush=True)
            gallery_manifest=prepare_gallery(data.root,source_manifest,base,output=output/'covr2',device=args.device,
                decode_workers=args.decode_workers,frame_batch_size=args.covr2_frame_batch_size)
            if gallery_manifest.get('failures') and not args.allow_incomplete:
                raise ValueError('CoVR-2 gallery extraction incomplete; retry or use --allow-incomplete.')
            print('Encoding trained CoVR-2 image + modification queries…',flush=True)
            extract_queries(data,output/'covr2_queries',base,trained,device=args.device,
                            allow_incomplete=args.allow_incomplete)
        if args.stage in ('evaluate','all'):
            evaluate(data,output/'covr2',output/'covr2_queries',results,gallery=args.gallery,
                     top_k=args.top_k,allow_incomplete=args.allow_incomplete,
                     validation_workers=args.validation_workers,gallery_cache_mb=args.gallery_cache_mb)
            print('CoVR-2 report:',results/'evaluation.json',flush=True)
        return 0
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        parser.exit(1,f'Error: {error}\n')


if __name__=='__main__':
    raise SystemExit(main())
