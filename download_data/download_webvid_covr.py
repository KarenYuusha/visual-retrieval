"""Download a bounded, resumable subset of complete WebVid-CoVR examples.

One sample = reference video + modification text + target video. The authors'
video mirror is tried first; only the requested videos are downloaded.
"""
import argparse
import csv
import html
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote, urlparse

import requests

ANNOTATION_REVISION = 'cb47f879890542578a2c3769cb168c69b927b4d8'
VIDEO_REVISION = '73c6585be02ce6fa32d08a1aef5b9c36e30c9510'
ANNOTATION_REPO = 'lucas-ventura/WebVid-CoVR'
VIDEO_REPO = 'lucas-ventura/WebVid'
SPLITS = {'train': 'webvid2m-covr_train.csv', 'val': 'webvid8m-covr_val.csv',
          'test': 'webvid8m-covr_test.csv'}
DEFAULT_ROOT = Path(r'S:\video_retrieval\webvid_covr')

def hub_url(repo, revision, filename):
    return f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(filename, safe="/")}'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def download_file(url, destination, timeout=30, retries=2):
    """Stream to a temporary file, checking HTTP status and download length."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    if urlparse(url).scheme not in ('http', 'https'):
        raise ValueError('Expected an HTTP(S) download URL.')
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(min(timeout, 15), timeout)) as response:
                response.raise_for_status()
                expected = response.headers.get('Content-Length')
                size = 0
                with temporary.open('wb') as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        size += len(block)
                        if size > 512 * 1024 * 1024:
                            raise ValueError('Download exceeds the 512 MiB per-file limit.')
                        handle.write(block)
                if size == 0 or (expected and not response.headers.get('Content-Encoding')
                                 and size != int(expected)):
                    raise OSError('Empty or truncated download.')
            os.replace(temporary, destination)
            return
        except (requests.RequestException, OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if attempt == retries or status in (400, 401, 403, 404) or isinstance(error, ValueError):
                raise OSError(f'Download failed: {url}: {error}') from error
            time.sleep(min(2 ** attempt, 4))


def probe_video(path, timeout=30):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=timeout, check=True)
    data = json.loads(result.stdout)
    streams = data.get('streams', [])
    duration = float(data.get('format', {}).get('duration', 0))
    if (not streams or streams[0].get('width', 0) < 1 or streams[0].get('height', 0) < 1
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError(f'No usable video stream: {path}')
    # A truncated faststart MP4 can have intact headers and duration. Decode
    # the video stream before treating either cached or new media as complete.
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-err_detect', 'explode',
                    '-i', str(path), '-map', '0:v:0', '-an', '-f', 'null', '-'],
                   capture_output=True, timeout=max(timeout, min(duration * 2, 120)), check=True)
    return duration


def save_reference_frame(video, destination, duration):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.tmp.jpg')
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(duration / 2),
                        '-i', str(video), '-frames:v', '1', '-q:v', '2', str(temporary)],
                       capture_output=True, timeout=60, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError('Could not decode a reference frame.')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def video_path(value):
    # Official pth1/pth2 have the form numeric-shard/numeric-video-id, no suffix.
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]+/[0-9]+', value):
        raise ValueError(f'Invalid annotation video path: {value!r}')
    return value + '.mp4'


def validate_row(row):
    video_path(row.get('pth1'))
    video_path(row.get('pth2'))
    if not isinstance(row.get('edit'), str) or not row['edit'].strip():
        raise ValueError('Annotation has no modification text.')


def write_outputs(root, folder, result, raw_rows):
    result['unique_videos'] = len({sample[key] for sample in result['samples']
                                   for key in ('reference_id','target_id')})
    atomic_json(folder / 'subset.json', result)
    temporary = folder / 'annotations.tmp'
    # Preserve all original fields, including the official annotation frame scores.
    fields = list(raw_rows[0]) if raw_rows else ['pth1', 'pth2', 'edit', 'txt1', 'txt2', 'scores']
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    os.replace(temporary, folder / 'annotations.csv')
    cards = []
    for sample in result['samples']:
        image = quote(os.path.relpath(root / sample['reference_image'], folder).replace('\\', '/'), safe='/')
        target = quote(os.path.relpath(root / sample['target_video'], folder).replace('\\', '/'), safe='/')
        cards.append(f'<article><h2>Sample {sample["row_index"] + 1}</h2>'
            f'<div class="pair"><div><p>Reference image</p><img src="{image}" alt="Reference frame"></div>'
            f'<div><p>Target video</p><video src="{target}" controls muted preload="none"></video></div></div>'
            f'<p><strong>Modification:</strong> {html.escape(sample["modification"])}</p>'
            f'<p>Reference caption: {html.escape(sample["reference_caption"])}</p>'
            f'<p>Target caption: {html.escape(sample["target_caption"])}</p></article>')
    preview = ('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>WebVid-CoVR samples</title><style>body{max-width:1000px;margin:32px auto;'
        'padding:0 16px;font:16px system-ui}article{margin:36px 0;padding-bottom:20px;border-bottom:1px solid #ccc}'
        '.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}'
        'img,video{width:100%;max-height:300px;object-fit:contain;background:#111}</style>'
        f'<h1>WebVid-CoVR · {html.escape(result["split"])} · {len(cards)} complete samples</h1>'
        '<p>Reference images are middle frames extracted locally. These examples form a subset, '
        'not the complete benchmark gallery.</p>' + ''.join(cards) + '</html>')
    (folder / 'preview.tmp').write_text(preview, encoding='utf-8')
    os.replace(folder / 'preview.tmp', folder / 'preview.html')


def prepare(root=DEFAULT_ROOT, samples=5, split='test', source='auto', annotation_file=None,
            max_attempts=None, start_row=0, timeout=30, retries=2):
    if type(samples) is not int or samples < 1:
        raise ValueError('--samples must be a positive integer.')
    if split not in SPLITS or source not in ('auto', 'hub', 'original'):
        raise ValueError('Invalid split or video source.')
    if start_row < 0 or not math.isfinite(timeout) or timeout <= 0 or retries < 0:
        raise ValueError('Use a nonnegative start row/retries and positive timeout.')
    max_attempts = max(50, samples * 10) if max_attempts is None else max_attempts
    if max_attempts < 1:
        raise ValueError('--max-attempts must be positive.')
    for command in ('ffprobe', 'ffmpeg'):
        if not shutil.which(command):
            raise ValueError(f'{command} is required on PATH (the demo already uses FFmpeg).')
    root = Path(root).expanduser().resolve()
    folder = root / 'samples' / split
    folder.mkdir(parents=True, exist_ok=True)
    (root / 'raw_videos').mkdir(exist_ok=True)
    (root / 'reference_frames').mkdir(exist_ok=True)
    raw = root / 'raw_data'
    raw.mkdir(exist_ok=True)
    custom_annotations = annotation_file is not None
    if annotation_file is None:
        annotation_file = raw / SPLITS[split]
        if not annotation_file.exists():
            print(f'Downloading {split} annotations…', flush=True)
            download_file(hub_url(ANNOTATION_REPO, ANNOTATION_REVISION, SPLITS[split]),
                          annotation_file, timeout=timeout, retries=retries)
    annotation_file = Path(annotation_file).resolve()
    with annotation_file.open('rb') as handle:
        annotation_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = dict(dataset='WebVid-CoVR', split=split, requested_samples=samples,
        complete=False, start_row=start_row, max_attempts=max_attempts,
        annotation_file=str(annotation_file), annotation_sha256=annotation_sha256,
        annotation_revision=None if custom_annotations else ANNOTATION_REVISION,
        video_revision=VIDEO_REVISION, source=source, samples=[], failures=[])
    path_to_url = None
    durations = {}
    raw_rows = []
    frames_ready = set()

    def ensure_video(pth):
        nonlocal path_to_url
        relative = video_path(pth)
        video_id = Path(relative).stem
        destination = root / 'raw_videos' / (video_id + '.mp4')
        if video_id in durations:
            return destination, durations[video_id]
        if destination.is_file():
            try:
                duration = probe_video(destination)
                durations[video_id] = duration
                return destination, duration
            except (ValueError, OSError, subprocess.SubprocessError):
                print('Replacing invalid cached video:', video_id, flush=True)
                destination.unlink()
        errors = []
        if source in ('auto', 'hub'):
            try:
                download_file(hub_url(VIDEO_REPO, VIDEO_REVISION, 'train/' + relative),
                              destination, timeout=timeout, retries=retries)
                duration = probe_video(destination)
                durations[video_id] = duration
                return destination, duration
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                destination.unlink(missing_ok=True)
                errors.append(str(error))
        if source in ('auto', 'original'):
            try:
                if path_to_url is None:
                    list_name = ('webvid2m' if split == 'train' else 'webvid8m') + f'-covr_paths-{split}.json'
                    url_file = raw / list_name
                    if not url_file.exists():
                        download_file('https://imagine.enpc.fr/~ventural/covr/dataset/' + list_name,
                                      url_file, timeout=timeout, retries=retries)
                    mapping = json.loads(url_file.read_text(encoding='utf-8'))
                    if not isinstance(mapping, dict):
                        raise ValueError('Invalid original URL mapping.')
                    path_to_url = mapping
                url = path_to_url.get(relative)
                if not isinstance(url, str):
                    raise ValueError(f'No original video URL for {relative}.')
                download_file(url, destination, timeout=timeout, retries=retries)
                duration = probe_video(destination)
                durations[video_id] = duration
                return destination, duration
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                destination.unlink(missing_ok=True)
                errors.append(str(error))
        raise OSError('; '.join(errors))

    # Stream the CSV: even the much larger training annotations need not fit RAM.
    with annotation_file.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'pth1', 'pth2', 'edit'}.issubset(reader.fieldnames or []):
            raise ValueError('Expected official CSV columns pth1, pth2 and edit.')
        attempted = 0
        try:
            for index, row in enumerate(reader):
                if index < start_row:
                    continue
                if len(result['samples']) >= samples or attempted >= max_attempts:
                    break
                attempted += 1
                try:
                    validate_row(row)
                    reference, duration = ensure_video(row['pth1'])
                    target, _ = ensure_video(row['pth2'])
                    frame = root / 'reference_frames' / (reference.stem + '.jpg')
                    if reference.stem not in frames_ready:
                        save_reference_frame(reference, frame, duration)
                        frames_ready.add(reference.stem)
                    sample = dict(row_index=index, reference_id=reference.stem, target_id=target.stem,
                        reference_video=reference.relative_to(root).as_posix(),
                        target_video=target.relative_to(root).as_posix(),
                        reference_image=frame.relative_to(root).as_posix(),
                        reference_time=duration / 2, modification=row['edit'].strip(),
                        reference_caption=row.get('txt1', '') or '', target_caption=row.get('txt2', '') or '')
                    result['samples'].append(sample)
                    raw_rows.append(row)
                    print(f'[{len(result["samples"])}/{samples}] {reference.stem} + {sample["modification"]} -> {target.stem}', flush=True)
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    result['failures'].append(dict(row_index=index, pth1=row.get('pth1'), pth2=row.get('pth2'), error=str(error)))
                    print(f'Skipped row {index + 1}: {error}', flush=True)
                result['complete'] = len(result['samples']) == samples
                write_outputs(root, folder, result, raw_rows)
        finally:
            # Preserve all completed pairs if interrupted with Ctrl+C.
            write_outputs(root, folder, result, raw_rows)
    print(f'Complete samples: {len(result["samples"])}/{samples}; failed rows: {len(result["failures"])}', flush=True)
    print('Preview:', folder / 'preview.html', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--samples', type=int, default=5, help='Number of complete reference/text/target pairs (default: 5).')
    parser.add_argument('--split', choices=list(SPLITS), default='test')
    parser.add_argument('--source', choices=['auto', 'hub', 'original'], default='auto',
                        help='auto: author video mirror, then original URL if missing.')
    parser.add_argument('--max-attempts', type=int, help='Maximum annotation rows to try; default max(50, samples*10).')
    parser.add_argument('--start-row', type=int, default=0, help='Zero-based annotation offset (default: 0).')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--annotation-file', type=Path, help='Use an already downloaded official CSV.')
    args = parser.parse_args(argv)
    try:
        result = prepare(**vars(args))
    except KeyboardInterrupt:
        print('\nStopped. Rerun the same command to reuse completed videos and continue.')
        return 130
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f'Error: {error}\n')
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
