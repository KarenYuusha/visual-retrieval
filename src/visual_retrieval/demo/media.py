"""Lazy FFmpeg browser previews with bounded timeouts and atomic cache files."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import uuid

from PIL import Image, ImageDraw


class MediaError(ValueError):
    pass


class MediaCache:
    def __init__(self, root, ffmpeg='ffmpeg', ffprobe='ffprobe'):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ffmpeg, self.ffprobe = shutil.which(ffmpeg), shutil.which(ffprobe)
        if not self.ffmpeg or not self.ffprobe:
            raise MediaError('Install FFmpeg and ffprobe, and add them to PATH before launching the demo.')
        self._locks, self._guard = {}, threading.Lock()

    def _identity(self, item):
        path = Path(item['path']).resolve()
        if not path.is_file():
            raise MediaError('The source video is unavailable.')
        start, end = item.get('start'), item.get('end')
        if (start is None) != (end is None):
            raise MediaError('Invalid preview interval.')
        if start is not None and (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start):
            raise MediaError('Invalid preview interval.')
        stat = path.stat()
        identity = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, start=start, end=end,
                        version='thumbnail480x270_mp4h264silent960_v1')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        return key, path, start, end, lock

    def _run(self, arguments, timeout):
        try:
            output = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=timeout)
            return output.stdout
        except (subprocess.SubprocessError, OSError) as error:
            # Do not expose private paths/large FFmpeg stderr in the public UI.
            raise MediaError('This video could not be prepared for playback.') from error

    def thumbnail(self, item):
        key, source, start, end, lock = self._identity(item)
        output = self.root / f'{key}.jpg'
        with lock:
            if output.is_file() and output.stat().st_size:
                return output
            if start is None:
                duration = self._run([self.ffprobe, '-v', 'error', '-show_entries', 'format=duration',
                                     '-of', 'default=noprint_wrappers=1:nokey=1', str(source)], 15)
                try:
                    position = float(duration.strip()) / 2
                    if not math.isfinite(position) or position < 0:
                        raise ValueError()
                except ValueError as error:
                    raise MediaError('The video duration is unavailable.') from error
            else:
                position = (start + end) / 2
            temporary = self.root / f'{key}.{uuid.uuid4().hex}.tmp.jpg'
            try:
                self._run([self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                           '-ss', str(position), '-i', str(source), '-frames:v', '1',
                           '-vf', 'scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2',
                           '-q:v', '3', str(temporary)], 30)
                if not temporary.is_file() or not temporary.stat().st_size:
                    raise MediaError('A thumbnail is unavailable for this video.')
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            return output

    def video(self, item):
        key, source, start, end, lock = self._identity(item)
        output = self.root / f'{key}.mp4'
        with lock:
            if output.is_file() and output.stat().st_size:
                return output
            temporary = self.root / f'{key}.{uuid.uuid4().hex}.tmp.mp4'
            args = [self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y']
            if start is not None:
                args += ['-ss', str(start)]
            args += ['-i', str(source)]
            if start is not None:
                args += ['-t', str(end - start)]
            args += ['-map', '0:v:0', '-an', '-vf', "scale=w='min(960,trunc(iw/2)*2)':h=-2,fps=24",
                     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                     '-movflags', '+faststart', str(temporary)]
            try:
                self._run(args, 180)
                if not temporary.is_file() or not temporary.stat().st_size:
                    raise MediaError('A video preview is unavailable.')
                # Empty streams can produce a small MP4 container despite exit code 0.
                duration = self._run([self.ffprobe, '-v', 'error', '-show_entries', 'format=duration',
                                     '-of', 'default=noprint_wrappers=1:nokey=1', str(temporary)], 15)
                if not math.isfinite(float(duration.strip())) or float(duration.strip()) <= 0:
                    raise MediaError('The selected interval contains no playable video.')
                temporary.replace(output)
            except ValueError as error:
                if isinstance(error, MediaError):
                    raise
                raise MediaError('The video preview is invalid.') from error
            finally:
                temporary.unlink(missing_ok=True)
            return output

    def placeholder(self):
        path = self.root / 'preview_unavailable.jpg'
        with self._guard:
            if not path.is_file():
                image = Image.new('RGB', (480,270), '#f1f0ec')
                draw = ImageDraw.Draw(image)
                draw.text((240,135), 'Preview unavailable', fill='#777771', anchor='mm')
                image.save(path, 'JPEG')
        return path
