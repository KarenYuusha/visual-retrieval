"""Bounded source-video decoding prefetch; inference stays on the main thread."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from visual_retrieval.data.video_sampling import read_video_segments


def prefetch_sources(sources, decode, workers=2):
    """Yield in source order, with at most `workers` futures retained.

    Zero workers provides synchronous decoding for debugging or low RAM.
    Thread workers are portable to Windows and never touch CUDA or models.
    """
    if workers < 0:
        raise ValueError('Decoder workers must be nonnegative.')
    if not workers:
        for source in sources:
            yield decode(source)
        return
    iterator = iter(sources)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    try:
        for _ in range(workers):
            source = next(iterator, None)
            if source is None:
                break
            pending.append(executor.submit(decode, source))
        while pending:
            result = pending.popleft().result()
            source = next(iterator, None)
            if source is not None:
                pending.append(executor.submit(decode, source))
            yield result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def decode_source(items, count_frames):
    try:
        results = read_video_segments(items[0]['path'], [(x['start'], x['end']) for x in items], count_frames)
    except (ValueError, OSError) as error:
        results = [error] * len(items)
    return list(zip(items, results))
