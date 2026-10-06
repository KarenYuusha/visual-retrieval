"""Deterministic query-history baselines; no external LLM needed."""
def query_text(history, mode='accumulated'):
    if not history:
        raise ValueError('Enter a query before searching.')
    if mode == 'latest':
        return history[-1]
    if mode != 'accumulated':
        raise ValueError('Query mode must be latest or accumulated.')
    return '. '.join(text.rstrip('. ') for text in history)
