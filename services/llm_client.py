"""
Single entry point for calling a chat LLM.

Provider order:
  1. OpenAI       - when OPENAI_API_KEY is set (model: OPENAI_MODEL, default gpt-4o-mini)
  2. Ollama       - free local model at OLLAMA_URL (default http://localhost:11434),
                    model OLLAMA_MODEL (default llama3.2:3b, else the first installed model)
  3. None         - callers fall back to the rule-based answer engine
"""

import os
import time

import httpx

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
# 2048 tokens fits the trimmed prompt + answer and needs far less memory than larger windows.
# Every call uses the same options: changing them makes Ollama reload the model (~6 s).
OLLAMA_NUM_CTX = int(os.environ.get('OLLAMA_NUM_CTX', '2048'))
OLLAMA_KEEP_ALIVE = os.environ.get('OLLAMA_KEEP_ALIVE', '60m')   # keep the model in memory between questions
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
LLM_TIMEOUT = float(os.environ.get('LLM_TIMEOUT', '240'))


class LLMUnavailable(Exception):
    pass


_ollama_cache = {'checked_at': 0.0, 'model': None}


def _openai_key():
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    return key if key and key not in ('your-openai-api-key-here', 'sk-') else ''


def _ollama_model():
    """Return the Ollama model to use, or None if Ollama isn't reachable. Cached for 30s."""
    now = time.time()
    if now - _ollama_cache['checked_at'] < 30:
        return _ollama_cache['model']
    model = None
    try:
        r = httpx.get(f'{OLLAMA_URL}/api/tags', timeout=2.0)
        r.raise_for_status()
        names = [m.get('name') for m in r.json().get('models', [])]
        if OLLAMA_MODEL in names:
            model = OLLAMA_MODEL
        elif names:
            model = names[0]
    except Exception:
        model = None
    _ollama_cache.update(checked_at=now, model=model)
    return model


def provider_label():
    """Human-readable name of the active model, or None when no LLM is available."""
    if _openai_key():
        return f'{OPENAI_MODEL} (OpenAI)'
    model = _ollama_model()
    return f'{model} (local, Ollama)' if model else None


def is_available():
    return provider_label() is not None


def _ollama_payload(model, messages, max_tokens, temperature, stream):
    return {
        'model': model, 'messages': messages, 'stream': stream, 'keep_alive': OLLAMA_KEEP_ALIVE,
        'options': {'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': OLLAMA_NUM_CTX},
    }


def chat_stream(messages, max_tokens=500, temperature=0.3):
    """Yield the reply in small text chunks as the model writes it. Raises LLMUnavailable."""
    import json as _json
    key = _openai_key()
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, timeout=LLM_TIMEOUT)
            for chunk in client.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                                        max_tokens=max_tokens, temperature=temperature,
                                                        stream=True):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return
        except Exception as e:
            print(f'OpenAI stream failed, trying Ollama: {e}')
    model = _ollama_model()
    if not model:
        raise LLMUnavailable('No LLM provider is available (set OPENAI_API_KEY or start Ollama).')
    with httpx.stream('POST', f'{OLLAMA_URL}/api/chat',
                      json=_ollama_payload(model, messages, max_tokens, temperature, stream=True),
                      timeout=LLM_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = _json.loads(line)
            piece = (data.get('message') or {}).get('content')
            if piece:
                yield piece
            if data.get('done'):
                break


def warm_up(persona=None):
    """Load the model into memory and pre-process the fixed persona prompt, so the first
    real question doesn't pay for model loading. Safe to call in a background thread."""
    try:
        if _openai_key() or not _ollama_model():
            return
        messages = ([{'role': 'system', 'content': persona}] if persona else []) + [{'role': 'user', 'content': 'Hi'}]
        chat(messages, max_tokens=1, temperature=0)
        print('LLM warmed up.')
    except Exception as e:
        print(f'LLM warm-up skipped: {e}')


def chat(messages, max_tokens=500, temperature=0.3):
    """
    messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
    Returns the assistant's reply text. Raises LLMUnavailable if no provider works.
    """
    key = _openai_key()
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, timeout=LLM_TIMEOUT)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f'OpenAI call failed, trying Ollama: {e}')

    model = _ollama_model()
    if model:
        try:
            r = httpx.post(
                f'{OLLAMA_URL}/api/chat',
                json=_ollama_payload(model, messages, max_tokens, temperature, stream=False),
                timeout=LLM_TIMEOUT,
            )
            r.raise_for_status()
            return (r.json().get('message', {}).get('content') or '').strip()
        except Exception as e:
            print(f'Ollama call failed: {e}')
            _ollama_cache['checked_at'] = 0.0  # re-check availability next time

    raise LLMUnavailable('No LLM provider is available (set OPENAI_API_KEY or start Ollama).')
