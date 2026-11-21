"""
Web helpers shared by the FastAPI routers: templates, flash messages, sign-in
dependencies and a template `url_for` that accepts route names plus query parameters.
"""

import contextvars
import json
import os
import re
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context
from markupsafe import Markup, escape

from extensions import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))


# --------------------------------------------------------------------------
# Users and sign-in
# --------------------------------------------------------------------------

class AnonymousUser:
    is_authenticated = False
    id = None
    name = ''
    username = ''
    role = None
    status = None


class SignInRequired(Exception):
    """Raised by the auth dependencies; app.py turns it into a redirect to the sign-in page."""

    def __init__(self, message=None, category='info'):
        self.message, self.category = message, category


def get_current_user(request: Request):
    from models.user import User
    uid = request.session.get('uid')
    user = db.session.get(User, uid) if uid else None
    return user or AnonymousUser()


def require_user(request: Request):
    """Dependency: any signed-in, active account (administrators included)."""
    user = get_current_user(request)
    if not user.is_authenticated:
        raise SignInRequired()
    if user.status != 'active':
        request.session.pop('uid', None)
        raise SignInRequired('Your account is not active.', 'danger')
    return user


def require_admin(request: Request):
    """Dependency: signed-in administrator."""
    user = require_user(request)
    if user.role != 'admin':
        raise SignInRequired('Admin access required.', 'danger')
    return user


def login_user(request: Request, user):
    request.session.clear()
    request.session['uid'] = user.id


def logout_user(request: Request):
    request.session.clear()


# --------------------------------------------------------------------------
# Flash messages (stored in the signed session cookie)
# --------------------------------------------------------------------------

def flash(request: Request, message, category='info'):
    request.session.setdefault('_flashes', [])
    request.session['_flashes'] = request.session['_flashes'] + [[category, str(message)]]


def pop_flashes(request: Request):
    return [tuple(f) for f in request.session.pop('_flashes', [])]


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------

def url_for_name(request: Request, name, **values):
    """Route name -> URL. Path parameters fill the path; anything else becomes the query string."""
    if name == 'static':
        return '/static/' + str(values.get('filename', values.get('path', ''))).lstrip('/')
    route = next((r for r in request.app.router.routes if getattr(r, 'name', None) == name), None)
    if route is None:
        raise ValueError(f'No route named {name!r}')
    path_keys = set(getattr(route, 'param_convertors', {}).keys())
    path = request.app.url_path_for(name, **{k: v for k, v in values.items() if k in path_keys})
    query = {k: v for k, v in values.items() if k not in path_keys and v is not None}
    return str(path) + ('?' + urlencode(query) if query else '')


_current_request = contextvars.ContextVar('current_request', default=None)


@pass_context
def _template_url_for(context, name, **values):
    # Imported macros (e.g. the logo) don't receive the page context, so fall back to the
    # request being rendered.
    request = context.get('request') or _current_request.get()
    if name == 'static' or request is None:
        return '/static/' + str(values.get('filename', values.get('path', ''))).lstrip('/')
    return url_for_name(request, name, **values)


def redirect(url, status_code=303):
    return RedirectResponse(url, status_code=status_code)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render(request: Request, name, status_code=200, **context):
    """Render a template with the values every page uses."""
    from services import knowledge_base, llm_client
    user = get_current_user(request)
    route = request.scope.get('route')
    flashes = pop_flashes(request)
    try:
        vf_model, vf_kb_topics = llm_client.provider_label(), knowledge_base.topic_count()
    except Exception:
        vf_model, vf_kb_topics = None, 0
    context.update(
        current_user=user,
        current_endpoint=getattr(route, 'name', None),
        get_flashed_messages=lambda with_categories=False: flashes if with_categories else [m for _, m in flashes],
        vf_model=vf_model,
        vf_kb_topics=vf_kb_topics,
    )
    token = _current_request.set(request)
    try:
        return templates.TemplateResponse(request, name, context, status_code=status_code)
    finally:
        _current_request.reset(token)


# --------------------------------------------------------------------------
# Template filters
# --------------------------------------------------------------------------

def _from_json(value):
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return [str(value)]


def _first_name(user):
    from services.chat_service import _first_name
    return _first_name(user)


def _markdown(text):
    """Minimal, safe Markdown for answers: escapes HTML first, then bold, bullets, citations."""
    lines = str(escape(text or '')).split('\n')
    html, in_list, para = [], None, []

    def inline(s):
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        return re.sub(r'\[(\d{1,2})\]', r'<sup class="chat-cite">\1</sup>', s)

    def flush():
        if para:
            html.append('<p>' + inline(' '.join(para)) + '</p>')
            para.clear()

    for raw in lines:
        line = raw.strip()
        m = re.match(r'^([-*•]|\d+[.)])\s+(.*)$', line)
        if m:
            flush()
            tag = 'ol' if m.group(1)[0].isdigit() else 'ul'
            if in_list != tag:
                if in_list:
                    html.append(f'</{in_list}>')
                html.append(f'<{tag}>')
                in_list = tag
            html.append('<li>' + inline(m.group(2)) + '</li>')
        elif not line:
            flush()
            if in_list:
                html.append(f'</{in_list}>')
                in_list = None
        else:
            if in_list:
                html.append(f'</{in_list}>')
                in_list = None
            para.append(line)
    flush()
    if in_list:
        html.append(f'</{in_list}>')
    return Markup(''.join(html))


templates.env.globals['url_for'] = _template_url_for
templates.env.filters['from_json'] = _from_json
templates.env.filters['first_name'] = _first_name
templates.env.filters['md'] = _markdown
