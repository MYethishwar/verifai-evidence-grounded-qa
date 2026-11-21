from fastapi import APIRouter, Form, Request

from extensions import db
from models.user import User
from web import flash, get_current_user, login_user, logout_user, redirect, render, url_for_name

router = APIRouter()


def _home_for(request: Request, user):
    return redirect(url_for_name(request, 'admin.dashboard' if user.role == 'admin' else 'user.dashboard'))


@router.get('/', name='index', include_in_schema=False)
def index(request: Request):
    return redirect(url_for_name(request, 'auth.home'))


@router.get('/home', name='auth.home')
def home(request: Request):
    from services import knowledge_base
    from services.evaluation_service import _bench_cache
    groups, topics = knowledge_base.load()
    return render(request, 'home.html',
                  bench=_bench_cache.get('result'),        # shown once a benchmark has run
                  kb_topics=len(topics), kb_groups=[g['name'] for g in groups])


@router.get('/register', name='auth.register')
def register_form(request: Request):
    user = get_current_user(request)
    if user.is_authenticated:
        return _home_for(request, user)
    return render(request, 'auth/register.html')


@router.post('/register')
def register(request: Request,
             name: str = Form(''), username: str = Form(''), email: str = Form(''),
             mobile: str = Form(''), password: str = Form(''), confirm_password: str = Form('')):
    name, username, email, mobile = name.strip(), username.strip(), email.strip(), mobile.strip()
    error = None
    if not all([name, username, email, mobile, password, confirm_password]):
        error = 'All fields are required.'
    elif password != confirm_password:
        error = 'Passwords do not match.'
    elif User.query.filter_by(username=username).first():
        error = 'Username already taken.'
    elif User.query.filter_by(email=email).first():
        error = 'Email already registered.'
    if error:
        flash(request, error, 'danger')
        return render(request, 'auth/register.html', status_code=400)
    user = User(name=name, username=username, email=email, mobile=mobile)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    flash(request, 'Request received. An administrator will activate your account.', 'success')
    return redirect(url_for_name(request, 'auth.login'))


@router.get('/login', name='auth.login')
def login_form(request: Request):
    user = get_current_user(request)
    if user.is_authenticated:
        return _home_for(request, user)
    return render(request, 'auth/login.html')


@router.post('/login')
def login(request: Request, username: str = Form(''), password: str = Form('')):
    user = User.query.filter_by(username=username.strip()).first()
    if not user or not user.check_password(password):
        flash(request, 'Invalid username or password.', 'danger')
        return render(request, 'auth/login.html', status_code=401)
    if user.status == 'pending':
        flash(request, 'Your account is pending admin activation.', 'warning')
        return render(request, 'auth/login.html', status_code=403)
    if user.status == 'inactive':
        flash(request, 'Your account has been deactivated.', 'danger')
        return render(request, 'auth/login.html', status_code=403)
    login_user(request, user)
    flash(request, f'Welcome back, {user.name}!', 'success')
    return _home_for(request, user)


@router.get('/logout', name='auth.logout')
def logout(request: Request):
    logout_user(request)
    flash(request, 'You have been signed out.', 'info')
    return redirect(url_for_name(request, 'auth.login'))
