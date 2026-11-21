import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'verifai-secret-key-2024-biomedical')
    # Same database file the Flask version used (instance/verifai.db); override with VERIFAI_DATABASE_URL
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'VERIFAI_DATABASE_URL',
        'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'verifai.db').replace('\\', '/'))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt', 'json'}
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'admin123'
    TOP_K_RESULTS = 5
    VERIFICATION_THRESHOLD = 0.6
