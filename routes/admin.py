import json
import os

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from werkzeug.utils import secure_filename

from config import Config
from extensions import db
from models.document import Dataset, Document, Passage
from models.question import ExistingResult, ProposedResult, QASession
from models.user import User
from services.evaluation_service import aggregate_reports, run_benchmark
from services.indexing_service import get_index_stats, run_indexing, run_passage_segmentation, run_preprocessing
from services.retrieval_service import build_index_from_db
from web import flash, redirect, render, require_admin, url_for_name

router = APIRouter(prefix='/admin')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt', 'json'}


# --------------------------------------------------------------------------
# Overview & users
# --------------------------------------------------------------------------

@router.get('/', name='admin.dashboard')
def dashboard(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/dashboard.html',
                  total_users=User.query.filter_by(role='user').count(),
                  active_users=User.query.filter_by(role='user', status='active').count(),
                  pending_users=User.query.filter_by(role='user', status='pending').count(),
                  total_datasets=Dataset.query.count(),
                  total_documents=Document.query.count(),
                  total_passages=Passage.query.count(),
                  total_questions=QASession.query.count(),
                  verified_answers=ProposedResult.query.filter_by(verification_status='Verified').count(),
                  sessions=QASession.query.order_by(QASession.created_at.desc()).limit(10).all(),
                  datasets=Dataset.query.order_by(Dataset.created_at.desc()).all(),
                  agg=aggregate_reports(),
                  index_stats=get_index_stats())


@router.get('/users', name='admin.users')
def users(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/users.html',
                  users=User.query.filter_by(role='user').order_by(User.created_at.desc()).all())


def _set_status(request, uid, status, message, category):
    user = User.query.get_or_404(uid)
    user.status = status
    db.session.commit()
    flash(request, message.format(user=user.username), category)
    return redirect(url_for_name(request, 'admin.users'))


# Links in the existing templates use GET; POST is accepted too for forms and API clients.
@router.post('/users/activate/{uid}', name='admin.activate_user_post')
@router.get('/users/activate/{uid}', name='admin.activate_user')
def activate_user(request: Request, uid: int, admin=Depends(require_admin)):
    return _set_status(request, uid, 'active', 'User {user} activated.', 'success')


@router.post('/users/deactivate/{uid}', name='admin.deactivate_user_post')
@router.get('/users/deactivate/{uid}', name='admin.deactivate_user')
def deactivate_user(request: Request, uid: int, admin=Depends(require_admin)):
    return _set_status(request, uid, 'inactive', 'User {user} deactivated.', 'warning')


@router.post('/users/delete/{uid}', name='admin.delete_user_post')
@router.get('/users/delete/{uid}', name='admin.delete_user')
def delete_user(request: Request, uid: int, admin=Depends(require_admin)):
    user = User.query.get_or_404(uid)
    db.session.delete(user)
    db.session.commit()
    flash(request, 'User deleted.', 'success')
    return redirect(url_for_name(request, 'admin.users'))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def _find_field(rec, keys, default=''):
    if not isinstance(rec, dict):
        return default
    for k in keys:
        for rk in rec.keys():
            if str(k).lower() == str(rk).strip().lower():
                val = rec[rk]
                if val is not None and str(val).strip() != '' and str(val).lower() != 'nan':
                    return str(val).strip()
    return default


def _parse_dataset_file(file_path, filename):
    try:
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext == 'csv':
            df = None
            for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']:
                try:
                    df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
                    break
                except Exception:
                    continue
            if df is None:
                df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
            return df.fillna('').to_dict('records'), None
        if ext in ('xlsx', 'xls'):
            return pd.read_excel(file_path).fillna('').to_dict('records'), None
        if ext == 'json':
            for enc in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        data = json.load(f)
                    return (data if isinstance(data, list) else [data]), None
                except UnicodeDecodeError:
                    continue
            return [], 'Invalid JSON format'
        if ext == 'txt':
            content = ''
            for enc in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if not content:
                return [], 'Empty or unreadable text file'
            if '===' in content or 'Abstract:' in content or 'Record ' in content:
                blocks = [b.strip() for b in content.split('=' * 20) if b.strip()] or \
                         [b.strip() for b in content.split('\n\n\n') if b.strip()]
                parsed = []
                for idx, block in enumerate(blocks, 1):
                    rec = {'title': f'Record {idx}', 'abstract': block, 'authors': '', 'journal': '', 'keywords': ''}
                    for line in (l.strip() for l in block.split('\n') if l.strip()):
                        low = line.lower()
                        value = line.split(':', 1)[-1].strip()
                        if low.startswith('record') or low.startswith('title:'):
                            rec['title'] = value or rec['title']
                        elif low.startswith('abstract:'):
                            rec['abstract'] = value
                        elif low.startswith('authors:'):
                            rec['authors'] = value
                        elif low.startswith('journal:'):
                            rec['journal'] = value
                        elif low.startswith('keywords:'):
                            rec['keywords'] = value
                    parsed.append(rec)
                if parsed:
                    return parsed, None
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            return [{'title': f'Record {i + 1}', 'abstract': line} for i, line in enumerate(lines)], None
        return [], 'Unsupported file type'
    except Exception as e:
        return [], str(e)


@router.get('/upload', name='admin.upload_dataset')
def upload_page(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/upload.html', datasets=Dataset.query.order_by(Dataset.created_at.desc()).all())


@router.post('/upload')
def upload_dataset(request: Request, dataset_file: UploadFile | None = File(None),
                   name: str = Form(''), description: str = Form(''), admin=Depends(require_admin)):
    def fail(message):
        flash(request, message, 'danger')
        return render(request, 'admin/upload.html', status_code=400,
                      datasets=Dataset.query.order_by(Dataset.created_at.desc()).all())

    if dataset_file is None or not dataset_file.filename:
        return fail('Please select a file.')
    filename = secure_filename(dataset_file.filename)
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return fail('Unsupported file type. Use CSV, Excel, TXT, or JSON.')
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    with open(file_path, 'wb') as out:
        out.write(dataset_file.file.read())

    records, error = _parse_dataset_file(file_path, filename)
    if error:
        return fail(f'Error parsing file: {error}')

    dataset = Dataset(name=name.strip() or filename, filename=filename, file_path=file_path,
                      description=description.strip(), records_count=len(records), uploaded_by=admin.id)
    db.session.add(dataset)
    db.session.flush()
    for idx, rec in enumerate(records, 1):
        title = _find_field(rec, ['title', 'headline', 'name', 'question', 'query'], default=f'Biomedical Record {idx}')
        abstract = _find_field(rec, ['abstract', 'text', 'content', 'body', 'description', 'summary', 'answer', 'context']) or title
        db.session.add(Document(
            dataset_id=dataset.id, title=title, abstract=abstract,
            authors=_find_field(rec, ['authors', 'author', 'researchers']),
            journal=_find_field(rec, ['journal', 'source', 'publisher', 'category']),
            year=_find_field(rec, ['year', 'date', 'published_year']),
            pmid=_find_field(rec, ['pmid', 'pubmed_id', 'id']),
            doi=_find_field(rec, ['doi', 'doi_link']),
            source_url=_find_field(rec, ['url', 'source_url', 'link']),
            keywords=_find_field(rec, ['keywords', 'keyword', 'tags', 'mesh_terms'])))
    db.session.commit()
    flash(request, f'Dataset uploaded successfully with {len(records)} records.', 'success')
    return redirect(url_for_name(request, 'admin.upload_dataset'))


@router.post('/dataset/delete/{did}', name='admin.delete_dataset_post')
@router.get('/dataset/delete/{did}', name='admin.delete_dataset')
def delete_dataset(request: Request, did: int, admin=Depends(require_admin)):
    dataset = Dataset.query.get_or_404(did)
    ds_name = dataset.name
    for doc in Document.query.filter_by(dataset_id=did).all():
        Passage.query.filter_by(document_id=doc.id).delete()
        db.session.delete(doc)
    if dataset.file_path and os.path.exists(dataset.file_path):
        try:
            os.remove(dataset.file_path)
        except OSError:
            pass
    db.session.delete(dataset)
    db.session.commit()
    build_index_from_db(force=True)
    flash(request, f'Dataset "{ds_name}" and all associated documents deleted successfully.', 'success')
    return redirect(request.headers.get('referer') or url_for_name(request, 'admin.dashboard'))


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@router.get('/preprocessing', name='admin.preprocessing')
def preprocessing_page(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/preprocessing.html', datasets=Dataset.query.all(), result=None)


@router.post('/preprocessing')
def preprocessing(request: Request, dataset_id: str = Form(''), admin=Depends(require_admin)):
    result = None
    if dataset_id:
        try:
            res1 = run_preprocessing(int(dataset_id))
            res2 = run_passage_segmentation(int(dataset_id))
            result = {**res1, **res2}
            flash(request, f'Preprocessing complete: {res1["processed"]} docs, {res2["total_passages"]} passages.', 'success')
        except Exception as e:
            flash(request, f'Error: {e}', 'danger')
    return render(request, 'admin/preprocessing.html', datasets=Dataset.query.all(), result=result)


@router.get('/features', name='admin.features')
def features(request: Request, admin=Depends(require_admin)):
    avg_words = db.session.query(db.func.avg(Passage.word_count)).scalar() or 0
    return render(request, 'admin/features.html', stats={
        'preprocessed_docs': Document.query.filter_by(is_preprocessed=True).count(),
        'total_passages': Passage.query.count(),
        'total_words': db.session.query(db.func.sum(Passage.word_count)).scalar() or 0,
        'avg_words_per_passage': round(float(avg_words), 1),
    })


@router.get('/indexing', name='admin.indexing')
def indexing_page(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/indexing.html', index_stats=get_index_stats(), result=None)


@router.post('/indexing')
def indexing(request: Request, admin=Depends(require_admin)):
    result = None
    try:
        result = run_indexing()
        flash(request, f'Indexing complete: {result["indexed_passages"]} passages indexed.', 'success')
    except Exception as e:
        flash(request, f'Indexing error: {e}', 'danger')
    return render(request, 'admin/indexing.html', index_stats=get_index_stats(), result=result)


# --------------------------------------------------------------------------
# Models & evaluation (measured data only)
# --------------------------------------------------------------------------

@router.get('/existing-model', name='admin.existing_model')
def existing_model(request: Request, admin=Depends(require_admin)):
    avg_rel = db.session.query(db.func.avg(ExistingResult.relevance_score)).scalar() or 0
    avg_time = db.session.query(db.func.avg(ExistingResult.response_time)).scalar() or 0
    return render(request, 'admin/existing_model.html', agg=aggregate_reports(),
                  results=ExistingResult.query.order_by(ExistingResult.id.desc()).limit(20).all(),
                  stats={'total_results': ExistingResult.query.count(),
                         'avg_relevance': round(float(avg_rel), 4),
                         'avg_response_time': round(float(avg_time), 4),
                         'method': 'TF-IDF Cosine Similarity'})


@router.get('/proposed-model', name='admin.proposed_model')
def proposed_model(request: Request, admin=Depends(require_admin)):
    avg_conf = db.session.query(db.func.avg(ProposedResult.confidence_score)).scalar() or 0
    avg_rel = db.session.query(db.func.avg(ProposedResult.evidence_relevance)).scalar() or 0
    return render(request, 'admin/proposed_model.html', agg=aggregate_reports(),
                  results=ProposedResult.query.order_by(ProposedResult.id.desc()).limit(20).all(),
                  stats={'total_results': ProposedResult.query.count(),
                         'avg_confidence': round(float(avg_conf), 4),
                         'avg_evidence_relevance': round(float(avg_rel), 4),
                         'verified_count': ProposedResult.query.filter_by(verification_status='Verified').count(),
                         'method': 'BM25 + LLM generation + verification'})


@router.get('/accuracy', name='admin.accuracy')
def accuracy(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/accuracy.html', agg=aggregate_reports())


@router.get('/comparison', name='admin.comparison')
def comparison(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/comparison.html', agg=aggregate_reports(), bench=run_benchmark())


@router.get('/betterment', name='admin.betterment')
def betterment(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/betterment.html', agg=aggregate_reports())


@router.get('/reports', name='admin.reports')
def reports(request: Request, admin=Depends(require_admin)):
    return render(request, 'admin/reports.html', agg=aggregate_reports(),
                  total_questions=QASession.query.count(),
                  total_users=User.query.filter_by(role='user').count(),
                  active_users=User.query.filter_by(role='user', status='active').count())
