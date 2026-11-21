import json
import os
import random
import re

import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename

from config import Config
from extensions import db
from models.chat import ChatMessage
from models.question import Evidence, ExistingResult, Feedback, ProposedResult, QASession
from models.report import AnswerReport, Metrics
from services import chat_service, evaluation_service, knowledge_base, llm_client
from services.ai_service import generate_answer_ex
from services.domain_guard import check_domain
from services.retrieval_service import build_index_from_db, existing_retrieve, proposed_retrieve
from services.verification_service import verify_answer
from web import flash, redirect, render, require_user, url_for_name

router = APIRouter(prefix='/user')

GENERAL_KNOWLEDGE_STATUS = 'General Knowledge'


# --------------------------------------------------------------------------
# Question pipeline
# --------------------------------------------------------------------------

def retrieve_both(question_text):
    """Search with both systems (milliseconds). Returns everything later steps need."""
    build_index_from_db()
    ex_passages, ex_time = existing_retrieve(question_text, top_k=evaluation_service.K)
    pr_passages, pr_time = proposed_retrieve(question_text, top_k=evaluation_service.K)
    if ex_passages:
        ex_answer, ex_relevance = ex_passages[0].get('content', ''), float(ex_passages[0].get('score', 0.5))
    else:
        ex_answer = f"No direct matching extractive text found in the indexed dataset for: '{question_text}'"
        ex_relevance = 0.0
    return {'ex_passages': ex_passages, 'ex_time': ex_time, 'ex_answer': ex_answer,
            'ex_relevance': ex_relevance, 'pr_passages': pr_passages, 'pr_time': pr_time}


def save_answer(user_id, question_text, ret, gen):
    """
    Verify the answer, measure both systems and store everything in ONE short transaction.
    (SQLite allows a single writer, so nothing slow happens while the transaction is open.)
    Returns the new QASession id, or None on failure.
    """
    try:
        pr_passages = gen.get('evidence') if gen.get('evidence') is not None else ret['pr_passages']
        pr_answer = gen['answer']
        if gen['source'] == 'evidence' and pr_passages:
            ver_result = verify_answer(pr_answer, pr_passages)
        else:
            pr_passages = []
            ver_result = {
                'verification_status': GENERAL_KNOWLEDGE_STATUS if gen['source'] == 'knowledge'
                                       else 'Insufficient Evidence',
                'confidence_score': 0.0, 'evidence_relevance': 0.0, 'evidence_coverage': 0.0,
            }
        ex_passages = ret['ex_passages']
        report = evaluation_service.answer_report(
            question_text, ret['ex_answer'] if ex_passages else '', ex_passages, ret['ex_time'],
            pr_answer, pr_passages, ret['pr_time'])
        report['verification'] = {k: ver_result[k] for k in
                                  ('verification_status', 'confidence_score', 'evidence_coverage')}
        report['answer_source'] = gen['source']
        report['model'] = gen.get('method')

        qa_session = QASession(user_id=user_id, question=question_text, status='completed')
        db.session.add(qa_session)
        db.session.flush()
        db.session.add(ExistingResult(
            session_id=qa_session.id, answer=ret['ex_answer'],
            retrieved_docs=json.dumps([p.get('title', '') for p in ex_passages[:3]]),
            relevance_score=ret['ex_relevance'], response_time=ret['ex_time'], method='TF-IDF Cosine Similarity'))
        proposed_result = ProposedResult(
            session_id=qa_session.id, answer=pr_answer,
            confidence_score=ver_result['confidence_score'],
            evidence_relevance=ver_result['evidence_relevance'],
            evidence_coverage=ver_result['evidence_coverage'],
            verification_status=ver_result['verification_status'], response_time=ret['pr_time'])
        db.session.add(proposed_result)
        db.session.flush()
        for ev_p in pr_passages[:6]:
            db.session.add(Evidence(
                proposed_result_id=proposed_result.id, passage_id=ev_p.get('id'),
                relevance_score=float(ev_p.get('score', 0.0) or 0.0),
                passage_text=ev_p.get('content', '')[:800],
                source_title=ev_p.get('title', ''), source_journal=ev_p.get('journal', ''),
                source_year=str(ev_p.get('year', '') or ''), source_url=ev_p.get('source_url', '')))
        db.session.add(Metrics(session_id=qa_session.id,
                               **evaluation_service.to_metrics_row(report, ver_result['confidence_score'])))
        db.session.add(AnswerReport(session_id=qa_session.id, data=json.dumps(report)))
        db.session.commit()
        return qa_session.id
    except Exception as e:
        db.session.rollback()
        print(f'QA error: {e}')
        return None


def process_question(user, question_text, history=None, greeted=False):
    """Non-streaming pipeline (batch evaluation): retrieve, generate, verify, store."""
    try:
        ret = retrieve_both(question_text)
        gen = generate_answer_ex(question_text, ret['pr_passages'], history=history,
                                 user_name=chat_service._first_name(user), greeted=greeted)
    except Exception as e:
        print(f'QA error: {e}')
        return None
    return save_answer(user.id, question_text, ret, gen)


def _own_session(user, session_id):
    return QASession.query.filter_by(id=session_id, user_id=user.id).first_or_404()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

@router.get('/', name='user.dashboard')
def dashboard(request: Request, user=Depends(require_user)):
    my_sessions = (QASession.query.filter_by(user_id=user.id)
                   .order_by(QASession.created_at.desc()).limit(8).all())
    total_questions = QASession.query.filter_by(user_id=user.id).count()
    statuses = [r[0] for r in db.session.query(ProposedResult.verification_status)
                .join(QASession).filter(QASession.user_id == user.id).all()]
    answered = len(statuses) or 1
    verified = sum(1 for s in statuses if s == 'Verified')
    grounded = sum(1 for s in statuses if s in ('Verified', 'Partially Verified'))
    lat = []
    for (raw,) in (db.session.query(AnswerReport.data)
                   .join(QASession, QASession.id == AnswerReport.session_id)
                   .filter(QASession.user_id == user.id).all()):
        try:
            lat.append(json.loads(raw)['verifai']['latency_ms'])
        except Exception:
            pass
    lat.sort()
    groups, topics = knowledge_base.load()
    picks = random.sample(topics, min(10, len(topics)))
    starters = [(t['title'], t['sample_questions'][0]) for t in picks[:4] if t.get('sample_questions')]
    hero_questions = [t['sample_questions'][-1] for t in picks if t.get('sample_questions')][:8]
    hero_topics = [{'title': re.sub(r'\s*\(.*?\)', '', t['title']), 'id': t['id'],
                    'question': t['sample_questions'][0] if t.get('sample_questions') else t['title']}
                   for t in picks[4:10]]
    return render(request, 'user/dashboard.html',
                  sessions=my_sessions, total_questions=total_questions, verified_answers=verified,
                  stats={
                      'grounded_rate': grounded / answered if statuses else None,
                      'verified_rate': verified / answered if statuses else None,
                      'median_latency_ms': lat[len(lat) // 2] if lat else None,
                      'measured': len(lat), 'kb_topics': len(topics), 'kb_groups': len(groups),
                  },
                  starters=starters, hero_questions=hero_questions, hero_topics=hero_topics)


@router.get('/question', name='user.question')
def question_page(request: Request, user=Depends(require_user)):
    conv = chat_service.get_conversation(user.id, request.session.get('chat_conversation_id'), create=False)
    messages = [chat_service.serialize_message(m) for m in conv.messages] if conv else []
    return render(request, 'user/question.html', chat_messages=messages, llm_label=llm_client.provider_label())


@router.get('/knowledge-base', name='user.knowledge_base_page')
def knowledge_base_page(request: Request, user=Depends(require_user)):
    """Browse the curated knowledge base: every topic VerifAI can answer from, with sample questions."""
    catalog = knowledge_base.catalog()
    stats = {
        'groups': len(catalog),
        'topics': sum(len(g['topics']) for g in catalog),
        'sections': sum(len(t['sections']) for g in catalog for t in g['topics']),
        'questions': sum(len(t['sample_questions']) for g in catalog for t in g['topics']),
    }
    return render(request, 'user/knowledge_base.html', catalog=catalog, stats=stats)


@router.get('/benchmark', name='user.benchmark_page')
def benchmark_page(request: Request, refresh: str = '', user=Depends(require_user)):
    """Labelled retrieval benchmark: baseline (TF-IDF) vs VerifAI (BM25) on known-answer questions."""
    from services.retrieval_service import _index_state
    result = evaluation_service.run_benchmark(force=refresh == '1')
    return render(request, 'user/benchmark.html', b=result,
                  skipped=_index_state.get('skipped_question_only', 0))


@router.get('/existing-result/{session_id}', name='user.existing_result')
def existing_result(request: Request, session_id: int, user=Depends(require_user)):
    return render(request, 'user/existing_result.html', session=_own_session(user, session_id))


@router.get('/proposed-result/{session_id}', name='user.proposed_result')
def proposed_result(request: Request, session_id: int, user=Depends(require_user)):
    return render(request, 'user/proposed_result.html', session=_own_session(user, session_id))


@router.get('/comparison/{session_id}', name='user.comparison')
def comparison(request: Request, session_id: int, user=Depends(require_user)):
    qa_session = _own_session(user, session_id)
    row = AnswerReport.query.filter_by(session_id=session_id).first()
    report = None
    if row and row.data:
        try:
            report = json.loads(row.data)
        except Exception:
            report = None
    return render(request, 'user/comparison.html', session=qa_session,
                  metrics=Metrics.query.filter_by(session_id=session_id).first(),
                  report=report, aspects=list(evaluation_service.ASPECTS.keys()))


@router.get('/feedback/{session_id}', name='user.feedback')
def feedback_form(request: Request, session_id: int, user=Depends(require_user)):
    return render(request, 'user/feedback.html', session=_own_session(user, session_id),
                  existing_feedback=Feedback.query.filter_by(session_id=session_id).first())


@router.post('/feedback/{session_id}')
def feedback(request: Request, session_id: int, rating: int = Form(5), comment: str = Form(''),
             user=Depends(require_user)):
    _own_session(user, session_id)
    fb = Feedback.query.filter_by(session_id=session_id).first()
    if fb:
        fb.rating, fb.comment = rating, comment.strip()
    else:
        db.session.add(Feedback(session_id=session_id, user_id=user.id, rating=rating, comment=comment.strip()))
    db.session.commit()
    flash(request, 'Thank you for your feedback!', 'success')
    return redirect(url_for_name(request, 'user.dashboard'))


# --------------------------------------------------------------------------
# Conversation API (Ask VerifAI page)
# --------------------------------------------------------------------------

def _start_turn(request, user, message):
    """Fast, synchronous part of a chat turn: conversation, context window, routing."""
    conv = chat_service.get_conversation(user.id, request.session.get('chat_conversation_id'))
    db.session.commit()                          # never hold a write lock during model calls
    request.session['chat_conversation_id'] = conv.id
    history = chat_service.build_history(conv)   # context window, taken before this turn
    route = chat_service.classify(message, conv)
    db.session.add(ChatMessage(conversation_id=conv.id, role='user', content=message,
                               standalone_question=route.get('standalone')))
    db.session.commit()
    return conv, history, route


def _finish_turn(conv, route, user, reply=None, kind=None, qa_session_id=None):
    """Store the assistant message for this turn and return it serialised for the chat UI."""
    meta, standalone = {}, route.get('standalone')
    if qa_session_id is not None or kind is None:
        qa = db.session.get(QASession, qa_session_id) if qa_session_id else None
        if not qa or not qa.proposed_result:
            reply, kind = ("Sorry, something went wrong while I was working on that. "
                           "Could you try asking again?"), 'error'
        else:
            pr = qa.proposed_result
            reply = pr.answer
            kind = ('answer' if pr.evidences else
                    'knowledge' if pr.verification_status == GENERAL_KNOWLEDGE_STATUS else 'not_covered')
            meta = {
                'status': pr.verification_status,
                'confidence': round(pr.confidence_score or 0, 4),
                'sources': [{'title': e.source_title, 'journal': e.source_journal,
                             'year': e.source_year, 'url': e.source_url} for e in pr.evidences[:6]],
                'model': llm_client.provider_label(),
            }
            conv.last_topic = standalone
            if not conv.title or conv.title == 'New conversation':
                conv.title = standalone[:200]
    if standalone and standalone != route.get('question'):
        meta['interpreted_as'] = standalone
    bot_msg = ChatMessage(conversation_id=conv.id, role='assistant', content=reply, kind=kind,
                          standalone_question=standalone, qa_session_id=qa_session_id, meta=json.dumps(meta))
    db.session.add(bot_msg)
    db.session.commit()
    return chat_service.serialize_message(bot_msg)


def _validate(message):
    if not message:
        return JSONResponse({'success': False, 'error': 'Please type a message.'}, status_code=400)
    if len(message) > 2000:
        return JSONResponse({'success': False, 'error': 'Please keep messages under 2,000 characters.'}, status_code=400)
    return None


def _quick_reply(route, message, user):
    """Small talk and out-of-scope questions are answered instantly, without the model."""
    if route['intent'] == 'smalltalk':
        return chat_service.smalltalk_reply(route['smalltalk_kind'], message, user, []), 'smalltalk'
    if not route['scope']['in_scope']:
        return chat_service.out_of_scope_reply(route['question'] or message, user), 'out_of_scope'
    return None, None


@router.post('/chat', name='user.chat')
def chat(request: Request, payload: dict = Body(default={}), user=Depends(require_user)):
    """One conversational turn, answered in full (non-streaming clients)."""
    message = str(payload.get('message') or '').strip()
    invalid = _validate(message)
    if invalid:
        return invalid
    conv, history, route = _start_turn(request, user, message)
    reply, kind = _quick_reply(route, message, user)
    if kind:
        return {'success': True, 'message': _finish_turn(conv, route, user, reply, kind)}
    sid = process_question(user, route['standalone'], history=history, greeted=route['greeted'])
    return {'success': True, 'message': _finish_turn(conv, route, user, qa_session_id=sid)}


@router.post('/chat/stream', name='user.chat_stream')
def chat_stream(request: Request, payload: dict = Body(default={}), user=Depends(require_user)):
    """
    One conversational turn, streamed as newline-delimited JSON events:
      {"type": "status", "text": ...}   progress ("Searching...", "Writing...", "Verifying...")
      {"type": "token",  "text": ...}   answer text as the model writes it
      {"type": "final",  "message": {...}}  the stored message with sources and verification
    """
    from fastapi.responses import StreamingResponse
    from models.chat import Conversation
    from models.user import User
    from services.ai_service import MAX_ANSWER_TOKENS, finish_answer, plan_answer, _rule_based_answer

    message = str(payload.get('message') or '').strip()
    invalid = _validate(message)
    if invalid:
        return invalid
    conv, history, route = _start_turn(request, user, message)
    reply, kind = _quick_reply(route, message, user)
    if kind:
        final = _finish_turn(conv, route, user, reply, kind)
        return StreamingResponse(iter([json.dumps({'type': 'final', 'message': final}) + '\n']),
                                 media_type='application/x-ndjson')

    user_id, conv_id, user_name = user.id, conv.id, chat_service._first_name(user)
    question = route['standalone']

    import queue
    import threading
    events_q = queue.Queue()

    def put(**kw):
        events_q.put(json.dumps(kw) + '\n')

    def worker():
        # Runs in its own thread with its own database session, so the answer pipeline is
        # independent of how the web server iterates the streaming response.
        with db.request_scope():
            try:
                put(type='status', text='Searching the knowledge base…')
                ret = retrieve_both(question)
                plan = plan_answer(question, ret['pr_passages'], history, user_name, route['greeted'])
                if 'final' in plan:
                    gen = plan['final']
                else:
                    put(type='status', text='Writing the answer from the sources…')
                    chunks = []
                    try:
                        for piece in llm_client.chat_stream(plan['messages'], max_tokens=MAX_ANSWER_TOKENS,
                                                            temperature=0.3):
                            chunks.append(piece)
                            put(type='token', text=piece)
                        gen = finish_answer(plan, ''.join(chunks).strip()) if chunks else \
                            _rule_based_answer(question, plan['evidence'])
                    except Exception as e:
                        print(f'LLM stream failed: {e}')
                        gen = _rule_based_answer(question, plan['evidence'])
                put(type='status', text='Verifying against the sources…')
                sid = save_answer(user_id, question, ret, gen)
                conv2 = db.session.get(Conversation, conv_id)
                final = _finish_turn(conv2, route, db.session.get(User, user_id), qa_session_id=sid)
                put(type='final', message=final)
            except Exception as e:
                print(f'Chat stream error: {e}')
                put(type='error', text='Sorry, something went wrong while I was working on that. Please try again.')
            finally:
                events_q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def events():
        while True:
            item = events_q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(events(), media_type='application/x-ndjson',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@router.post('/chat/new', name='user.chat_new')
def chat_new(request: Request, user=Depends(require_user)):
    request.session.pop('chat_conversation_id', None)
    return {'success': True}


# --------------------------------------------------------------------------
# Batch evaluation
# --------------------------------------------------------------------------

def _out_of_scope(scope):
    return {'success': False, 'out_of_scope': True, 'message': scope['message']}


def _read_questions(file_path, ext):
    """First column that looks like questions, as a DataFrame column, or None."""
    df = None
    if ext == 'csv':
        for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']:
            try:
                df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
                break
            except Exception:
                continue
        if df is None:
            df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
    elif ext in ('xlsx', 'xls'):
        df = pd.read_excel(file_path)
    elif ext == 'txt':
        content = ''
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        df = pd.DataFrame({'question': [l.strip() for l in content.split('\n') if l.strip()]})
    elif ext == 'json':
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                df = pd.DataFrame(data if isinstance(data, list) else [data])
                break
            except Exception:
                continue
    return df


@router.get('/upload-test', name='user.upload_test')
def upload_test_page(request: Request, user=Depends(require_user)):
    return render(request, 'user/upload_test.html')


@router.post('/upload-test')
def upload_test(request: Request, question: str = Form(''), test_file: UploadFile | None = File(None),
                user=Depends(require_user)):
    question = question.strip()
    has_file = test_file is not None and bool(test_file.filename)
    if question and not has_file:
        scope = check_domain(question)
        if not scope['in_scope']:
            return _out_of_scope(scope)
        session_id = process_question(user, question)
        return {'success': True, 'session_id': session_id} if session_id else {'success': False, 'error': 'Processing failed'}
    if not has_file:
        return {'success': False, 'error': 'No question or file provided'}

    filename = secure_filename(test_file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(Config.UPLOAD_FOLDER, 'test_' + filename)
    with open(file_path, 'wb') as out:
        out.write(test_file.file.read())
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext not in ('csv', 'xlsx', 'xls', 'txt', 'json'):
        return {'success': False, 'error': 'Unsupported file format'}
    try:
        df = _read_questions(file_path, ext)
        if df is None or len(df) == 0:
            return {'success': False, 'error': 'No readable data in file'}
        q_col = next((c for c in df.columns
                      if any(k in str(c).lower() for k in ['question', 'query', 'title', 'text', 'prompt'])), df.columns[0])
        first_q = str(df[q_col].iloc[0]).strip()
        if not first_q:
            return {'success': False, 'error': 'Could not extract valid question from file'}
        scope = check_domain(first_q)
        if not scope['in_scope']:
            return _out_of_scope(scope)
        session_id = process_question(user, first_q)
        return {'success': True, 'session_id': session_id} if session_id else {'success': False, 'error': 'Processing failed'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
