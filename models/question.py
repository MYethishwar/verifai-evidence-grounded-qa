from extensions import db
from datetime import datetime
import json

class QASession(db.Model):
    __tablename__ = 'qa_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    question = db.Column(db.Text, nullable=False)
    source_file = db.Column(db.String(200))  # if uploaded via file
    status = db.Column(db.String(20), default='processing')  # processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    existing_result = db.relationship('ExistingResult', backref='session', uselist=False, cascade='all, delete-orphan')
    proposed_result = db.relationship('ProposedResult', backref='session', uselist=False, cascade='all, delete-orphan')
    metrics = db.relationship('Metrics', backref='session', uselist=False, cascade='all, delete-orphan')
    feedback = db.relationship('Feedback', backref='session', uselist=False, cascade='all, delete-orphan')

class ExistingResult(db.Model):
    __tablename__ = 'existing_results'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('qa_sessions.id'))
    answer = db.Column(db.Text)
    retrieved_docs = db.Column(db.Text)  # JSON list of doc titles
    relevance_score = db.Column(db.Float, default=0.0)
    response_time = db.Column(db.Float, default=0.0)  # seconds
    method = db.Column(db.String(50), default='TF-IDF')  # TF-IDF or BM25
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProposedResult(db.Model):
    __tablename__ = 'proposed_results'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('qa_sessions.id'))
    answer = db.Column(db.Text)
    confidence_score = db.Column(db.Float, default=0.0)
    evidence_relevance = db.Column(db.Float, default=0.0)
    evidence_coverage = db.Column(db.Float, default=0.0)
    verification_status = db.Column(db.String(30), default='Insufficient Evidence')
    response_time = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    evidences = db.relationship('Evidence', backref='proposed_result', lazy=True, cascade='all, delete-orphan')

class Evidence(db.Model):
    __tablename__ = 'evidences'
    id = db.Column(db.Integer, primary_key=True)
    proposed_result_id = db.Column(db.Integer, db.ForeignKey('proposed_results.id'))
    passage_id = db.Column(db.Integer, db.ForeignKey('passages.id'), nullable=True)
    relevance_score = db.Column(db.Float, default=0.0)
    passage_text = db.Column(db.Text)
    source_title = db.Column(db.Text)
    source_journal = db.Column(db.String(200))
    source_year = db.Column(db.String(10))
    source_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    __tablename__ = 'feedbacks'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('qa_sessions.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    rating = db.Column(db.Integer, default=5)  # 1-5
    comment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
