from extensions import db
from datetime import datetime

class Metrics(db.Model):
    __tablename__ = 'metrics'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('qa_sessions.id'))
    # QA Metrics
    exact_match_existing = db.Column(db.Float, default=0.0)
    exact_match_proposed = db.Column(db.Float, default=0.0)
    token_f1_existing = db.Column(db.Float, default=0.0)
    token_f1_proposed = db.Column(db.Float, default=0.0)
    accuracy_existing = db.Column(db.Float, default=0.0)
    accuracy_proposed = db.Column(db.Float, default=0.0)
    # Retrieval Metrics
    precision_k_existing = db.Column(db.Float, default=0.0)
    precision_k_proposed = db.Column(db.Float, default=0.0)
    recall_k_existing = db.Column(db.Float, default=0.0)
    recall_k_proposed = db.Column(db.Float, default=0.0)
    mrr_existing = db.Column(db.Float, default=0.0)
    mrr_proposed = db.Column(db.Float, default=0.0)
    ndcg_existing = db.Column(db.Float, default=0.0)
    ndcg_proposed = db.Column(db.Float, default=0.0)
    # Evidence Metrics
    evidence_relevance_existing = db.Column(db.Float, default=0.0)
    evidence_relevance_proposed = db.Column(db.Float, default=0.0)
    evidence_coverage = db.Column(db.Float, default=0.0)
    verification_accuracy = db.Column(db.Float, default=0.0)
    # Time Metrics
    response_time_existing = db.Column(db.Float, default=0.0)
    response_time_proposed = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemReport(db.Model):
    __tablename__ = 'system_reports'
    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.String(50))  # daily, weekly, monthly
    total_users = db.Column(db.Integer, default=0)
    active_users = db.Column(db.Integer, default=0)
    total_questions = db.Column(db.Integer, default=0)
    total_documents = db.Column(db.Integer, default=0)
    avg_accuracy_existing = db.Column(db.Float, default=0.0)
    avg_accuracy_proposed = db.Column(db.Float, default=0.0)
    avg_f1_existing = db.Column(db.Float, default=0.0)
    avg_f1_proposed = db.Column(db.Float, default=0.0)
    avg_verification_accuracy = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AnswerReport(db.Model):
    """Full measured report for one question (see services/evaluation_service.py)."""
    __tablename__ = 'answer_reports'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('qa_sessions.id'), unique=True)
    data = db.Column(db.Text)            # JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
