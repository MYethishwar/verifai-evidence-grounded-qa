from extensions import db
from datetime import datetime


class Conversation(db.Model):
    """One chat thread on the Ask a Question page."""
    __tablename__ = 'conversations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200))
    # Last standalone biomedical question; keeps follow-ups working after
    # older turns have slid out of the context window.
    last_topic = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('ChatMessage', backref='conversation', lazy=True,
                               cascade='all, delete-orphan', order_by='ChatMessage.id')


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)            # 'user' | 'assistant'
    content = db.Column(db.Text, nullable=False)
    # assistant: 'smalltalk' | 'answer' | 'knowledge' | 'out_of_scope' | 'error'
    kind = db.Column(db.String(20), default='message')
    standalone_question = db.Column(db.Text)                  # follow-up rewritten with context
    qa_session_id = db.Column(db.Integer, db.ForeignKey('qa_sessions.id'), nullable=True)
    meta = db.Column(db.Text)                                  # JSON: sources, status, model...
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
