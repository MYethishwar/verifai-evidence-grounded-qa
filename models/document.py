from extensions import db
from datetime import datetime
import json

class Dataset(db.Model):
    __tablename__ = 'datasets'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)
    records_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='uploaded')  # uploaded, preprocessed, indexed
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    documents = db.relationship('Document', backref='dataset', lazy=True, cascade='all, delete-orphan')

class Document(db.Model):
    __tablename__ = 'documents'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    title = db.Column(db.Text)
    abstract = db.Column(db.Text)
    authors = db.Column(db.Text)
    journal = db.Column(db.String(200))
    year = db.Column(db.String(10))
    pmid = db.Column(db.String(50))
    doi = db.Column(db.String(100))
    source_url = db.Column(db.String(500))
    keywords = db.Column(db.Text)
    preprocessed_text = db.Column(db.Text)
    is_preprocessed = db.Column(db.Boolean, default=False)
    is_indexed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    passages = db.relationship('Passage', backref='document', lazy=True, cascade='all, delete-orphan')

class Passage(db.Model):
    __tablename__ = 'passages'
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'))
    content = db.Column(db.Text, nullable=False)
    passage_index = db.Column(db.Integer, default=0)
    word_count = db.Column(db.Integer, default=0)
    tfidf_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    evidences = db.relationship('Evidence', backref='passage', lazy=True)
