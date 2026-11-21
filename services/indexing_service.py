from services.retrieval_service import build_index_from_db, _tfidf_retriever, _bm25_retriever
from models.document import Document, Passage, Dataset
from services.preprocessing_service import preprocess_text, segment_passages
from extensions import db

def run_preprocessing(dataset_id):
    """Preprocess all documents in a dataset."""
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return {'error': 'Dataset not found'}
    docs = Document.query.filter_by(dataset_id=dataset_id).all()
    processed = 0
    for doc in docs:
        combined = f"{doc.title or ''} {doc.abstract or ''}"
        doc.preprocessed_text = preprocess_text(combined)
        doc.is_preprocessed = True
        processed += 1
    dataset.status = 'preprocessed'
    db.session.commit()
    return {'processed': processed, 'status': 'success'}

def run_passage_segmentation(dataset_id):
    """Segment preprocessed documents into passages."""
    docs = Document.query.filter_by(dataset_id=dataset_id, is_preprocessed=True).all()
    total_passages = 0
    for doc in docs:
        # Remove old passages
        Passage.query.filter_by(document_id=doc.id).delete()
        text = doc.preprocessed_text or f"{doc.title or ''} {doc.abstract or ''}"
        passages = segment_passages(text, max_words=120)
        for i, p_text in enumerate(passages):
            p = Passage(
                document_id=doc.id,
                content=p_text,
                passage_index=i,
                word_count=len(p_text.split())
            )
            db.session.add(p)
        total_passages += len(passages)
    db.session.commit()
    return {'total_passages': total_passages, 'status': 'success'}

def run_indexing():
    """Build TF-IDF and BM25 indexes from all passages in DB."""
    count = build_index_from_db(force=True)
    return {'indexed_passages': count, 'status': 'success'}

def get_index_stats():
    total_passages = Passage.query.count()
    total_docs = Document.query.count()
    total_datasets = Dataset.query.count()
    fitted = _tfidf_retriever.is_fitted
    return {
        'total_passages': total_passages,
        'total_documents': total_docs,
        'total_datasets': total_datasets,
        'index_built': fitted
    }
