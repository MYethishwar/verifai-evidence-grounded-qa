from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    try:
        vect = TfidfVectorizer(stop_words='english').fit_transform([text1, text2])
        sim = cosine_similarity(vect[0:1], vect[1:2])[0][0]
        return float(sim)
    except:
        words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
        if not words1 or not words2:
            return 0.0
        overlap = words1.intersection(words2)
        return len(overlap) / max(len(words1), len(words2))

def verify_answer(answer, evidence_passages):
    """
    Verify an answer against its evidence passages. All values are measured:

      evidence_coverage  - share of answer sentences whose content words are mostly (>=50%)
                           found in the evidence ("supported sentences")
      evidence_relevance - mean normalised retrieval score of the evidence passages
      confidence_score   - 0.7 * evidence_coverage + 0.3 * best answer-evidence similarity

    Status: Verified (>= 0.75), Partially Verified (>= 0.40), otherwise Insufficient Evidence.
    """
    from services.evaluation_service import evidence_support
    if not answer or not answer.strip() or not evidence_passages:
        return {'verification_status': 'Insufficient Evidence', 'confidence_score': 0.0,
                'evidence_relevance': 0.0, 'evidence_coverage': 0.0}

    coverage, _, _ = evidence_support(answer, evidence_passages)
    sims = [compute_similarity(answer, ev.get('content', ev.get('passage_text', '')))
            for ev in evidence_passages]
    best_sim = max(sims) if sims else 0.0
    scores = [float(ev.get('score_norm', ev.get('score', 0.0)) or 0.0) for ev in evidence_passages]
    relevance = min(1.0, sum(scores) / len(scores)) if scores else 0.0

    confidence = 0.7 * coverage + 0.3 * min(1.0, best_sim)
    if confidence >= 0.75:
        status = 'Verified'
    elif confidence >= 0.40:
        status = 'Partially Verified'
    else:
        status = 'Insufficient Evidence'
    return {
        'verification_status': status,
        'confidence_score': round(confidence, 4),
        'evidence_relevance': round(relevance, 4),
        'evidence_coverage': round(coverage, 4),
    }
