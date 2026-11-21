import time
import json
import functools
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

GENERIC_TERMS = {
    'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'the', 'and', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'a', 'an', 'as', 'at', 'by', 'of', 'or', 'so', 'then', 'there', 'this', 'that', 'these', 'those', 'can', 'could',
    'tell', 'explain', 'describe', 'details', 'give', 'list', 'show',
    'structure', 'structures', 'function', 'functions', 'functional', 'process', 'processes',
    'overview', 'definition', 'definitions', 'detail', 'details', 'component', 'components',
    'mechanism', 'mechanisms', 'role', 'roles', 'feature', 'features', 'type', 'types', 'kind', 'kinds',
    'effect', 'effects', 'purpose', 'purposes', 'meaning', 'meanings', 'concept', 'concepts',
    'explanation', 'explanations', 'description', 'descriptions', 'characteristic', 'characteristics',
    'risk', 'risks', 'symptom', 'symptoms', 'cause', 'causes', 'treatment', 'treatments',
    'disease', 'diseases', 'condition', 'conditions', 'sign', 'signs', 'management',
    'factor', 'factors', 'pathology', 'pathophysiology', 'biomarkers', 'diagnostics',
    'therapeutics', 'clinical', 'presentation', 'syndrome', 'rate', 'rates',
    'progression', 'progressive', 'strategy', 'strategies', 'target', 'targets',
    'screening', 'guidelines', 'acute', 'chronic', 'study', 'studies', 'review', 'summary',
    # everyday filler that shouldn't be required to appear in a passage
    'common', 'commonly', 'option', 'options', 'available', 'main', 'best', 'usual', 'usually',
    'typical', 'typically', 'often', 'different', 'important', 'key', 'major', 'general',
    'help', 'helps', 'work', 'works', 'happen', 'happens', 'affect', 'affects', 'used', 'use',
    'more', 'most', 'some', 'many', 'much', 'any', 'all', 'other', 'others', 'like', 'such',
    'its', 'it', 'they', 'them', 'their', 'also', 'please', 'know', 'about', 'treated', 'caused',
    'diagnosed', 'prevent', 'prevented', 'prevention', 'manage', 'managed', 'way', 'ways', 'thing', 'things'
}

# --------------------------------------------------------------------------
# Subject-aware topic filter
#
# A passage must mention the question's SUBJECT (the disease, gene, drug or organ it is
# about, e.g. "asthma", "brca1", "hepatitis"), but not every descriptive word
# ("inflammatory", "pathways", "contribute"). This keeps off-topic protection
# ("lung cancer" never matches the colorectal cancer article) while letting naturally
# phrased questions find the right passages.
# --------------------------------------------------------------------------

# Biomedical words that describe a question rather than name its subject.
SUBJECT_DESCRIPTORS = {
    'inflammatory', 'inflammation', 'immune', 'immunity', 'medical', 'medicine', 'medicines',
    'patient', 'patients', 'health', 'healthcare', 'therapy', 'therapies', 'therapeutic',
    'treatment', 'treatments', 'diagnosis', 'diagnose', 'diagnosed', 'diagnostic', 'prognosis',
    'drug', 'drugs', 'medication', 'medications', 'molecular', 'receptor', 'receptors',
    'protein', 'proteins', 'gene', 'genes', 'genetic', 'genetics', 'biomarker', 'biomarkers',
    'clinical', 'chronic', 'acute', 'pain', 'mortality', 'prevalence', 'incidence', 'infection',
    'infections', 'infectious', 'disorder', 'disorders', 'illness', 'syndrome', 'cure', 'dose',
    'dosage', 'adverse', 'side', 'effect', 'effects', 'pathogenesis', 'etiology', 'aetiology',
    'epidemiology', 'physiology', 'anatomy', 'biochemistry', 'biochemical', 'metabolism',
    'metabolic', 'tissue', 'tissues', 'cell', 'cells', 'blood', 'serum', 'plasma', 'fatigue',
    'fever', 'cough', 'nausea', 'vomiting', 'headache', 'rash', 'inhibitor', 'inhibitors',
    'antibody', 'antibodies', 'hormone', 'hormones', 'vitamin', 'vitamins', 'deficiency',
    'surgery', 'surgical', 'screening', 'vaccine', 'vaccines', 'vaccination', 'remission',
    'relapse', 'complication', 'complications', 'lifestyle', 'pregnancy', 'biomedical',
    'pathway', 'pathways', 'mechanism', 'mechanisms', 'disease', 'diseases', 'condition',
    'conditions', 'symptom', 'symptoms', 'risk', 'risks', 'cause', 'causes', 'tumor', 'tumour',
    'tumors', 'tumours', 'malignant', 'benign', 'metastasis', 'metastatic', 'injury', 'wound',
    'inherited', 'hereditary', 'mutation', 'mutations', 'viral', 'bacterial', 'doctor', 'doctors',
    'physician', 'nurse', 'hospital', 'child', 'children', 'adult', 'adults', 'baby', 'babies',
    'women', 'men', 'elderly', 'early', 'warning', 'signs', 'sign', 'difference', 'different',
    'test', 'tests', 'testing', 'normal', 'level', 'levels', 'result', 'results', 'stage', 'stages',
}

_STEM_SUFFIXES = ('ically', 'ical', 'ation', 'ating', 'ated', 'ions', 'ion', 'ives', 'ive',
                  'atic', 'etic', 'itic', 'otic', 'ing', 'ics', 'ies', 'ic', 'es', 'ed', 's')

_corpus_subject_terms = set()   # document keywords + knowledge-base names, refreshed in build_index_from_db


@functools.lru_cache(maxsize=65536)
def _stems(word):
    """All light stems of a word: 'diabetic' -> {'diabetic', 'diabet', 'diab'}."""
    out = {word}
    for suf in _STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            out.add(word[:-len(suf)])
    return out


@functools.lru_cache(maxsize=262144)
def _same_term(a, b):
    """'asthma'~'asthmatic', 'diabetes'~'diabetic', 'hypertension'~'hypertensive',
    'demyelination'~'demyelinating'; but never 'malar'~'malaria'."""
    if a == b:
        return True
    sa, sb = _stems(a), _stems(b)
    if sa & sb:
        return True
    # A stripped stem may be a short prefix of the other word ('asthm' from 'asthmatic'
    # ~ 'asthma'); a plain word never is ('malar' must not match 'malaria').
    for stems, word, other in ((sa, a, b), (sb, b, a)):
        for s in stems - {word}:
            if len(s) >= 5 and other.startswith(s) and len(other) - len(s) <= 4:
                return True
    return False


_QUALIFIED = [
    (re.compile(r'\btype\s*-?\s*([12])\b', re.I), 'type{0}'),               # type 1 / type 2 (diabetes)
    (re.compile(r'\bhepatitis\s*-?\s*([a-e])\b', re.I), 'hepatitis{0}'),     # hepatitis A-E
]
_QUALIFIED_TOKEN = re.compile(r'^(type[12]|hepatitis[a-e])$')


def normalize_entities(text):
    """Keep qualifiers that tokenisers drop: 'type 2 diabetes' -> 'type 2 type2 diabetes',
    'hepatitis C' -> 'hepatitis C hepatitisc'. Applied identically to both retrievers."""
    text = text or ''
    for rx, fmt in _QUALIFIED:
        text = rx.sub(lambda m: f"{m.group(0)} {fmt.format(m.group(1).lower())}", text)
    return text


def query_subjects(query):
    """Split query keywords into (keywords, subjects). Subjects must appear in a passage."""
    from services.domain_guard import _has_medical_suffix
    words = set(re.findall(r'\b[a-z0-9][a-z0-9\-]{2,}\b', normalize_entities(query).lower()))
    keywords = words - GENERIC_TERMS
    subjects = set()
    for w in keywords:
        if _QUALIFIED_TOKEN.match(w):
            subjects.add(w)
            continue
        if w in SUBJECT_DESCRIPTORS or (_stems(w) - {w}) & SUBJECT_DESCRIPTORS:
            continue
        parts = [w] + w.split('-')          # 'brca1-mutated' -> also 'brca1'
        for part in parts:
            if part in SUBJECT_DESCRIPTORS or len(part) < 3:
                continue
            # A subject is a named entity: a knowledge-base / dataset topic name or alias,
            # or a word with a medical word-form ("glaucoma", "pembrolizumab").
            if (part in _corpus_subject_terms or _has_medical_suffix(part)
                    or any(_same_term(part, t) for t in (_corpus_subject_terms if len(part) >= 5 else ()))):
                subjects.add(part)
                break
    return keywords, subjects


def passage_tokens(passage):
    """Word set of a passage (title + content, hyphen parts included); computed once and cached
    on the passage dict, so topic filtering costs a set lookup per query."""
    tokens = passage.get('_tokens')
    if tokens is None:
        text = normalize_entities(f"{passage.get('title') or ''} {passage.get('content') or ''}").lower()
        words = set(re.findall(r'\b[a-z0-9][a-z0-9\-]{2,}\b', text))
        words |= {piece for t in words if '-' in t for piece in t.split('-') if len(piece) >= 3}
        tokens = passage['_tokens'] = frozenset(words)
    return tokens


def _passage_has_term(term, tokens):
    if term in tokens:
        return True
    return any(_same_term(term, t) for t in tokens if t[:2] == term[:2])


def make_topic_filter(query):
    """Return (keywords, predicate(passage) -> bool) for a query, computed once per query."""
    keywords, subjects = query_subjects(query)
    if not subjects:
        # No recognisable subject: most (>= 60%) of the keywords must appear.
        if not keywords:
            return keywords, (lambda p: True)
        need = max(1, math.ceil(0.6 * len(keywords)))

        def by_keywords(passage):
            tokens = passage_tokens(passage)
            return sum(1 for k in keywords if _passage_has_term(k, tokens)) >= need
        return keywords, by_keywords

    special = subjects & {'dna', 'rna'}
    qualified = {s for s in subjects if _QUALIFIED_TOKEN.match(s)}     # always mandatory
    regular = subjects - special - qualified
    # One subject: it must appear. Several ("rheumatoid arthritis vs osteoarthritis"):
    # at least half of them must appear.
    need = len(regular) if len(regular) <= 1 else math.ceil(len(regular) / 2)

    def predicate(passage):
        if special and not is_passage_topic_relevant(passage, special):
            return False
        tokens = passage_tokens(passage)
        if not all(q in tokens for q in qualified):
            return False
        return sum(1 for s in regular if _passage_has_term(s, tokens)) >= need

    return keywords, predicate


def is_passage_topic_relevant(passage, disease_keywords):
    if not disease_keywords:
        return True
    title = (passage.get('title') or '').lower()
    content = (passage.get('content') or '').lower()
    title_tokens = set(re.findall(r'\b[a-z0-9\-]{3,}\b', title))
    content_tokens = set(re.findall(r'\b[a-z0-9\-]{3,}\b', content))
    
    for kw in disease_keywords:
        if kw == 'dna':
            dna_terms = {'dna', 'deoxyribonucleic', 'double-helix', 'genetics', 'genome', 'chromosomal'}
            if not dna_terms.intersection(title_tokens):
                dna_count = sum(1 for t in ['dna', 'deoxyribonucleic', 'double helix', 'nucleotide', 'base pair', 'chargaff', 'adenine', 'thymine'] if t in content)
                if dna_count < 2:
                    return False
        elif kw == 'rna':
            rna_terms = {'rna', 'ribonucleic', 'mrna', 'trna', 'rrna'}
            if not rna_terms.intersection(title_tokens):
                rna_count = sum(1 for t in ['rna', 'ribonucleic', 'mrna', 'trna', 'rrna', 'uracil'] if t in content)
                if rna_count < 2:
                    return False
        else:
            if kw not in title_tokens and kw not in content_tokens:
                return False
    return True

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
        self.tfidf_matrix = None
        self.passages = []
        self.is_fitted = False

    def fit(self, passages):
        """Fit vectorizer on passages."""
        if not passages:
            return
        self.passages = passages
        texts = [normalize_entities(f"{p.get('title', '')} {p.get('content', '')}") for p in passages]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def retrieve(self, query, top_k=5):
        """Retrieve top-k passages for query with positive topic relevance only."""
        start = time.perf_counter()
        if not self.is_fitted or not self.passages:
            return [], 0.0
        query_vec = self.vectorizer.transform([normalize_entities(query)])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        disease_keywords, topic_ok = make_topic_filter(query)

        # Sort all indices by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        matching_indices = []
        for idx in sorted_indices:
            if scores[idx] > 0.01:
                if topic_ok(self.passages[idx]):
                    matching_indices.append(idx)
            if len(matching_indices) >= top_k:
                break
        
        if not matching_indices:
            return [], max(0.0001, time.perf_counter() - start)
            
        results = []
        for idx in matching_indices:
            score = float(scores[idx])
            p = self.passages[idx].copy()
            p['score'] = max(score, 0.15)
            results.append(p)
            
        elapsed = max(0.0001, time.perf_counter() - start)
        return results, elapsed


STOP_WORDS = GENERIC_TERMS

class BM25Retriever:
    """
    BM25 with an inverted index built once at fit time, so each query only touches
    passages that contain its terms (instead of re-counting every passage per query).
    Scores are identical to the original per-passage implementation.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.passages = []
        self.idf = {}
        self.doc_freqs = []
        self.avgdl = 0
        self.postings = {}      # term -> [(doc_idx, term_freq), ...]
        self.norms = []         # per-doc k1 * (1 - b + b * dl / avgdl)
        self._partial_cache = {}  # unknown query term -> passages with a partial match
        self.is_fitted = False

    def _tokenize(self, text):
        text = normalize_entities(text)
        words = re.findall(r'\b[a-z0-9\-]{2,}\b', text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def fit(self, passages):
        if not passages:
            return
        self.passages = passages
        # Title counted twice: a light BM25F-style field weight (section titles such as
        # "Malaria: Symptoms" are strong evidence of what a passage is about).
        corpus = [self._tokenize(f"{p.get('title', '')} {p.get('title', '')} {p.get('content', '')}")
                  for p in passages]
        self.doc_freqs = corpus
        N = len(corpus)
        self.avgdl = (sum(len(d) for d in corpus) / N if N > 0 else 1) or 1
        postings = {}
        for idx, doc in enumerate(corpus):
            tf = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                postings.setdefault(t, []).append((idx, f))
        self.postings = postings
        self.idf = {w: math.log((N - len(pl) + 0.5) / (len(pl) + 0.5) + 1) for w, pl in postings.items()}
        self.norms = [self.k1 * (1 - self.b + self.b * len(d) / self.avgdl) for d in corpus]
        self._partial_cache = {}
        for p in passages:
            passage_tokens(p)   # warm the per-passage word sets used by the topic filter
        self.is_fitted = True

    def _scores(self, query_tokens):
        scores = [0.0] * len(self.passages)
        k1p1 = self.k1 + 1
        for term in query_tokens:            # duplicates count, as in the original
            if term in self.idf:
                idf = self.idf[term]
                for idx, f in self.postings[term]:
                    scores[idx] += idf * f * k1p1 / (f + self.norms[idx] + 1e-10)
            elif len(term) >= 4:
                # Substring/partial match bonus: +0.5 per passage containing a related token
                hit_docs = self._partial_cache.get(term)
                if hit_docs is None:
                    hit_docs = set()
                    for word, pl in self.postings.items():
                        if term in word or word in term:
                            hit_docs.update(idx for idx, _ in pl)
                    self._partial_cache[term] = hit_docs
                for idx in hit_docs:
                    scores[idx] += 0.5
        return scores

    def retrieve(self, query, top_k=5):
        start = time.perf_counter()
        if not self.is_fitted or not self.passages:
            return [], 0.0

        query_tokens = self._tokenize(query)

        # Extract disease/topic-specific keywords from the query for strict filtering
        disease_keywords, topic_ok = make_topic_filter(query)

        scores = self._scores(query_tokens)

        max_score = max(scores) if scores else 0
        # Sort all passages by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # ── Strict topic filtering: only include passages that pass topic relevance check ──
        filtered_indices = []
        for idx in sorted_indices:
            if scores[idx] <= 0:
                break                         # sorted, so nothing further can match
            if topic_ok(self.passages[idx]):
                filtered_indices.append(idx)
                if len(filtered_indices) >= top_k:
                    break

        # If strict topic keywords were present but none matched, do not return off-topic passages
        if disease_keywords and not filtered_indices:
            return [], max(0.0001, time.perf_counter() - start)

        if not filtered_indices:
            filtered_indices = sorted_indices[:top_k]

        results = []
        for idx in filtered_indices:
            p = self.passages[idx].copy()
            raw_score = float(scores[idx])
            if max_score > 0 and raw_score > 0:
                p['score'] = raw_score
                p['score_norm'] = min(0.98, max(0.40, raw_score / (max_score + 1e-5)))
            else:
                p['score'] = 0.35
                p['score_norm'] = 0.45
            results.append(p)

        elapsed = max(0.0001, time.perf_counter() - start)
        return results, elapsed


# Global retrievers (in-memory for the session)
_tfidf_retriever = TFIDFRetriever()
_bm25_retriever = BM25Retriever()

_index_state = {'key': None, 'count': 0, 'skipped_question_only': 0}

_QUESTION_START = re.compile(
    r'^(record\s+\d+\s+)?(what|how|which|why|when|who|whom|whose|where|is|are|can|does|do|should|could)\b', re.I)


def is_question_only(text):
    """True for passages that contain just a question (no informational content)."""
    t = (text or '').strip()
    if not t:
        return True
    words = len(t.split())
    if t.endswith('?') and words < 40 and t.count('.') == 0:
        return True
    # Starts like a question and contains no complete sentence (e.g. a test question stored
    # twice as "title + text"): informational passages always contain full sentences.
    return bool(_QUESTION_START.match(t)) and words < 80 and '.' not in t.rstrip('.')


def build_index_from_db(force=False):
    """Build the retrieval index from uploaded datasets plus the curated knowledge base.
    Skips the rebuild when neither the database passages nor the knowledge base changed."""
    try:
        from models.document import Passage, Document
        from services import knowledge_base
        from extensions import db

        key = (Document.query.count(), Passage.query.count(),
               db.session.query(db.func.max(Passage.id)).scalar(), knowledge_base._dir_mtime())
        if not force and key == _index_state['key'] and _bm25_retriever.is_fitted:
            return _index_state['count']

        all_docs = Document.query.all()

        # Single-word document keywords ("metformin", "troponin", "brca1") and knowledge-base
        # topic names count as subjects.
        _corpus_subject_terms.clear()
        _corpus_subject_terms.update(t for t in knowledge_base.subject_terms()
                                     if t not in GENERIC_TERMS and t not in SUBJECT_DESCRIPTORS)
        for doc in all_docs:
            for kw in (doc.keywords or '').split(','):
                kw = kw.strip().lower()
                if kw and ' ' not in kw and len(kw) >= 3 and kw not in GENERIC_TERMS \
                        and kw not in SUBJECT_DESCRIPTORS:
                    _corpus_subject_terms.add(kw)

        passages_list = []
        
        for doc in all_docs:
            doc_passages = Passage.query.filter_by(document_id=doc.id).all()
            if doc_passages:
                for p in doc_passages:
                    passages_list.append({
                        'id': p.id,
                        'content': p.content or doc.abstract or doc.title or '',
                        'document_id': doc.id,
                        'title': doc.title or 'Biomedical Article',
                        'journal': doc.journal or '',
                        'year': doc.year or '',
                        'authors': doc.authors or '',
                        'source_url': doc.source_url or '',
                        'doi': doc.doi or '',
                    })
            else:
                # Use document abstract and title directly as passage
                text = (doc.abstract or doc.title or '').strip()
                if text:
                    passages_list.append({
                        'id': doc.id,
                        'content': text,
                        'document_id': doc.id,
                        'title': doc.title or 'Biomedical Article',
                        'journal': doc.journal or '',
                        'year': doc.year or '',
                        'authors': doc.authors or '',
                        'source_url': doc.source_url or '',
                        'doi': doc.doi or '',
                    })

        # A passage that is only a question carries no evidence (e.g. a test-question file
        # uploaded as a dataset); keep it out of the index so it cannot outrank real content.
        skipped = [p for p in passages_list if is_question_only(p.get('content', ''))]
        if skipped:
            passages_list = [p for p in passages_list if not is_question_only(p.get('content', ''))]
            _index_state['skipped_question_only'] = len(skipped)

        # Curated knowledge base: one passage per topic section
        passages_list.extend(knowledge_base.passages())

        if passages_list:
            _tfidf_retriever.fit(passages_list)
            _bm25_retriever.fit(passages_list)
            _index_state.update(key=key, count=len(passages_list))
            return len(passages_list)
        return 0
    except Exception as e:
        print(f'Index build error: {e}')
        return 0

def existing_retrieve(query, top_k=5):
    return _tfidf_retriever.retrieve(query, top_k)

def proposed_retrieve(query, top_k=5):
    return _bm25_retriever.retrieve(query, top_k)

