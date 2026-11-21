"""
Measured evaluation for VerifAI. Every number shown in the app comes from here and is
computed from real data; nothing is randomised or hard-coded.

1. Per-question answer report (no reference answer needed)
   Retrieval is judged automatically: a retrieved passage counts as relevant when it
   mentions every subject of the question (e.g. "asthma") and at least half of the
   question's key terms (word variants allowed). With that judgement we compute the
   standard IR metrics Precision@k, Recall@k, MRR and nDCG@k for both systems, plus
   answer-level measures (question coverage, evidence support, clinical breadth,
   citations) and measured latency.

2. Labelled benchmark (run_benchmark)
   Questions with known correct sources: the 20 curated test questions
   (datasets/biomedical_qa_test_20.csv) and every sample question in the knowledge
   base. Reports Hit@1, Recall@5, MRR@10, nDCG@5 and latency (p50/p95) for the baseline
   (TF-IDF) and VerifAI (BM25) retrievers, plus scope-guard accuracy on in-scope vs
   out-of-scope questions.
"""

import csv
import math
import os
import re
import statistics
import time

K = 6  # passages retrieved per question

ASPECTS = {
    'Definition': ('is a', 'are a', 'refers to', 'defined', 'condition', 'disease', 'disorder', 'type of'),
    'Causes & risks': ('cause', 'caused', 'risk', 'trigger', 'due to', 'result of', 'factor'),
    'Symptoms': ('symptom', 'sign', 'present', 'pain', 'fatigue', 'fever', 'feel'),
    'Diagnosis': ('diagnos', 'test', 'screen', 'scan', 'blood test', 'biopsy', 'imaging', 'measure'),
    'Treatment': ('treat', 'therapy', 'medication', 'medicine', 'drug', 'surgery', 'manage'),
    'Prevention & outlook': ('prevent', 'vaccin', 'lifestyle', 'outlook', 'prognosis', 'reduce the risk', 'avoid'),
}

_FILLER = {'what', 'which', 'how', 'why', 'when', 'where', 'who', 'does', 'do', 'is', 'are', 'can',
           'the', 'and', 'for', 'with', 'that', 'this', 'there', 'will', 'would', 'should', 'about',
           'tell', 'me', 'explain', 'describe', 'need', 'get', 'have', 'has', 'any', 'into', 'from'}

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')
_WORD = re.compile(r'[a-z0-9][a-z0-9\-]{2,}')


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def _words(text):
    return set(_WORD.findall((text or '').lower()))


def key_terms(question):
    from services.retrieval_service import query_subjects
    keywords, subjects = query_subjects(question)
    terms = {k for k in keywords if k not in _FILLER}
    return terms, subjects


def _has(term, words):
    from services.retrieval_service import _passage_has_term
    return _passage_has_term(term, frozenset(words))


def judge_relevant(passage, terms, subjects):
    """Automatic relevance judgement for one passage (see module docstring)."""
    from services.retrieval_service import passage_tokens
    tokens = passage_tokens(passage)
    if subjects and not all(_has(s, tokens) for s in subjects):
        return False
    if not terms:
        return bool(subjects)
    hits = sum(1 for t in terms if _has(t, tokens))
    return hits / len(terms) >= 0.5


def _ir_metrics(rels, total_relevant, k=K):
    """Precision@k, Recall@k, MRR, nDCG@k from a ranked list of 0/1 judgements."""
    top = rels[:k]
    precision = sum(top) / k
    recall = (sum(top) / min(total_relevant, k)) if total_relevant else 0.0
    mrr = next((1.0 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    ndcg = dcg / ideal if ideal else 0.0
    return precision, recall, mrr, ndcg


def _sentences(text):
    text = re.sub(r'\*\*|__|`', '', text or '')
    out = []
    for s in _SENT_SPLIT.split(text):
        s = re.sub(r'^\s*[-*•\d.)]+\s*', '', s).strip()
        if len(s.split()) >= 6 and not s.endswith('?'):
            out.append(s)
    return out


def evidence_support(answer, passages):
    """Share of answer sentences whose content words are mostly (>=50%) found in the evidence."""
    sents = _sentences(answer)
    if not sents or not passages:
        return 0.0, 0, len(sents)
    ev_words = set()
    for p in passages:
        ev_words |= _words(p.get('content', '')) | _words(p.get('title', ''))
    supported = 0
    for s in sents:
        w = {x for x in _words(s) if x not in _FILLER}
        if w and sum(1 for x in w if _has(x, ev_words)) / len(w) >= 0.5:
            supported += 1
    return supported / len(sents), supported, len(sents)


def question_coverage(text, terms, subjects):
    """Share of the question's key terms (and all subjects) that the text addresses."""
    allt = set(terms) | set(subjects)
    if not allt:
        return 0.0
    w = _words(text)
    return sum(1 for t in allt if _has(t, w)) / len(allt)


def aspects_covered(text):
    low = (text or '').lower()
    return [name for name, cues in ASPECTS.items() if any(c in low for c in cues)]


def citation_count(text):
    return len(set(re.findall(r'\[(\d{1,2})\]', text or '')))


# --------------------------------------------------------------------------
# Per-question report
# --------------------------------------------------------------------------

def _all_passages():
    from services.retrieval_service import _bm25_retriever
    return _bm25_retriever.passages


def answer_report(question, ex_answer, ex_passages, ex_time, pr_answer, pr_passages, pr_time):
    terms, subjects = key_terms(question)
    corpus = _all_passages()
    total_rel = sum(1 for p in corpus if judge_relevant(p, terms, subjects))

    def side(answer, passages, latency):
        rels = [1 if judge_relevant(p, terms, subjects) else 0 for p in passages]
        precision, recall, mrr, ndcg = _ir_metrics(rels, total_rel)
        support, supported, n_sents = evidence_support(answer, passages)
        return {
            'precision_at_k': round(precision, 4), 'recall_at_k': round(recall, 4),
            'mrr': round(mrr, 4), 'ndcg_at_k': round(ndcg, 4),
            'passages_retrieved': len(passages), 'relevant_retrieved': sum(rels),
            'question_coverage': round(question_coverage(answer, terms, subjects), 4),
            'evidence_support': round(support, 4), 'supported_sentences': supported,
            'answer_sentences': n_sents,
            'aspects': aspects_covered(answer), 'aspect_breadth': round(len(aspects_covered(answer)) / len(ASPECTS), 4),
            'citations': citation_count(answer),
            'sources': len({p.get('document_id') or p.get('title') for p in passages}),
            'latency_ms': round(latency * 1000, 3),
        }

    return {
        'k': K, 'key_terms': sorted(terms), 'subjects': sorted(subjects),
        'relevant_in_index': total_rel,
        'baseline': side(ex_answer, ex_passages, ex_time),
        'verifai': side(pr_answer, pr_passages, pr_time),
    }


def to_metrics_row(report, verification_confidence):
    """Map the report onto the existing Metrics table columns (all real values)."""
    b, v = report['baseline'], report['verifai']
    return {
        'exact_match_existing': 0.0, 'exact_match_proposed': 0.0,      # no reference answer
        'token_f1_existing': b['evidence_support'], 'token_f1_proposed': v['evidence_support'],
        'accuracy_existing': b['question_coverage'], 'accuracy_proposed': v['question_coverage'],
        'precision_k_existing': b['precision_at_k'], 'precision_k_proposed': v['precision_at_k'],
        'recall_k_existing': b['recall_at_k'], 'recall_k_proposed': v['recall_at_k'],
        'mrr_existing': b['mrr'], 'mrr_proposed': v['mrr'],
        'ndcg_existing': b['ndcg_at_k'], 'ndcg_proposed': v['ndcg_at_k'],
        'evidence_relevance_existing': b['precision_at_k'], 'evidence_relevance_proposed': v['precision_at_k'],
        'evidence_coverage': v['evidence_support'],
        'verification_accuracy': verification_confidence,
        'response_time_existing': b['latency_ms'] / 1000, 'response_time_proposed': v['latency_ms'] / 1000,
    }


# --------------------------------------------------------------------------
# Labelled benchmark
# --------------------------------------------------------------------------

OUT_OF_SCOPE_PROBES = [
    'How many players are there in a cricket team?', 'What is the capital of France?',
    'Who won the football world cup in 2018?', 'How do I fix a computer virus on my laptop?',
    'What is the price of bitcoin today?', 'Best recipe for chicken biryani',
    'How do I learn Python programming?', 'Which movie won the Oscar for best picture?',
    'How do I change a car tyre?', 'What is the weather like in London?',
]

_bench_cache = {'key': None, 'result': None}


def _benchmark_questions():
    from services import knowledge_base
    items = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'datasets', 'biomedical_qa_test_20.csv')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rel = {r['context_title'].strip()}
                topic = knowledge_base.find_topic(r.get('category', '') + ' ' + r['question'])
                if topic:
                    rel.add(f"kb:{topic['id']}")
                items.append({'question': r['question'], 'relevant': rel, 'set': 'Curated test set'})
    for t in knowledge_base.load()[1]:
        for q in t.get('sample_questions', []):
            items.append({'question': q, 'relevant': {f"kb:{t['id']}"}, 'set': 'Knowledge-base questions'})
    return items


def _doc_key(p):
    d = p.get('document_id')
    return d if isinstance(d, str) and d.startswith('kb:') else (p.get('title') or '')


def _pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def run_benchmark(force=False):
    """Evaluate both retrievers on labelled questions. Cached until the index changes."""
    from services import retrieval_service as rs
    from services.domain_guard import check_domain
    rs.build_index_from_db()
    key = rs._index_state.get('key')
    if not force and _bench_cache['key'] == key and _bench_cache['result']:
        return _bench_cache['result']

    questions = _benchmark_questions()
    systems = {'baseline': rs.existing_retrieve, 'verifai': rs.proposed_retrieve}
    per = {name: {'hit1': [], 'hit5': [], 'mrr': [], 'ndcg5': [], 'lat': [], 'empty': 0} for name in systems}
    by_set = {}

    for item in questions:
        for name, fn in systems.items():
            # median of 3 runs for a stable latency figure
            runs = [fn(item['question'], 10) for _ in range(3)]
            res = runs[0][0]
            lat = statistics.median(r[1] for r in runs)
            rels = [1 if _doc_key(p) in item['relevant'] else 0 for p in res]
            top5 = rels[:5]
            hit1 = 1.0 if rels[:1] == [1] else 0.0
            hit5 = 1.0 if any(top5) else 0.0
            mrr = next((1.0 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
            dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top5))
            ndcg = dcg / 1.0 if dcg else 0.0   # one relevant source group -> ideal DCG = 1
            p = per[name]
            p['hit1'].append(hit1); p['hit5'].append(hit5); p['mrr'].append(mrr)
            p['ndcg5'].append(min(1.0, ndcg)); p['lat'].append(lat * 1000)
            p['empty'] += 0 if res else 1
            s = by_set.setdefault(item['set'], {}).setdefault(name, {'hit1': [], 'hit5': [], 'mrr': []})
            s['hit1'].append(hit1); s['hit5'].append(hit5); s['mrr'].append(mrr)

    def summary(p):
        n = len(p['hit1']) or 1
        return {
            'hit_at_1': round(sum(p['hit1']) / n, 4), 'recall_at_5': round(sum(p['hit5']) / n, 4),
            'mrr': round(sum(p['mrr']) / n, 4), 'ndcg_at_5': round(sum(p['ndcg5']) / n, 4),
            'latency_p50_ms': round(_pct(p['lat'], 0.5), 3), 'latency_p95_ms': round(_pct(p['lat'], 0.95), 3),
            'no_result_rate': round(p['empty'] / n, 4),
        }

    # Scope guard: in-scope = benchmark questions, out-of-scope = probes
    t0 = time.perf_counter()
    in_scope_qs = [q['question'] for q in questions]
    accepted = sum(1 for q in in_scope_qs if check_domain(q, use_llm=False)['in_scope'])
    rejected = sum(1 for q in OUT_OF_SCOPE_PROBES if not check_domain(q, use_llm=False)['in_scope'])
    guard_ms = (time.perf_counter() - t0) * 1000 / (len(in_scope_qs) + len(OUT_OF_SCOPE_PROBES))

    result = {
        'questions': len(questions),
        'sets': {name: sum(1 for q in questions if q['set'] == name) for name in by_set},
        'baseline': summary(per['baseline']),
        'verifai': summary(per['verifai']),
        'by_set': {set_name: {sys: {m: round(sum(v) / len(v), 4) if v else 0.0 for m, v in d.items()}
                              for sys, d in systems_d.items()} for set_name, systems_d in by_set.items()},
        'scope_guard': {
            'in_scope_total': len(in_scope_qs), 'in_scope_accepted': accepted,
            'out_scope_total': len(OUT_OF_SCOPE_PROBES), 'out_scope_rejected': rejected,
            'avg_ms': round(guard_ms, 3),
        },
        'index_passages': rs._index_state.get('count', 0),
        'generated_at': time.strftime('%Y-%m-%d %H:%M'),
    }
    _bench_cache.update(key=key, result=result)
    return result


# --------------------------------------------------------------------------
# Aggregates over measured answer reports (admin pages)
# --------------------------------------------------------------------------

# Shown on every admin evaluation page, in this order. kind: pct | num | int | ms
METRIC_DEFS = [
    {'key': 'evidence_support', 'name': 'Evidence support', 'kind': 'pct', 'lower_better': False,
     'group': 'Answer quality', 'baseline_na': True,
     'definition': "Share of answer sentences whose content words are mostly (at least 50%) found in that "
                   "system's retrieved passages. The baseline quotes a passage verbatim, so it scores 100% by construction."},
    {'key': 'question_coverage', 'name': 'Question coverage', 'kind': 'pct', 'lower_better': False,
     'group': 'Answer quality', 'definition': "Share of the question's key terms that the answer addresses."},
    {'key': 'aspect_breadth', 'name': 'Clinical breadth', 'kind': 'pct', 'lower_better': False,
     'group': 'Answer quality',
     'definition': 'Clinical aspects covered out of six: definition, causes, symptoms, diagnosis, treatment, prevention.'},
    {'key': 'citations', 'name': 'Citations per answer', 'kind': 'num', 'lower_better': False,
     'group': 'Answer quality', 'definition': 'Distinct sources cited inline in the answer. The baseline never cites.'},
    {'key': 'precision_at_k', 'name': 'Precision@6', 'kind': 'pct', 'lower_better': False,
     'group': 'Retrieval quality',
     'definition': "Share of retrieved passages judged relevant (they name the question's topic and at least half of its key terms)."},
    {'key': 'recall_at_k', 'name': 'Recall@6', 'kind': 'pct', 'lower_better': False, 'group': 'Retrieval quality',
     'definition': 'Relevant passages retrieved, out of the most that could fit in the top 6.'},
    {'key': 'mrr', 'name': 'MRR', 'kind': 'num', 'lower_better': False, 'group': 'Retrieval quality',
     'definition': 'How early the first relevant passage appears (1 = first position).'},
    {'key': 'ndcg_at_k', 'name': 'nDCG@6', 'kind': 'num', 'lower_better': False, 'group': 'Retrieval quality',
     'definition': 'Ranking quality, rewarding relevant passages placed higher.'},
    {'key': 'latency_ms', 'name': 'Retrieval latency', 'kind': 'ms', 'lower_better': True, 'group': 'Speed',
     'definition': 'Time to search the full index for one question. Lower is better.'},
]


def aggregate_reports(limit_recent=50):
    """Averages, per-question series and breakdowns over every stored, measured answer report."""
    import json
    from models.report import AnswerReport
    from models.question import QASession
    rows = (AnswerReport.query.join(QASession, QASession.id == AnswerReport.session_id)
            .with_entities(AnswerReport.session_id, AnswerReport.data, QASession.question, QASession.created_at)
            .order_by(AnswerReport.session_id).all())
    reports = []
    for sid, raw, question, created in rows:
        try:
            r = json.loads(raw)
        except Exception:
            continue
        r['session_id'], r['question'], r['created_at'] = sid, question, created
        reports.append(r)

    keys = [m['key'] for m in METRIC_DEFS]
    # Answer-quality metrics only make sense when the system produced an evidence-based answer
    grounded = [r for r in reports if r.get('baseline', {}).get('passages_retrieved') and
                r.get('verifai', {}).get('passages_retrieved')]

    def mean(side, key, pool):
        vals = [float(r[side].get(key, 0) or 0) for r in pool if side in r]
        return sum(vals) / len(vals) if vals else None

    def median(side, key, pool):
        vals = sorted(float(r[side].get(key, 0) or 0) for r in pool if side in r)
        return vals[len(vals) // 2] if vals else None

    averages = {}
    for m in METRIC_DEFS:
        k = m['key']
        pool = grounded or reports
        if k == 'latency_ms':
            b, v = median('baseline', k, reports), median('verifai', k, reports)
        else:
            b, v = mean('baseline', k, pool), mean('verifai', k, pool)
        if m.get('baseline_na'):
            # Not comparable: the baseline's answer IS a retrieved passage, so it "supports"
            # itself by construction. Reported for VerifAI only.
            averages[k] = {'baseline': None, 'verifai': v, 'change_pct': None, 'winner': None, 'na': True}
            continue
        if b is None or v is None:
            change = None
        elif m['lower_better']:
            change = ((b - v) / b * 100) if b else None
        else:
            change = ((v - b) / abs(b) * 100) if b else None
        if b is None or v is None:
            winner = None
        elif m['lower_better']:
            winner = 'verifai' if v < b * 0.98 else ('baseline' if b < v * 0.98 else 'tie')
        else:
            winner = 'verifai' if v > b + 0.0049 else ('baseline' if b > v + 0.0049 else 'tie')
        averages[k] = {'baseline': b, 'verifai': v, 'change_pct': change, 'winner': winner}

    statuses, sources = {}, {}
    for r in reports:
        s = (r.get('verification') or {}).get('verification_status', 'Unknown')
        statuses[s] = statuses.get(s, 0) + 1
        src = r.get('answer_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    series = {
        'labels': [f"Q{r['session_id']}" for r in reports],
        'baseline': {k: [r['baseline'].get(k) for r in reports] for k in keys},
        'verifai': {k: [r['verifai'].get(k) for r in reports] for k in keys},
    }
    recent = [{
        'session_id': r['session_id'], 'question': r['question'],
        'created_at': r['created_at'].strftime('%d %b %H:%M') if r.get('created_at') else '',
        'status': (r.get('verification') or {}).get('verification_status', ''),
        'baseline': r['baseline'], 'verifai': r['verifai'],
    } for r in reversed(reports[-limit_recent:])]

    return {
        'count': len(reports), 'grounded_count': len(grounded),
        'metrics': METRIC_DEFS, 'averages': averages,
        'statuses': statuses, 'sources': sources, 'series': series, 'recent': recent,
    }
