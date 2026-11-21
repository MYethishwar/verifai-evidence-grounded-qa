import os
import re
import string
from config import Config

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Kept short and identical in every request: the local model then re-uses the already
# processed prefix instead of reading it again (reading the prompt is the slowest step on CPU).
PERSONA = (
    "You are VerifAI, a warm, conversational biomedical assistant for clinicians, students and "
    "researchers. You only discuss biomedical topics, never claim to be a doctor, and suggest a "
    "clinician for personal medical decisions. Answer using ONLY the numbered passages you are given "
    "and cite each fact with its passage number at the end of the sentence, like [1] (never write "
    "'according to passage'). Give a brief, broad overview in plain "
    "language (about 100-170 words, short bullets welcome). If the passages do not cover something, "
    "say so in one sentence. Never add facts, drug names, doses or numbers that are not in the passages."
)

MAX_ANSWER_TOKENS = 300      # ~170 words
PROMPT_PASSAGES = 4          # passages shown to the model (and cited)
PASSAGE_CHARS = 420          # characters per passage in the prompt
HISTORY_TURNS = 2            # previous messages kept for conversational tone
HISTORY_CHARS = 300

UNAVAILABLE_MESSAGE = (
    "I couldn't find this topic in the indexed medical literature, and no AI model is "
    "running right now to answer it from general knowledge. Start Ollama (or set an "
    "OPENAI_API_KEY) and ask again, or try a question about one of the indexed topics."
)


# Answers come only from the provided data (knowledge base + uploaded datasets). Set the
# environment variable VERIFAI_ALLOW_GENERAL_KNOWLEDGE=1 to let the model answer uncovered
# biomedical questions from its own knowledge (always labelled "General Knowledge").
ALLOW_GENERAL_KNOWLEDGE = os.environ.get('VERIFAI_ALLOW_GENERAL_KNOWLEDGE', '0') == '1'


def not_covered_message(question):
    """Polite reply for in-scope questions that the provided data does not cover."""
    from services import knowledge_base
    q_words = set(re.findall(r'[a-z0-9]{4,}', (question or '').lower()))
    scored = []
    for t in knowledge_base.load()[1]:
        text = ' '.join([t['title'], ' '.join(t.get('aliases', [])), t.get('category', '')]).lower()
        overlap = len(q_words & set(re.findall(r'[a-z0-9]{4,}', text)))
        if overlap:
            scored.append((overlap, t['title']))
    related = [title for _, title in sorted(scored, reverse=True)[:3]]
    msg = ("I couldn't find this in VerifAI's sources: neither the curated knowledge base nor the "
           "uploaded datasets cover it, so I won't answer from outside information.")
    if related:
        msg += " Related topics I can answer from: " + ', '.join(related) + "."
    msg += (" You can browse every covered topic in the Knowledge base, or ask an administrator "
            "to add a dataset on this subject.")
    return msg


def _evidence_block(evidence_passages):
    parts = []
    for i, ev in enumerate(evidence_passages[:PROMPT_PASSAGES], 1):
        text = ev.get('content', ev.get('passage_text', ''))
        title = ev.get('title', ev.get('source_title', 'Document'))
        parts.append(f'[{i}] {title}: {text[:PASSAGE_CHARS]}')
    return '\n'.join(parts)


_ASPECT_CUES = [
    (('symptom', 'sign', 'feel', 'present'), ('symptom', 'common concerns')),
    (('cause', 'risk', 'why do', 'trigger'), ('cause',)),
    (('diagnos', 'test', 'detect', 'screen', 'check'), ('diagnosis', 'how it is done', 'why it is ordered', 'check-ups')),
    (('treat', 'therap', 'cure', 'manage', 'medication', 'medicine'), ('treatment', 'care & management', 'common uses')),
    (('prevent', 'outlook', 'prognosis', 'avoid', 'reduce the risk'), ('prevention', 'outlook', 'practical tips')),
    (('side effect', 'adverse', 'safe'), ('side effects', 'precautions')),
    (('how does', 'how do', 'work', 'mechanism'), ('how it works', 'function')),
    (('result', 'normal', 'level', 'high', 'low'), ('understanding results',)),
]


def _prioritise_sections(question, passages):
    """
    Put the section that matches what the question asks about first: for "What are the
    symptoms of hypothyroidism?" the "Hypothyroidism: Symptoms" passage leads, even when
    keyword ranking placed it lower (aspect words are not used for ranking).
    """
    q = (question or '').lower()
    wanted = [labels for cues, labels in _ASPECT_CUES if any(c in q for c in cues)]
    if not wanted:
        return list(passages)

    def rank(item):
        i, p = item
        title = (p.get('title') or '').lower()
        section = title.split(':', 1)[1] if ':' in title else ''
        hit = any(any(l in section for l in labels) for labels in wanted)
        return (0 if hit else 1, i)
    return [p for _, p in sorted(enumerate(passages), key=rank)]


def plan_answer(question, evidence_passages, history=None, user_name=None, greeted=False):
    """
    Decide how to answer. Returns either
      {'final': {...answer dict...}}             - no model call needed, or
      {'messages': [...], 'evidence': [...], 'source': 'evidence'|'knowledge', 'label': str}
    """
    from services import llm_client, knowledge_base

    if not evidence_passages:
        # Retrieval found nothing on-topic. If the question names a knowledge-base topic,
        # answer from that topic's sections; otherwise say it isn't covered.
        topic = knowledge_base.find_topic(question)
        if topic:
            evidence_passages = [p for p in knowledge_base.passages() if p['kb_id'] == topic['id']]
        elif not ALLOW_GENERAL_KNOWLEDGE:
            return {'final': {'answer': not_covered_message(question), 'method': 'VerifAI', 'source': 'not_covered'}}

    label = llm_client.provider_label()
    if not label:
        return {'final': _rule_based_answer(question, evidence_passages)}

    evidence = _prioritise_sections(question, evidence_passages)[:PROMPT_PASSAGES]
    hints = []
    if greeted:   # only greet back when the user greeted; otherwise answer straight away
        who = f' ({user_name})' if user_name else ''
        hints.append(f'The user{who} greeted you: greet them back in a few words, then answer.')
    if evidence:
        body = 'Passages:\n' + _evidence_block(evidence)
        source = 'evidence'
    else:
        body = ('No indexed passage covers this question. Answer from well-established biomedical '
                'knowledge, start by saying it is general information, and do not cite anything.')
        source = 'knowledge'
    history = [{'role': m['role'], 'content': m['content'][:HISTORY_CHARS]}
               for m in (history or [])[-HISTORY_TURNS:]]
    messages = ([{'role': 'system', 'content': PERSONA}] + history +
                [{'role': 'user', 'content': (' '.join(hints) + '\n' if hints else '') + body +
                  f'\n\nQuestion: {question}'}])
    return {'messages': messages, 'evidence': evidence, 'source': source, 'label': label}


def finish_answer(plan, text):
    """Attach measured citations and package the answer."""
    if plan['source'] == 'evidence':
        text = attach_citations(text, plan['evidence'])
    return {'answer': text, 'method': f"VerifAI · {plan['label']}", 'source': plan['source'],
            'evidence': plan['evidence']}


def generate_answer_ex(question, evidence_passages, history=None, user_name=None, greeted=False):
    """
    Generate the VerifAI answer (non-streaming).

    Returns dict: answer, method, source ('evidence' | 'knowledge' | 'unavailable' | 'not_covered'),
    and evidence (the passages shown to the model, in citation order).
    """
    from services import llm_client
    plan = plan_answer(question, evidence_passages, history, user_name, greeted)
    if 'final' in plan:
        return plan['final']
    try:
        text = llm_client.chat(plan['messages'], max_tokens=MAX_ANSWER_TOKENS, temperature=0.3)
        if text:
            return finish_answer(plan, text)
    except Exception as e:
        print(f'LLM generation error: {e}. Falling back to rule-based engine.')
    return _rule_based_answer(question, plan['evidence'])


def _rule_based_answer(question, evidence_passages):
    """No model available: extractive answer from the passages / knowledge base."""
    if evidence_passages:
        answer, method = _synthesize_biomedical_answer(question, evidence_passages)
        if answer:
            return {'answer': answer, 'method': method, 'source': 'evidence', 'evidence': evidence_passages}
    kb_answer = _biomedical_knowledge_fallback(question)
    if kb_answer:
        return {'answer': kb_answer, 'method': 'VerifAI Knowledge Base',
                'source': 'evidence' if evidence_passages else 'knowledge', 'evidence': evidence_passages}
    return {'answer': UNAVAILABLE_MESSAGE, 'method': 'VerifAI', 'source': 'unavailable'}


_CITE_RE = re.compile(r'\[\d{1,2}\]')


def attach_citations(answer, passages, threshold=0.5):
    """
    Automatic source attribution. For every answer sentence (or bullet) that has no citation,
    find the passage that contains the largest share of its content words; if that share is at
    least `threshold`, append the passage number, e.g. "... airways [2]." Sentences that no
    passage supports are left uncited, so citations only ever reflect measured overlap.
    """
    from services.evaluation_service import _words, _has, _FILLER
    if not passages or not answer:
        return answer
    pw = [_words(p.get('content', '')) | _words(p.get('title', '')) for p in passages]
    out_lines = []
    for line in answer.split('\n'):
        parts = re.split(r'(?<=[.!?])\s+', line)
        new_parts = []
        for part in parts:
            words = {w for w in _words(part) if w not in _FILLER}
            if len(words) < 4 or _CITE_RE.search(part) or part.strip().endswith('?'):
                new_parts.append(part)
                continue
            best_i, best = None, 0.0
            for i, ws in enumerate(pw):
                share = sum(1 for w in words if _has(w, ws)) / len(words)
                if share > best:
                    best_i, best = i, share
            if best_i is not None and best >= threshold:
                m = re.match(r'^(.*?)([.!?:;]*)(\s*)$', part, re.S)
                part = f"{m.group(1)} [{best_i + 1}]{m.group(2)}{m.group(3)}"
            new_parts.append(part)
        out_lines.append(' '.join(new_parts))
    return '\n'.join(out_lines)


def generate_answer(question, evidence_passages):
    """Backwards-compatible wrapper. Returns (answer_text, method_used)."""
    r = generate_answer_ex(question, evidence_passages)
    return r['answer'], r['method']


# ---------------------------------------------------------------------------
# OpenAI generator  (gpt-4o-mini — free-tier eligible)
# ---------------------------------------------------------------------------

def _openai_generate(question, context, api_key):
    """Use OpenAI gpt-4o-mini to produce a unique, question-specific answer."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are VerifAI, an advanced biomedical clinical question-answering assistant. "
        "You MUST answer the EXACT question asked. "
        "Every answer must be UNIQUE and SPECIFIC to the precise topic in the question. "
        "Never give a generic or repeated answer. "
        "If the question is about a specific cancer type (e.g. lung cancer), answer only "
        "about that specific cancer, not cancer in general or other types. "
        "Base your answer firmly on the provided evidence passages. "
        "SCOPE: You only answer biomedical, medical, health, human biology, genetics, "
        "pharmacology and clinical-science questions. If the question is about anything "
        "else (sports, entertainment, technology, finance, general knowledge, etc.), do NOT "
        "answer it and do NOT just say 'no'. Instead, politely explain that the question is "
        "not related to VerifAI's purpose as a biomedical question-answering assistant, and "
        "suggest asking a biomedical question such as 'What are the symptoms of Type 2 diabetes?'"
    )

    prompt = (
        f"Question: {question}\n\n"
        f"Evidence Passages:\n{context}\n\n"
        "Instructions:\n"
        "1. Answer the EXACT question above in 4–6 well-structured sentences.\n"
        "2. Be SPECIFIC to the topic — do not mix content from other diseases or cancer types.\n"
        "3. Cover: definition/pathophysiology, key symptoms, risk factors, diagnosis, and treatment where relevant.\n"
        "4. Use precise medical terminology grounded in the evidence above.\n"
        "5. Do NOT repeat information across sentences."
    )

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user',   'content': prompt}
        ],
        max_tokens=600,
        temperature=0.15
    )
    answer = response.choices[0].message.content.strip()
    return answer, 'VerifAI GPT-4o-mini (OpenAI)'


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

def _clean_sentence(text):
    """Clean and capitalise a sentence."""
    text = text.strip()
    if not text:
        return ''
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text


def _sentence_similarity(s1, s2):
    """
    Simple Jaccard token overlap similarity.
    Returns float in [0, 1].
    """
    t1 = set(re.findall(r'\b[a-z0-9]{3,}\b', s1.lower()))
    t2 = set(re.findall(r'\b[a-z0-9]{3,}\b', s2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _deduplicate_sentences(sentences, threshold=0.55):
    """
    Remove sentences that are >threshold similar to an already-selected sentence.
    Guarantees unique content in the assembled answer.
    Uses Jaccard token overlap — a threshold of 0.55 reliably removes near-paraphrase duplicates.
    """
    selected = []
    for s in sentences:
        is_dup = any(_sentence_similarity(s, kept) >= threshold for kept in selected)
        if not is_dup:
            selected.append(s)
    return selected


def _extract_sentences_from_single_doc(content, title='', disease_keywords=None):
    """
    Extract distinct, informative sentences from a single passage.
    """
    if not content:
        return []

    # 1. Strip title prefix if content starts with title
    if title and len(title) > 5:
        pattern = re.compile(r'^' + re.escape(title.strip()), re.IGNORECASE)
        content = pattern.sub('', content).strip()

    # 2. Strip leading question prefix if content starts with question header
    content = re.sub(r'^(what are the|what is the|how does|what clinical|which|tell me)[^\.\?!]*[\.\?!:]?\s*', '', content, flags=re.IGNORECASE).strip()

    raw_chunks = re.split(r'[\n\r]+|(?<=[.!?])\s+', content)
    sentences  = []
    seen       = set()

    for s in raw_chunks:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) < 4:
            continue
        # Skip pure question strings or header stubs
        if s.endswith('?') or s.lower().startswith(('what ', 'how ', 'which ', 'where ', 'why ', 'tell me ', 'explain ')):
            continue

        norm = re.sub(r'[^\w\s]', '', s.lower())
        if norm not in seen:
            seen.add(norm)
            sentences.append(_clean_sentence(s))

    return sentences


# ---------------------------------------------------------------------------
# Generic stop-words / non-discriminating terms
# ---------------------------------------------------------------------------

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
    'screening', 'guidelines', 'acute', 'chronic', 'study', 'studies', 'review', 'summary'
}


# ---------------------------------------------------------------------------
# Local synthesis (no API key required)
# ---------------------------------------------------------------------------

def _is_document_topic_aligned(question, passage, disease_keywords):
    """Verify whether a passage's main topic aligns with the user question."""
    if not passage or not disease_keywords:
        return False
    title = (passage.get('title') or '').lower()
    content = (passage.get('content') or '').lower()
    title_tokens = set(re.findall(r'\b[a-z0-9\-]{3,}\b', title))
    content_tokens = set(re.findall(r'\b[a-z0-9\-]{3,}\b', content))
    
    # Check each disease keyword in the query
    for kw in disease_keywords:
        if kw == 'dna':
            dna_terms = {'dna', 'deoxyribonucleic', 'double-helix', 'genetics', 'genome', 'chromosomal'}
            if not dna_terms.intersection(title_tokens):
                dna_content_count = sum(1 for t in ['dna', 'deoxyribonucleic', 'double helix', 'nucleotide', 'base pair', 'chargaff', 'adenine', 'thymine'] if t in content)
                if dna_content_count < 2:
                    return False
        elif kw == 'rna':
            rna_terms = {'rna', 'ribonucleic', 'mrna', 'trna', 'rrna'}
            if not rna_terms.intersection(title_tokens):
                rna_content_count = sum(1 for t in ['rna', 'ribonucleic', 'mrna', 'trna', 'rrna', 'uracil'] if t in content)
                if rna_content_count < 2:
                    return False
        else:
            if kw in title_tokens or kw in content_tokens:
                continue
            return False

    return True

def _synthesize_biomedical_answer(question, passages):
    """
    Advanced local Biomedical Knowledge & Evidence Synthesizer.
    Extracts sentences ONLY from the single best-matching document and
    filters them by disease keywords to guarantee unique, on-topic answers.
    """
    q_lower = question.lower()
    q_words  = set(re.findall(r'\b[a-z0-9\-]{3,}\b', q_lower))
    disease_keywords = {w for w in q_words if w not in GENERIC_TERMS and len(w) >= 3}

    # ── No passages → pure knowledge fallback ──
    if not passages:
        return _biomedical_knowledge_fallback(question), 'VerifAI Knowledge Base'

    # ── Step 2: Extract sentences from top document groups ──
    raw_sentences = []
    seen_norms    = set()
    best_passage  = None

    # Group passages by document
    doc_groups = {}
    for p in passages:
        doc_key = p.get('document_id') or p.get('title')
        doc_groups.setdefault(doc_key, []).append(p)

    # Try each document group in order of score to find real informative sentences
    for doc_key, group in doc_groups.items():
        # Verify topic alignment for this document
        if disease_keywords and not _is_document_topic_aligned(question, group[0], disease_keywords):
            continue

        doc_sents = []
        for p in group:
            sents = _extract_sentences_from_single_doc(
                p.get('content', ''),
                p.get('title', '')
            )
            for s in sents:
                norm = re.sub(r'[^\w\s]', '', s.lower())
                if norm not in seen_norms:
                    seen_norms.add(norm)
                    doc_sents.append(s)
        if doc_sents:
            raw_sentences = doc_sents
            best_passage = group[0]
            break

    if not raw_sentences:
        return _biomedical_knowledge_fallback(question), 'VerifAI Knowledge Base'

    # ── Step 3: Deduplicate and assemble answer (up to 4 sentences) ──
    unique_sentences = _deduplicate_sentences(raw_sentences, threshold=0.72)
    final_answer     = ' '.join(unique_sentences[:4])

    return final_answer, 'VerifAI Biomedical Engine'


# ---------------------------------------------------------------------------
# Knowledge fallback: the curated knowledge base (knowledge_base/*.json)
# ---------------------------------------------------------------------------

def _biomedical_knowledge_fallback(question):
    """
    Breadth-first overview from the curated knowledge base for the question's topic,
    or None when the knowledge base has no matching topic. The domain knowledge itself
    lives in knowledge_base/*.json (see services/knowledge_base.py), not in code.
    """
    from services import knowledge_base
    topic = knowledge_base.find_topic(question)
    return knowledge_base.overview_answer(topic) if topic else None
