"""
VerifAI curated knowledge base.

The domain knowledge lives as plain JSON in the top-level `knowledge_base/` folder,
one file per topic group (heart & lungs, metabolism, brain & infections, cancer,
biology basics, medicines & tests). Each topic has a short summary, five sections
(e.g. causes, symptoms, diagnosis, treatment, prevention) and sample questions.

At startup every section becomes a retrievable passage alongside the uploaded
datasets, so answers about these topics are grounded in - and cite - the knowledge
base. Edit or add JSON files to extend what VerifAI knows; no code changes needed.
"""

import json
import os
import re

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_base')
KB_SOURCE_NAME = 'VerifAI Knowledge Base'

_cache = {'mtime': None, 'groups': [], 'topics': []}


def _dir_mtime():
    try:
        return max((os.path.getmtime(os.path.join(KB_DIR, f))
                    for f in os.listdir(KB_DIR) if f.endswith('.json')), default=0)
    except FileNotFoundError:
        return 0


def load():
    """Load (and cache) all knowledge-base files. Reloads automatically when a file changes."""
    mtime = _dir_mtime()
    if _cache['mtime'] == mtime:
        return _cache['groups'], _cache['topics']
    groups, topics, seen = [], [], set()
    if os.path.isdir(KB_DIR):
        for fname in sorted(os.listdir(KB_DIR)):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(KB_DIR, fname), encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f'Knowledge base: skipped {fname}: {e}')
                continue
            group = {'name': data.get('group', fname[:-5]), 'file': fname, 'topics': []}
            for t in data.get('topics', []):
                if not t.get('id') or t['id'] in seen or not t.get('sections'):
                    continue
                seen.add(t['id'])
                t = dict(t, group=group['name'])
                t['aliases'] = [a.lower().strip() for a in t.get('aliases', []) if a.strip()]
                group['topics'].append(t)
                topics.append(t)
            groups.append(group)
    _cache.update(mtime=mtime, groups=groups, topics=topics)
    return groups, topics


def topic_count():
    return len(load()[1])


def passages():
    """One retrievable passage per topic section (plus the summary)."""
    out = []
    for t in load()[1]:
        alias_text = f" (also: {', '.join(t['aliases'][:6])})" if t['aliases'] else ''
        base = {
            'id': None,                         # not a DB passage
            'kb_id': t['id'],
            'document_id': f"kb:{t['id']}",
            'journal': KB_SOURCE_NAME,
            'year': '',
            'authors': '',
            'source_url': f"/user/knowledge-base#{t['id']}",
            'doi': '',
            'category': t.get('category', ''),
        }
        out.append(dict(base, title=f"{t['title']}: Overview",
                        content=f"{t['summary']}{alias_text}"))
        for s in t['sections']:
            out.append(dict(base, title=f"{t['title']}: {s['label']}",
                            content=s['text']))
    return out


def subject_terms():
    """Single-word topic names and aliases ('asthma', 'metformin', 'hba1c') that identify a
    question's subject. Multi-word names ('heart attack') are not split, so everyday words
    like 'heart' or 'high' never become mandatory subjects."""
    terms = set()
    for t in load()[1]:
        title = re.sub(r'\s*\(.*?\)', '', t['title'].lower()).strip()
        inner = re.findall(r'\((.*?)\)', t['title'].lower())
        for name in [title, t['id']] + inner + t['aliases']:
            name = name.strip()
            if name and ' ' not in name and len(name) >= 3:
                terms.add(name)
        # Distinctive words inside multi-word names: 'hba1c' from 'HbA1c Test',
        # 'psoriasis' from 'Plaque Psoriasis' - but not everyday words like 'heart' or 'high'.
        for name in [title, t['id'].replace('-', ' ')] + inner:
            for w in re.findall(r"[a-z0-9][a-z0-9']+", name):
                if len(w) >= 3 and w not in _EVERYDAY_TITLE_WORDS:
                    terms.add(w)
    return terms


_EVERYDAY_TITLE_WORDS = {
    'the', 'and', 'of', 'in', 'for', 'how', 'with', 'test', 'tests', 'disease', 'diseases',
    'disorder', 'disorders', 'syndrome', 'cancer', 'therapy', 'high', 'low', 'back', 'pain',
    'attack', 'failure', 'heart', 'blood', 'healthy', 'health', 'human', 'care', 'physical',
    'activity', 'function', 'work', 'works', 'diet', 'sleep', 'type', 'chronic', 'acute',
    'deficiency', 'common', 'fever', 'infection', 'tract', 'artery', 'vein', 'deep', 'basics',
    'cell', 'system', 'and', 'childhood', 'prenatal', 'skin', 'fatty', 'liver', 'kidney',
    'urinary', 'inflammatory', 'bowel', 'irritable', 'peripheral', 'metabolic', 'stones',
    'painkillers', 'medicines', 'inhibitors', 'panel', 'count', 'complete', 'resistance',
}


def subject_phrases():
    phrases = set()
    for t in load()[1]:
        for name in [re.sub(r'\s*\(.*?\)', '', t['title'].lower())] + t['aliases']:
            if ' ' in name.strip():
                phrases.add(name.strip())
    return phrases


def find_topic(question):
    """Best-matching topic for a question by title/alias, or None."""
    q = ' ' + re.sub(r'[^a-z0-9\- ]', ' ', (question or '').lower()) + ' '
    best, best_len = None, 0
    for t in load()[1]:
        names = [re.sub(r'\s*\(.*?\)', '', t['title'].lower()), t['id'].replace('-', ' ')] + t['aliases']
        for n in names:
            n = n.strip()
            if len(n) >= 3 and f' {n} ' in q and len(n) > best_len:
                best, best_len = t, len(n)
    return best


def overview_answer(topic):
    """Breadth-first plain answer assembled from one topic (used when no LLM is running)."""
    lines = [topic['summary'], '']
    for s in topic['sections']:
        lines.append(f"- **{s['label']}:** {s['text']}")
    return '\n'.join(lines)


def catalog():
    """Groups -> categories -> topics with sample questions, for the topics page."""
    groups, _ = load()
    return [{
        'name': g['name'],
        'topics': [{
            'id': t['id'], 'title': t['title'], 'category': t.get('category', ''),
            'type': t.get('type', ''), 'summary': t['summary'],
            'sections': [s['label'] for s in t['sections']],
            'sample_questions': t.get('sample_questions', []),
        } for t in g['topics']],
    } for g in groups]
