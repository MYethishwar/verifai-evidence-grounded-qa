"""
Conversational layer for the Ask a Question page.

Each user message is routed dynamically:
  smalltalk   -> greetings, thanks, "who are you", goodbyes: friendly reply, no retrieval
  follow-up   -> rewritten into a standalone question using the conversation so far
  out of scope-> polite explanation of what VerifAI is for
  biomedical  -> full pipeline (TF-IDF vs BM25 + LLM + verification), grounded in the
                 indexed literature when possible, otherwise from model knowledge

Context window: the last turns of the conversation are sent to the LLM within a
character budget (CONTEXT_BUDGET_CHARS); the conversation's last biomedical topic is
kept separately so follow-ups still resolve after older turns slide out of the window.
"""

import json
import random
import re

from extensions import db
from models.chat import Conversation, ChatMessage
from services import llm_client
from services.ai_service import PERSONA
from services.domain_guard import check_domain, topic_signals
from services.retrieval_service import GENERIC_TERMS

CONTEXT_BUDGET_CHARS = 6000   # ~1,500 tokens of recent conversation sent to the model
CONTEXT_MAX_MESSAGES = 12
PER_MESSAGE_CHARS = 1500

# --------------------------------------------------------------------------
# Intent detection
# --------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|howdy|greetings|namaste|namaskar|yo|hola|"
    r"good\s+(morning|afternoon|evening|day)|gm)\b[\s,!.:-]*(there|team|verifai|bot)?[\s,!.:-]*",
    re.I,
)
_SMALLTALK_PATTERNS = [
    ('thanks',  re.compile(r"^\s*(thanks|thank\s*(you|u)|thx|ty|much appreciated|great,? thanks)\b", re.I)),
    ('bye',     re.compile(r"^\s*(bye|goodbye|see\s+(you|ya)|good\s*night|take care|that'?s all)\b", re.I)),
    ('howareyou', re.compile(r"^\s*(how\s+are\s+(you|u)|how'?s\s+it\s+going|how\s+do\s+you\s+do|what'?s\s+up|wassup|sup)\b", re.I)),
    ('identity', re.compile(r"^\s*(who|what)\s+are\s+(you|u)\b|^\s*what\s+is\s+verifai\b|^\s*introduce\s+yourself", re.I)),
    ('help',    re.compile(r"^\s*(help|what\s+can\s+(you|u)\s+do|how\s+(do|can)\s+i\s+use\s+(you|this)|what\s+do\s+you\s+know)\b", re.I)),
    ('ack',     re.compile(r"^\s*(ok(ay)?|cool|great|nice|awesome|got\s+it|perfect|alright|sure|wow)\s*[.!]*\s*$", re.I)),
]

_FOLLOWUP_WORDS = {
    'it', 'its', "it's", 'this', 'that', 'these', 'those', 'they', 'them', 'their',
    'he', 'she', 'his', 'her', 'same', 'above', 'former', 'latter', 'such',
}
_FOLLOWUP_STARTS = ('and ', 'what about', 'how about', 'also', 'then ', 'so ', 'but ',
                    'why ', 'is it', 'are they', 'can it', 'does it', 'more ', 'tell me more',
                    'explain more', 'elaborate', 'in children', 'in women', 'in men')


def split_greeting(message):
    """Return (greeted: bool, remainder: str)."""
    m = _GREETING_RE.match(message)
    if not m:
        return False, message.strip()
    return True, message[m.end():].strip(' ,.!?-:')


def smalltalk_kind(message):
    for kind, rx in _SMALLTALK_PATTERNS:
        if rx.search(message):
            return kind
    return None


def _topic_terms(question):
    words = re.findall(r"[a-z0-9][a-z0-9\-]+", (question or '').lower())
    return [w for w in words if w not in GENERIC_TERMS and w not in _FOLLOWUP_WORDS and len(w) > 2]


def is_follow_up(message, last_topic):
    if not last_topic:
        return False
    text = message.lower().strip()
    sig = topic_signals(message)
    if sig['off_domain']:
        return False
    words = set(re.findall(r"[a-z']+", text))
    # A message that names its own subject ("what is malaria and how is it spread?") is a
    # new question even if it contains pronouns.
    subject = {t for t in sig['static_strong'] | sig['corpus_strong'] if t not in GENERIC_TERMS}
    if subject:
        return False
    if words & _FOLLOWUP_WORDS or text.startswith(_FOLLOWUP_STARTS):
        return True
    # Short question without its own subject: "symptoms?", "any vaccines?"
    return len(words) <= 6


_PRONOUN_RE = re.compile(r"\b(it|its|it's|this|that|these|those|they|them|their|the condition|the disease|this condition|this disease)\b", re.I)


def _topic_name(last_topic):
    """Short name of what the previous question was about ('asthma', 'type 2 diabetes')."""
    from services import knowledge_base
    topic = knowledge_base.find_topic(last_topic or '')
    if topic:
        return re.sub(r'\s*\(.*?\)', '', topic['title']).strip().lower()
    terms = list(dict.fromkeys(_topic_terms(last_topic)))
    return ' '.join(terms[:3])


def contextualize(message, last_topic):
    """
    Rewrite a follow-up into a standalone question without a model call:
    'how is it treated?' after a question on asthma -> 'how is asthma treated?'.
    """
    name = _topic_name(last_topic)
    if not name:
        return message
    replaced, n = _PRONOUN_RE.subn(lambda m: name + ("'s" if m.group(0).lower() in ('its', 'their') else ''),
                                   message, count=1)
    if n:
        return replaced
    return f"{message.rstrip(' ?.')} ({name})?"


# --------------------------------------------------------------------------
# Conversation memory
# --------------------------------------------------------------------------

def get_conversation(user_id, conversation_id=None, create=True):
    conv = None
    if conversation_id:
        conv = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()
    if conv is None and create:
        conv = Conversation(user_id=user_id, title='New conversation')
        db.session.add(conv)
        db.session.flush()
    return conv


def build_history(conv):
    """Most recent turns that fit the context budget, oldest first, in chat-API format."""
    recent = (ChatMessage.query.filter_by(conversation_id=conv.id)
              .order_by(ChatMessage.id.desc()).limit(CONTEXT_MAX_MESSAGES * 2).all())
    picked, used = [], 0
    for m in recent:
        content = m.content[:PER_MESSAGE_CHARS]
        if picked and used + len(content) > CONTEXT_BUDGET_CHARS:
            break
        picked.append({'role': m.role, 'content': content})
        used += len(content)
        if len(picked) >= CONTEXT_MAX_MESSAGES:
            break
    picked.reverse()
    # Chat APIs expect the history to start with a user turn.
    while picked and picked[0]['role'] != 'user':
        picked.pop(0)
    return picked


def serialize_message(m):
    meta = {}
    if m.meta:
        try:
            meta = json.loads(m.meta)
        except Exception:
            meta = {}
    return {
        'id': m.id, 'role': m.role, 'content': m.content, 'kind': m.kind,
        'standalone_question': m.standalone_question,
        'session_id': m.qa_session_id, 'meta': meta,
    }


# --------------------------------------------------------------------------
# Replies
# --------------------------------------------------------------------------

_HONORIFICS = {'dr', 'dr.', 'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.', 'miss', 'prof', 'prof.', 'sir', 'madam'}


def _first_name(user):
    parts = (getattr(user, 'name', '') or getattr(user, 'username', '') or '').split()
    parts = [p for p in parts if p.lower() not in _HONORIFICS] or parts
    return parts[0] if parts else ''


_TEMPLATE_REPLIES = {
    'greeting': [
        "Hi {name}! 👋 Great to see you. What would you like to explore today — a disease, a symptom, a treatment, or maybe some genetics?",
        "Hello {name}! 😊 I'm ready when you are. Ask me anything biomedical, from how a drug works to what causes a condition.",
    ],
    'howareyou': [
        "I'm doing great, thanks for asking, {name}! Always happy to dig into some biomedical questions. What's on your mind?",
    ],
    'thanks': [
        "You're very welcome, {name}! 😊 Feel free to ask a follow-up or start a new topic anytime.",
        "Happy to help! Is there anything else you'd like to know about this, or something new?",
    ],
    'bye': [
        "Goodbye, {name}! Take care, and come back anytime you have a biomedical question. 👋",
    ],
    'identity': [
        "I'm VerifAI, your biomedical question-answering assistant. I search indexed medical literature, "
        "write an answer from the evidence, and verify it against the sources so you can see how well "
        "it's supported. When the literature doesn't cover something, I'll answer from general medical "
        "knowledge and tell you so. What would you like to ask?",
    ],
    'help': [
        "Here's what I can do for you, {name}:\n- Explain diseases, symptoms, causes and risk factors\n"
        "- Describe diagnostics, treatments and how drugs work\n- Cover genetics, molecular biology and physiology\n"
        "- Show the evidence behind each answer and compare it with a classic TF-IDF search\n\n"
        "You can also ask follow-ups like \"what are its side effects?\" and I'll remember the context. "
        "Try: \"What is the role of BRCA1 in breast cancer?\"",
    ],
    'ack': [
        "Glad that helps! Want to go deeper on anything, or explore a new topic?",
    ],
}


def smalltalk_reply(kind, message, user, history):
    """Instant, friendly replies to greetings, thanks, goodbyes and 'what can you do?'."""
    name = _first_name(user) or 'there'
    key = kind if kind in _TEMPLATE_REPLIES else 'greeting'
    return random.choice(_TEMPLATE_REPLIES[key]).format(name=name)


def out_of_scope_reply(question, user):
    name = _first_name(user)
    lead = f"Thanks for asking, {name}! " if name else "Thanks for asking! "
    return (
        lead + "That one is outside what I'm built for, though. I'm VerifAI, a biomedical assistant: "
        "I can help with diseases, symptoms, diagnosis, treatments, drugs, genetics and human biology, "
        "all backed by indexed medical literature. "
        f"Your question (\"{question.strip()}\") isn't a biomedical topic, so I'd rather not guess at it. "
        "Is there something health- or biology-related I can help you with? For example: "
        "\"What are the early signs of Type 2 diabetes?\""
    )


def classify(message, conv):
    """
    Decide how to handle a user message.
    Returns dict: intent ('smalltalk'|'question'), smalltalk_kind, greeted, question, standalone, scope
    """
    greeted, remainder = split_greeting(message)
    kind = smalltalk_kind(remainder) if remainder else None
    if greeted and not remainder:
        return {'intent': 'smalltalk', 'smalltalk_kind': 'greeting', 'greeted': True}
    if kind and (kind != 'help' or len(remainder.split()) <= 8):
        # "thanks! and what about its treatment?" is a question, not just small talk
        raw_tail = dict(_SMALLTALK_PATTERNS)[kind].sub('', remainder, count=1)
        tail = raw_tail.strip(' ,.!?')
        sig = topic_signals(tail) if tail else None
        real_question = tail and ('?' in raw_tail or len(tail.split()) >= 4 or
                                  (sig and (sig['static_strong'] or sig['corpus_strong'])))
        if not real_question or kind in ('identity', 'help'):
            return {'intent': 'smalltalk', 'smalltalk_kind': kind, 'greeted': greeted}
        remainder, greeted = tail, True

    question = remainder
    standalone = question
    if is_follow_up(question, conv.last_topic):
        standalone = contextualize(question, conv.last_topic)
    scope = check_domain(standalone)
    return {'intent': 'question', 'greeted': greeted, 'question': question,
            'standalone': standalone, 'scope': scope}
