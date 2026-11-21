"""
Domain guard for VerifAI.

Decides whether a question belongs to the biomedical domain BEFORE any
retrieval or answer generation runs. Works fully offline (no API key needed):

  1. Biomedical lexicon  - curated terms/phrases (diseases, anatomy, drugs, ...)
  2. Medical morphology  - suffixes such as -itis, -emia, -ectomy, -mab, -statin
  3. Corpus vocabulary   - keywords of the documents indexed by the admin, so
                           uploading a new dataset automatically widens the domain
  4. Retrieval evidence  - BM25 found topic-aligned passages for the question
  5. LLM check (optional)- OpenAI or local Ollama, only when steps 1-4 are unsure

If none of these signals fire, the question is out of scope and the app shows
OUT_OF_SCOPE_MESSAGE instead of generating an answer.
"""

import os
import re

# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------

# Single words that on their own clearly signal a biomedical question.
STRONG_TERMS = {
    # general medicine / care
    'medical', 'medicine', 'medicines', 'clinical', 'clinic', 'patient', 'patients',
    'doctor', 'physician', 'nurse', 'hospital', 'diagnosis', 'diagnose', 'diagnosed',
    'diagnostic', 'prognosis', 'symptom', 'symptoms', 'syndrome', 'disease', 'diseases',
    'disorder', 'disorders', 'illness', 'infection', 'infections', 'infectious',
    'inflammation', 'inflammatory', 'chronic', 'acute', 'pathology', 'pathogen',
    'pathogenesis', 'pathophysiology', 'etiology', 'aetiology', 'epidemiology',
    'mortality', 'morbidity', 'prevalence', 'incidence', 'comorbidity', 'therapy',
    'therapies', 'therapeutic', 'treatment', 'treatments', 'cure', 'vaccine',
    'vaccines', 'vaccination', 'immunization', 'surgery', 'surgical', 'transplant',
    'biopsy', 'screening', 'prescription', 'dosage', 'dose', 'overdose', 'side-effect',
    'adverse', 'contraindication', 'contraindications', 'remission', 'relapse',
    'injury', 'wound', 'fracture', 'pain', 'fever', 'fatigue', 'nausea', 'vomiting',
    'diarrhea', 'diarrhoea', 'cough', 'headache', 'migraine', 'rash', 'allergy',
    'allergic', 'pregnancy', 'pregnant', 'fertility', 'menstrual', 'menopause',
    'pediatric', 'paediatric', 'geriatric', 'obesity', 'nutrition', 'vitamin',
    'vitamins', 'deficiency', 'healthcare', 'health',
    # diseases & conditions
    'cancer', 'cancers', 'tumor', 'tumour', 'tumors', 'tumours', 'carcinoma', 'malignancy',
    'malignant', 'benign', 'metastasis', 'metastatic', 'neoplasm', 'oncology', 'leukemia',
    'leukaemia', 'lymphoma', 'melanoma', 'sarcoma', 'myeloma', 'glioma', 'glioblastoma',
    'diabetes', 'diabetic', 'insulin', 'hypertension', 'hypotension', 'anemia', 'anaemia',
    'asthma', 'alzheimer', 'alzheimers', 'dementia', 'parkinson', 'parkinsons', 'stroke',
    'epilepsy', 'seizure', 'seizures', 'arthritis', 'osteoporosis', 'lupus', 'sclerosis',
    'copd', 'emphysema', 'bronchitis', 'pneumonia', 'tuberculosis', 'hepatitis',
    'cirrhosis', 'covid', 'covid-19', 'sars-cov-2', 'coronavirus', 'influenza', 'flu',
    'hiv', 'aids', 'malaria', 'dengue', 'cholera', 'measles', 'mumps', 'rubella',
    'polio', 'ebola', 'zika', 'sepsis', 'celiac', 'coeliac', 'crohn', 'colitis',
    'psoriasis', 'eczema', 'dermatitis', 'depression', 'depressive', 'anxiety',
    'schizophrenia', 'bipolar', 'autism', 'adhd', 'ptsd', 'insomnia', 'thrombosis',
    'embolism', 'aneurysm', 'angina', 'arrhythmia', 'atherosclerosis', 'infarction',
    'ischemia', 'ischaemia', 'myocardial', 'cardiovascular', 'cardiac', 'coronary',
    'renal', 'kidney', 'kidneys', 'liver', 'hepatic', 'pulmonary', 'respiratory',
    'gastrointestinal', 'neurological', 'neurodegenerative', 'autoimmune', 'immune',
    'immunity', 'immunology', 'endocrine', 'hormone', 'hormones', 'thyroid', 'gout',
    'sickle', 'thalassemia', 'hemophilia', 'haemophilia', 'cystic', 'fibrosis',
    'obesity', 'cholesterol', 'triglycerides', 'jaundice', 'ulcer', 'ulcers',
    # anatomy & physiology
    'anatomy', 'physiology', 'tissue', 'tissues', 'blood', 'plasma',
    'serum', 'platelet', 'platelets', 'hemoglobin', 'haemoglobin', 'erythrocyte',
    'leukocyte', 'lymphocyte', 'neutrophil', 'eosinophil', 'macrophage', 'antibody',
    'antibodies', 'antigen', 'cytokine', 'cytokines', 'lung', 'lungs', 'brain',
    'neuron', 'neurons', 'nerve', 'nerves', 'spinal', 'muscle', 'muscles', 'bone',
    'bones', 'skeletal', 'skin', 'artery', 'arteries', 'vein', 'veins', 'capillary',
    'pancreas', 'pancreatic', 'stomach', 'intestine', 'intestinal', 'colon', 'bowel',
    'bladder', 'prostate', 'ovary', 'ovarian', 'uterus', 'cervix', 'cervical',
    'breast', 'retina', 'cornea', 'cardiology', 'neurology', 'nephrology',
    'gastroenterology', 'dermatology', 'radiology', 'hematology', 'haematology',
    'metabolism', 'metabolic', 'glucose', 'hyperglycemia', 'hypoglycemia',
    # molecular biology & genetics
    'dna', 'rna', 'mrna', 'trna', 'rrna', 'gene', 'genes', 'genetic', 'genetics',
    'genome', 'genomic', 'genomics', 'chromosome', 'chromosomes', 'mutation',
    'mutations', 'allele', 'alleles', 'genotype', 'phenotype', 'hereditary',
    'protein', 'proteins', 'enzyme', 'enzymes', 'amino', 'peptide', 'receptor',
    'receptors', 'mitochondria', 'mitochondrial', 'organelle', 'ribosome', 'nucleotide',
    'transcription', 'apoptosis', 'biomarker', 'biomarkers',
    'biomedical', 'biochemistry', 'biochemical', 'molecular', 'microbiome',
    'bacteria', 'bacterial', 'viral', 'fungal', 'parasite', 'parasitic', 'microbe',
    'microbial', 'pathogens', 'crispr', 'pcr', 'brca', 'brca1', 'brca2', 'tp53', 'kras',
    'egfr', 'her2',
    # pharmacology
    'drug', 'drugs', 'medication', 'medications', 'pharmacology', 'pharmacological',
    'pharmaceutical', 'antibiotic', 'antibiotics', 'antiviral', 'antivirals',
    'chemotherapy', 'immunotherapy', 'radiotherapy', 'analgesic',
    'steroid', 'steroids', 'corticosteroid', 'corticosteroids', 'aspirin', 'insulin',
    'metformin', 'paracetamol', 'acetaminophen', 'ibuprofen', 'penicillin', 'inhibitor',
    'inhibitors', 'agonist', 'antagonist', 'placebo', 'clinical-trial',
}

# Multi-word phrases (matched as substrings of the normalized question).
STRONG_PHRASES = {
    'blood pressure', 'heart attack', 'heart disease', 'heart failure', 'heart rate',
    'side effect', 'side effects', 'clinical trial', 'clinical trials', 'white blood',
    'red blood', 'immune system', 'nervous system', 'amino acid', 'amino acids',
    'double helix', 'cell division', 'cell cycle', 'stem cell', 'stem cells',
    'public health', 'mental health', 'sickle cell', 'cystic fibrosis',
    'multiple sclerosis', 'iron deficiency', 'chest pain', 'weight loss',
    'blood sugar', 'body mass index', 'human body',
}

# Words that are medical only in context ("cell phone", "computer virus").
AMBIGUOUS_TERMS = {
    'cell', 'cells', 'virus', 'viruses', 'heart', 'cold', 'pressure', 'shot', 'shots',
    'operation', 'operations', 'bug', 'bugs', 'mouse', 'recovery', 'healthy', 'diet',
    'exercise', 'sleep', 'stress', 'eye', 'eyes', 'ear', 'ears', 'teeth', 'tooth',
    'body', 'weight', 'age', 'aging', 'ageing', 'growth', 'test', 'tests', 'scan',
    'treat', 'stem', 'organ', 'organs', 'pathway', 'radiation',
}

# Words that strongly suggest a non-biomedical topic.
OFF_DOMAIN_TERMS = {
    'cricket', 'football', 'soccer', 'basketball', 'tennis', 'hockey', 'baseball',
    'team', 'teams', 'player', 'players', 'match', 'tournament', 'score', 'goal',
    'movie', 'movies', 'film', 'films', 'actor', 'actress', 'song', 'songs', 'music',
    'album', 'singer', 'computer', 'laptop', 'software', 'hardware', 'phone', 'mobile',
    'smartphone', 'app', 'website', 'internet', 'wifi', 'battery', 'python', 'java',
    'javascript', 'code', 'coding', 'programming', 'database', 'server', 'stock',
    'stocks', 'share', 'market', 'price', 'prices', 'crypto', 'bitcoin', 'bank',
    'loan', 'tax', 'salary', 'job', 'recipe', 'cook', 'cooking', 'restaurant',
    'weather', 'election', 'president', 'minister', 'politics', 'government', 'war',
    'car', 'cars', 'bike', 'engine', 'travel', 'hotel', 'flight', 'tourism', 'game',
    'games', 'gaming', 'capital', 'country', 'history', 'king', 'queen', 'celebrity',
}

# Medical word-formation suffixes (checked on words of length >= 7).
MEDICAL_SUFFIXES = (
    'itis', 'osis', 'emia', 'aemia', 'oma', 'pathy', 'ectomy', 'otomy', 'ostomy',
    'plasty', 'scopy', 'algia', 'plasia', 'trophy', 'penia', 'cytosis', 'cyte', 'cytes',
    'genesis', 'uria', 'mab', 'tinib', 'nib', 'vir', 'statin', 'cillin', 'mycin',
    'cycline', 'azole', 'pril', 'sartan', 'olol', 'dipine', 'gliptin', 'gliflozin',
    'prazole', 'parin', 'oxacin', 'thiazide', 'dronate',
)

# Common words that happen to end in a medical suffix.
SUFFIX_EXCEPTIONS = {
    'diploma', 'aroma', 'stroma', 'karma', 'cinema', 'panorama', 'enigma', 'dogma',
    'oklahoma', 'sympathy', 'apathy', 'empathy', 'telepathy', 'philosophy',
    'hypnosis', 'osmosis', 'genesis',
}

_STOPWORDS = {
    'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 'is', 'are',
    'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'the', 'and',
    'for', 'with', 'about', 'into', 'from', 'can', 'could', 'will', 'would', 'should',
    'there', 'their', 'this', 'that', 'these', 'those', 'many', 'much', 'some', 'any',
    'tell', 'explain', 'describe', 'give', 'list', 'show', 'please', 'between', 'more',
    'most', 'other', 'than', 'then', 'also', 'very', 'such', 'only', 'used', 'using',
    'major', 'common', 'new', 'role', 'type', 'types', 'factors', 'factor', 'risk',
    'management', 'guidelines', 'strategies', 'dual', 'rapid', 'acting', 'targeted',
}

OUT_OF_SCOPE_STATUS = 'Out of Scope'

OUT_OF_SCOPE_MESSAGE = (
    "This question doesn't appear to be related to what VerifAI is built for. "
    "VerifAI is a biomedical question-answering assistant: it answers questions about "
    "diseases, symptoms, diagnosis, treatments, drugs, genetics and human biology, "
    "backed by indexed medical literature. Your question \"{question}\" falls outside "
    "that scope, so no answer was generated. Please ask a biomedical question instead, "
    "for example: \"What are the common symptoms of Type 2 diabetes?\""
)


def _normalize(text):
    text = (text or '').lower()
    text = text.replace("'s", ' ').replace('’s', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def _tokens(text):
    return re.findall(r'[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]', text)


# --------------------------------------------------------------------------
# Corpus vocabulary (keywords of documents the admin has indexed)
# --------------------------------------------------------------------------

_corpus_cache = {'doc_count': -1, 'terms': set(), 'phrases': set()}


def _corpus_vocabulary():
    """Terms from indexed document keywords and knowledge-base topic names; cached and
    refreshed when the corpus or the knowledge base changes."""
    from services import knowledge_base
    kb_terms = {t for t in knowledge_base.subject_terms() if t not in _STOPWORDS}
    kb_phrases = knowledge_base.subject_phrases()
    kb_key = knowledge_base._dir_mtime()
    try:
        from models.document import Document
        count = (Document.query.count(), kb_key)
        if count == _corpus_cache['doc_count']:
            return _corpus_cache['terms'], _corpus_cache['phrases']
        terms, phrases = set(kb_terms), set(kb_phrases)
        for (keywords,) in Document.query.with_entities(Document.keywords).all():
            for kw in (keywords or '').split(','):
                kw = _normalize(kw)
                if not kw:
                    continue
                if ' ' in kw:
                    phrases.add(kw)          # e.g. 'dna repair' only as a phrase
                elif len(kw) >= 3 and kw not in _STOPWORDS:
                    terms.add(kw)
        _corpus_cache.update(doc_count=count, terms=terms, phrases=phrases)
        return terms, phrases
    except Exception:
        # No app context / DB not ready: static lexicon + knowledge base only.
        return kb_terms, kb_phrases


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def _has_medical_suffix(token):
    if len(token) < 7 or token in SUFFIX_EXCEPTIONS:
        return False
    return token.endswith(MEDICAL_SUFFIXES)


def _retrieval_supports(question):
    """True when BM25 finds passages that pass its strict topic filter."""
    try:
        from services.retrieval_service import proposed_retrieve, GENERIC_TERMS
        q_words = set(re.findall(r'\b[a-z0-9\-]{3,}\b', question.lower()))
        if not (q_words - GENERIC_TERMS - _STOPWORDS):
            return False   # nothing topical to search for
        passages, _ = proposed_retrieve(question, top_k=1)
        return bool(passages)
    except Exception:
        return False


def _llm_says_biomedical(question):
    """Tie-breaker using the configured LLM (OpenAI or local Ollama). Returns True/False/None."""
    try:
        from services import llm_client
        if not llm_client.is_available():
            return None
        reply = llm_client.chat([
            {'role': 'system', 'content': (
                'You are a strict classifier. Reply with exactly one word: YES if the '
                'question is about biomedicine, medicine, health, human/animal biology, '
                'genetics, pharmacology or clinical science; otherwise NO.')},
            {'role': 'user', 'content': question},
        ], max_tokens=3, temperature=0)
        return reply.strip().upper().startswith('YES')
    except Exception as e:
        print(f'Domain guard LLM check failed: {e}')
        return None


def topic_signals(question):
    """
    Lexical analysis shared by the domain check and the chat follow-up detector.
    Returns sets: static_strong, corpus_strong, ambiguous, off_domain.
    """
    q = _normalize(question)
    toks = _tokens(q)
    tokset = set(toks)
    # Light plural handling: 'injuries' -> 'injury', 'fractures' -> 'fracture'
    for t in toks:
        if len(t) > 4 and t.endswith('ies'):
            tokset.add(t[:-3] + 'y')
        elif len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
            tokset.add(t[:-1])

    corpus_terms, corpus_phrases = _corpus_vocabulary()

    static_strong = tokset & STRONG_TERMS
    static_strong |= {p for p in STRONG_PHRASES if p in q}
    static_strong |= {t for t in toks if _has_medical_suffix(t)}
    return {
        'static_strong': static_strong,
        'corpus_strong': (tokset & corpus_terms) | {p for p in corpus_phrases if p in q},
        'ambiguous': tokset & AMBIGUOUS_TERMS,
        'off_domain': tokset & OFF_DOMAIN_TERMS,
    }


def check_domain(question, use_llm=True):
    """
    Returns dict: {'in_scope': bool, 'reason': str, 'message': str|None}
    """
    sig = topic_signals(question)
    static_strong, corpus_strong = sig['static_strong'], sig['corpus_strong']
    ambiguous, off_domain = sig['ambiguous'], sig['off_domain']

    if static_strong:
        return {'in_scope': True, 'reason': f'biomedical terms: {sorted(static_strong)[:5]}', 'message': None}

    # Corpus keywords and context-dependent words only count without off-domain context
    # (so 'virus' passes, but 'computer virus' does not).
    if (corpus_strong or ambiguous) and not off_domain:
        hits = sorted(corpus_strong | ambiguous)[:5]
        return {'in_scope': True, 'reason': f'corpus/context terms: {hits}', 'message': None}

    if _retrieval_supports(question):
        return {'in_scope': True, 'reason': 'matching passages found in indexed corpus', 'message': None}

    # Ask the LLM only when nothing clearly points to another domain.
    if use_llm and not off_domain and _llm_says_biomedical(question):
        return {'in_scope': True, 'reason': 'LLM classifier', 'message': None}

    return {
        'in_scope': False,
        'reason': f'no biomedical signal (off-domain terms: {sorted(off_domain)})',
        'message': OUT_OF_SCOPE_MESSAGE.format(question=question.strip()),
    }


def is_biomedical_question(question):
    return check_domain(question)['in_scope']
