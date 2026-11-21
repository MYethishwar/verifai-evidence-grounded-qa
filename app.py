"""
VerifAI - FastAPI application.

Run:  python app.py            (development server with auto-reload on http://127.0.0.1:5000)
 or:  uvicorn app:app --port 5000
API docs: /docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from config import Config
from extensions import db
from web import BASE_DIR, SignInRequired, flash, redirect, render, url_for_name


def init_database():
    """Create tables, seed the admin account and sample data, and build the search index."""
    import models  # noqa: F401  (registers every model)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)   # SQLite won't create the folder
    db.create_all()
    with db.request_scope():
        seed_admin()
        seed_sample_biomedical_data()


def create_app():
    db.init(Config.SQLALCHEMY_DATABASE_URI)
    init_database()

    @asynccontextmanager
    async def lifespan(_app):
        # Load the local model and pre-process the fixed system prompt in the background,
        # so the first question doesn't wait for the model to load.
        import threading
        from services import llm_client
        from services.ai_service import PERSONA
        threading.Thread(target=llm_client.warm_up, args=(PERSONA,), daemon=True).start()
        yield
        if db.engine is not None:
            db.engine.dispose()

    app = FastAPI(title='VerifAI', version='2.0',
                  description='Evidence-grounded biomedical question answering.', lifespan=lifespan)

    # One database session per request (closed afterwards, even on errors)
    @app.middleware('http')
    async def db_session_per_request(request: Request, call_next):
        with db.request_scope():
            return await call_next(request)

    # Signed session cookie: sign-in state, current conversation and flash messages
    app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY,
                       max_age=int(Config.PERMANENT_SESSION_LIFETIME.total_seconds()),
                       same_site='lax', session_cookie='verifai_session')

    app.mount('/static', StaticFiles(directory=os.path.join(BASE_DIR, 'static')), name='static')

    @app.exception_handler(SignInRequired)
    async def _sign_in_required(request: Request, exc: SignInRequired):
        if request.url.path.startswith(('/api/', '/user/chat')) or request.method != 'GET':
            from fastapi.responses import JSONResponse
            return JSONResponse({'success': False, 'error': exc.message or 'Please sign in.'}, status_code=401)
        if exc.message:
            flash(request, exc.message, exc.category)
        return redirect(url_for_name(request, 'auth.login'))

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith('/api/') or 'application/json' in request.headers.get('accept', ''):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)
        return HTMLResponse(
            f'<div style="font-family:Inter,system-ui;padding:64px;text-align:center;color:#0F172A">'
            f'<h1 style="font-size:48px;margin:0">{exc.status_code}</h1>'
            f'<p style="color:#64748B">{exc.detail}</p><a href="/">Back to VerifAI</a></div>',
            status_code=exc.status_code)

    from routes.admin import router as admin_router
    from routes.api import router as api_router
    from routes.auth import router as auth_router
    from routes.user import router as user_router
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(admin_router)
    app.include_router(api_router)
    return app


def seed_admin():
    from models.user import User
    from config import Config
    from extensions import db
    if not User.query.filter_by(username=Config.ADMIN_USERNAME).first():
        admin = User(
            name='Administrator',
            username=Config.ADMIN_USERNAME,
            email='admin@verifai.com',
            mobile='0000000000',
            role='admin',
            status='active'
        )
        admin.set_password(Config.ADMIN_PASSWORD)
        db.session.add(admin)
        db.session.commit()
        print('Admin user seeded.')

def seed_sample_biomedical_data():
    from models.document import Dataset, Document, Passage
    from models.user import User
    from services.preprocessing_service import preprocess_text, segment_passages
    from services.retrieval_service import build_index_from_db
    from extensions import db

    if Dataset.query.first():
        build_index_from_db()
        return

    admin = User.query.filter_by(role='admin').first()
    admin_id = admin.id if admin else 1

    sample_dataset = Dataset(
        name='PubMed Open Biomedical Knowledge Base',
        filename='pubmed_biomedical_sample.csv',
        file_path='internal_seed',
        description='Curated biomedical abstracts covering Type 2 Diabetes, BRCA1 & Breast Cancer, Iron Deficiency Anemia, Hypertension, and Alzheimer Disease.',
        records_count=6,
        status='indexed',
        uploaded_by=admin_id
    )
    db.session.add(sample_dataset)
    db.session.flush()

    sample_articles = [
        {
            'title': 'Pathophysiology, Symptoms, and Risk Factors of Type 2 Diabetes Mellitus',
            'abstract': 'Type 2 diabetes mellitus is a chronic metabolic disorder characterized by persistent hyperglycemia resulting from progressive insulin resistance and impaired pancreatic beta-cell insulin secretion. Common symptoms include polydipsia, polyuria, unexplained weight loss, fatigue, blurry vision, and slow-healing sores. Major risk factors include obesity (especially central adiposity), physical inactivity, advancing age, genetic predisposition, family history, gestational diabetes history, and hypertension. Early diagnosis and lifestyle interventions combining diet and aerobic exercise significantly mitigate long-term microvascular and macrovascular complications.',
            'authors': 'Harrison T, Mitchell K, Reynolds S',
            'journal': 'The Lancet Diabetes & Endocrinology',
            'year': '2023',
            'pmid': '34981204',
            'doi': '10.1016/S2213-8587(23)00112-X',
            'keywords': 'type 2 diabetes, insulin resistance, hyperglycemia, symptoms, obesity, risk factors'
        },
        {
            'title': 'BRCA1 and BRCA2 Tumor Suppressor Genes in Hereditary Breast and Ovarian Cancer',
            'abstract': 'BRCA1 is a critical tumor suppressor gene located on chromosome 17q21 that encodes a nuclear phosphoprotein maintaining genomic stability through homologous recombination DNA double-strand break repair. Deleterious germline mutations in BRCA1 markedly elevate lifetime susceptibility to hereditary breast cancer (up to 72% cumulative risk by age 80) and epithelial ovarian cancer (up to 44% risk). BRCA1-deficient tumors frequently present as high-grade triple-negative breast carcinomas with distinct genomic instability and heightened sensitivity to poly(ADP-ribose) polymerase (PARP) inhibitors such as olaparib.',
            'authors': 'Chen L, Al-Mansoor J, Patel V, O’Connor D',
            'journal': 'New England Journal of Medicine',
            'year': '2022',
            'pmid': '35129987',
            'doi': '10.1056/NEJMoa2201945',
            'keywords': 'BRCA1, breast cancer, homologous recombination, DNA repair, PARP inhibitors, genetics'
        },
        {
            'title': 'Clinical Manifestations, Etiology, and Management of Iron Deficiency Anemia',
            'abstract': 'Iron deficiency anemia (IDA) is the leading cause of microcytic hypochromic anemia globally. It develops when total body iron stores become depleted due to inadequate dietary intake, impaired gastrointestinal absorption (e.g., celiac disease), or chronic blood loss. Hallmarks and symptoms include extreme fatigue, physical weakness, pallor of the skin and conjunctiva, dyspnea on exertion, lightheadedness, headache, brittle nails, cold extremities, and pica (craving non-nutritive substances). Diagnostic confirmation requires low serum ferritin (<30 ng/mL), diminished transferrin saturation, and decreased hemoglobin. First-line management comprises oral or intravenous iron supplementation alongside identifying underlying sources of bleeding.',
            'authors': 'Goddard AF, James MW, McIntyre AS, Scott BB',
            'journal': 'British Medical Journal (BMJ)',
            'year': '2023',
            'pmid': '36201889',
            'doi': '10.1136/bmj.2023.072115',
            'keywords': 'iron deficiency anemia, fatigue, ferritin, hemoglobin, pallor, microcytic anemia'
        },
        {
            'title': 'Mechanisms of Essential Hypertension, Vascular Remodeling, and Therapeutic Targets',
            'abstract': 'Essential hypertension arises from complex interactions between elevated systemic vascular resistance, chronic renin-angiotensin-aldosterone system (RAAS) overactivation, arterial stiffness, endothelial dysfunction, and sympathetic hyperactivity. Clinical manifestations often remain asymptomatic until target organ damage ensues, including hypertensive retinopathy, left ventricular hypertrophy, and nephrosclerosis. First-line pharmacological interventions include ACE inhibitors, angiotensin receptor blockers (ARBs), calcium channel blockers, and thiazide diuretics, complemented by dietary sodium restriction.',
            'authors': 'Williams B, Mancia G, Spiering W, Agabiti Rosei E',
            'journal': 'Journal of Hypertension',
            'year': '2023',
            'pmid': '35891440',
            'doi': '10.1097/HJH.0000000000003200',
            'keywords': 'hypertension, blood pressure, RAAS, ACE inhibitors, arterial stiffness, cardiovascular'
        },
        {
            'title': 'Amyloid-Beta Cascade, Tau Pathology, and Diagnostic Biomarkers in Alzheimer Disease',
            'abstract': 'Alzheimer disease is the predominant neurodegenerative etiology of dementia worldwide. Pathological hallmarks feature extracellular amyloid-beta peptide plaques and intracellular neurofibrillary tangles composed of hyperphosphorylated tau proteins, causing progressive synaptic failure and cerebral cortical atrophy. Established cerebrospinal fluid (CSF) and plasma biomarkers include reduced Aβ42/Aβ40 ratios alongside elevated phosphorylated tau-181 (p-tau181) and neurofilament light chain (NfL). Monoclonal antibodies targeting amyloid oligomers have recently demonstrated clinical slowing of cognitive decline.',
            'authors': 'Jack CR, Bennett DA, Blennow K, Carrillo MC',
            'journal': 'Nature Reviews Neurology',
            'year': '2023',
            'pmid': '36412890',
            'doi': '10.1038/s41582-023-00780-z',
            'keywords': 'Alzheimer disease, amyloid-beta, tau pathology, dementia, neurodegeneration, biomarkers'
        },
        {
            'title': 'Immunopathogenesis of Bronchial Asthma and Biologic Therapeutics',
            'abstract': 'Bronchial asthma is a chronic inflammatory airway disease characterized by variable airflow obstruction, bronchial hyperresponsiveness, and airway remodeling. Common clinical symptoms encompass recurrent wheezing, breathlessness, chest tightness, and nocturnal cough. T-helper 2 (Th2) cytokines including IL-4, IL-5, and IL-13 drive eosinophilic inflammation and IgE synthesis. Contemporary management escalates from inhaled corticosteroids and long-acting beta2-agonists (ICS-LABA) to targeted monoclonal biologics such as dupilumab, mepolizumab, and omalizumab for severe refractory phenotypes.',
            'authors': 'Wenzel SE, Fahy JV, Holgate ST, Israel E',
            'journal': 'American Journal of Respiratory and Critical Care Medicine',
            'year': '2023',
            'pmid': '36780120',
            'doi': '10.1164/rccm.202301-0120OC',
            'keywords': 'asthma, eosinophil, bronchospasm, inhaled corticosteroids, biologics, airway inflammation'
        }
    ]

    for item in sample_articles:
        combined = f"{item['title']} {item['abstract']}"
        preprocessed = preprocess_text(combined)
        doc = Document(
            dataset_id=sample_dataset.id,
            title=item['title'],
            abstract=item['abstract'],
            authors=item['authors'],
            journal=item['journal'],
            year=item['year'],
            pmid=item['pmid'],
            doi=item['doi'],
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{item['pmid']}/",
            keywords=item['keywords'],
            preprocessed_text=preprocessed,
            is_preprocessed=True,
            is_indexed=True
        )
        db.session.add(doc)
        db.session.flush()

        passages = segment_passages(item['abstract'], max_words=90)
        if not passages:
            passages = [item['abstract']]
        for idx, p_text in enumerate(passages):
            p = Passage(
                document_id=doc.id,
                content=p_text,
                passage_index=idx,
                word_count=len(p_text.split())
            )
            db.session.add(p)

    db.session.commit()
    build_index_from_db()
    print('Sample biomedical knowledge base seeded & indexed.')


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='127.0.0.1', port=5000, reload=True,
                reload_excludes=['instance/*', 'static/uploads/*', '*.db*'])
