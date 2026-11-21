from fastapi import APIRouter, Depends

from services.evaluation_service import aggregate_reports, run_benchmark
from web import require_admin, require_user

router = APIRouter(prefix='/api', tags=['api'])


@router.get('/health', name='api.health')
def health():
    """Liveness probe."""
    from services import knowledge_base, llm_client
    return {'status': 'ok', 'system': 'VerifAI', 'model': llm_client.provider_label(),
            'knowledge_base_topics': knowledge_base.topic_count()}


@router.get('/benchmark', name='api.benchmark')
def benchmark(user=Depends(require_user)):
    """Labelled retrieval benchmark results (baseline vs VerifAI)."""
    return run_benchmark()


@router.get('/metrics', name='api.metrics')
def metrics(admin=Depends(require_admin)):
    """Aggregated, measured answer-quality metrics."""
    agg = aggregate_reports()
    return {'count': agg['count'], 'averages': agg['averages'],
            'statuses': agg['statuses'], 'sources': agg['sources']}
