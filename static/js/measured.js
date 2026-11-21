// Charts for admin evaluation pages. Data: services/evaluation_service.aggregate_reports()
// Each chart is drawn only if its canvas exists on the page.
function renderMeasured(agg) {
  if (!window.Chart || !agg) return;
  const BASE = '#94A3B8', VF = '#2563EB', OK = '#059669', BAD = '#DC2626';
  const S = agg.series || { labels: [], baseline: {}, verifai: {} };
  const pctAxis = { min: 0, max: 1, ticks: { callback: v => Math.round(v * 100) + '%' } };
  const el = id => document.getElementById(id);
  const line = (label, data, color, dashed) => ({
    label, data, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 2,
    tension: 0.3, borderDash: dashed ? [5, 4] : [], spanGaps: true,
  });

  if (el('mTrend')) new Chart(el('mTrend'), {
    type: 'line',
    data: { labels: S.labels, datasets: [
      line('VerifAI · question coverage', S.verifai.question_coverage, VF),
      line('Baseline · question coverage', S.baseline.question_coverage, BASE),
      line('VerifAI · clinical breadth', S.verifai.aspect_breadth, '#0D9488', true),
      line('Baseline · clinical breadth', S.baseline.aspect_breadth, '#CBD5E1', true) ] },
    options: { scales: { y: pctAxis }, plugins: { legend: { position: 'bottom' } } }
  });

  if (el('mLatency')) new Chart(el('mLatency'), {
    type: 'line',
    data: { labels: S.labels, datasets: [
      line('VerifAI · BM25', S.verifai.latency_ms, VF), line('Baseline · TF-IDF', S.baseline.latency_ms, BASE) ] },
    options: { scales: { y: { beginAtZero: true, ticks: { callback: v => v + ' ms' } } }, plugins: { legend: { position: 'bottom' } } }
  });

  const pctMetrics = (agg.metrics || []).filter(m => m.kind === 'pct');
  if (el('mAverages')) new Chart(el('mAverages'), {
    type: 'bar',
    data: { labels: pctMetrics.map(m => m.name), datasets: [
      { label: 'Baseline · TF-IDF', data: pctMetrics.map(m => agg.averages[m.key].baseline), backgroundColor: BASE, borderRadius: 4 },
      { label: 'VerifAI · BM25 + LLM', data: pctMetrics.map(m => agg.averages[m.key].verifai), backgroundColor: VF, borderRadius: 4 } ] },
    options: { scales: { y: pctAxis }, plugins: { legend: { position: 'bottom' } } }
  });

  const changed = (agg.metrics || []).filter(m => agg.averages[m.key].change_pct !== null);
  if (el('mImprove')) new Chart(el('mImprove'), {
    type: 'bar',
    data: { labels: changed.map(m => m.name), datasets: [{
      label: 'VerifAI vs baseline (%)', data: changed.map(m => agg.averages[m.key].change_pct),
      backgroundColor: changed.map(m => agg.averages[m.key].change_pct >= 0 ? OK : BAD), borderRadius: 4 }] },
    options: { indexAxis: 'y', scales: { x: { ticks: { callback: v => v + '%' } } }, plugins: { legend: { display: false } } }
  });

  const st = agg.statuses || {};
  const stColors = { 'Verified': OK, 'Partially Verified': '#D97706', 'General Knowledge': VF,
                     'Insufficient Evidence': BASE, 'Conflicting Evidence': '#7C3AED' };
  if (el('mStatus')) new Chart(el('mStatus'), {
    type: 'doughnut',
    data: { labels: Object.keys(st), datasets: [{ data: Object.values(st),
      backgroundColor: Object.keys(st).map(k => stColors[k] || '#CBD5E1'), borderWidth: 0 }] },
    options: { cutout: '62%', plugins: { legend: { position: 'bottom' } } }
  });
}
