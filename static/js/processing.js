// VerifAI Processing Animation

const STEPS = [
  { id: 1, label: 'Uploading document...' },
  { id: 2, label: 'Preprocessing text...' },
  { id: 3, label: 'Extracting features...' },
  { id: 4, label: 'Semantic indexing...' },
  { id: 5, label: 'Running Existing Model (TF-IDF)...' },
  { id: 6, label: 'Running VerifAI Model (BM25+AI)...' },
  { id: 7, label: 'Extracting evidence passages...' },
  { id: 8, label: 'Verifying answer...' },
  { id: 9, label: 'Computing metrics...' },
  { id: 10, label: 'Generating results...' },
];

const STEP_DURATIONS = [600, 800, 700, 900, 1000, 1200, 800, 700, 600, 500];

let currentStep = 0;
let sessionId = null;
let processInterval = null;
let pipelineCancelled = false;

// Show a friendly "outside VerifAI's scope" notice above the submitted form.
function showScopeNotice(form, message) {
  const anchor = form.closest('.content-card') || form;
  let notice = document.getElementById('scopeNotice');
  if (!notice) {
    notice = document.createElement('div');
    notice.id = 'scopeNotice';
    notice.className = 'flash-message flash-info';
    notice.setAttribute('role', 'status');
    notice.innerHTML = '<i class="fas fa-circle-info"></i><span></span>';
    anchor.parentNode.insertBefore(notice, anchor);
  }
  notice.querySelector('span').textContent = message;
  notice.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showProcessingModal() {
  const modal = document.getElementById('processingModal');
  if (modal) {
    modal.style.display = 'flex';
    currentStep = 0;
    renderSteps();
    updateProgress(0);
  }
}

function hideProcessingModal() {
  const modal = document.getElementById('processingModal');
  if (modal) modal.style.display = 'none';
  if (processInterval) clearTimeout(processInterval);
}

function renderSteps() {
  const container = document.getElementById('stepList');
  if (!container) return;
  container.innerHTML = STEPS.map((s, i) => `
    <div class="step-item" id="step-${s.id}" style="transition: all 0.3s;">
      <div class="step-num pending" id="step-num-${s.id}">${s.id}</div>
      <span class="step-text">${s.label}</span>
      <div class="step-spinner" id="step-spinner-${s.id}" style="display:none;"></div>
      <span class="step-done-icon" id="step-done-${s.id}" style="display:none; margin-left:auto; color:#4CAF50; font-size:16px;">✓</span>
    </div>
  `).join('');
}

function activateStep(stepIndex) {
  if (stepIndex >= STEPS.length) return;
  const step = STEPS[stepIndex];
  const item = document.getElementById(`step-${step.id}`);
  const numEl = document.getElementById(`step-num-${step.id}`);
  const spinner = document.getElementById(`step-spinner-${step.id}`);
  if (item) item.classList.add('active');
  if (numEl) { numEl.className = 'step-num active'; }
  if (spinner) spinner.style.display = 'block';
  const progress = ((stepIndex + 1) / STEPS.length) * 100;
  updateProgress(progress);
}

function completeStep(stepIndex) {
  if (stepIndex >= STEPS.length) return;
  const step = STEPS[stepIndex];
  const item = document.getElementById(`step-${step.id}`);
  const numEl = document.getElementById(`step-num-${step.id}`);
  const spinner = document.getElementById(`step-spinner-${step.id}`);
  const doneIcon = document.getElementById(`step-done-${step.id}`);
  if (item) { item.classList.remove('active'); item.classList.add('done'); }
  if (numEl) { numEl.className = 'step-num done'; numEl.textContent = '✓'; }
  if (spinner) spinner.style.display = 'none';
  if (doneIcon) doneIcon.style.display = 'block';
}

function updateProgress(percent) {
  const fill = document.getElementById('progressFill');
  const label = document.getElementById('progressLabel');
  if (fill) fill.style.width = percent + '%';
  if (label) label.textContent = Math.round(percent) + '%';
}

function updateStatusText(text) {
  const el = document.getElementById('processingStatus');
  if (el) el.textContent = text;
}

function runAnimationSequence(stepIndex, onComplete) {
  if (pipelineCancelled) return;
  if (stepIndex >= STEPS.length) {
    updateProgress(100);
    updateStatusText('Processing complete!');
    if (onComplete) setTimeout(onComplete, 600);
    return;
  }
  activateStep(stepIndex);
  updateStatusText(STEPS[stepIndex].label);
  processInterval = setTimeout(() => {
    completeStep(stepIndex);
    runAnimationSequence(stepIndex + 1, onComplete);
  }, STEP_DURATIONS[stepIndex]);
}

function submitQuestionWithAnimation(formId) {
  const form = document.getElementById(formId);
  if (!form) return;
  const question = form.querySelector('[name=question]')?.value?.trim();
  const file = form.querySelector('[name=test_file]')?.files?.[0];
  if (!question && !file) {
    alert('Please enter a question or upload a file.');
    return;
  }
  const oldNotice = document.getElementById('scopeNotice');
  if (oldNotice) oldNotice.remove();
  sessionId = null;
  pipelineCancelled = false;
  showProcessingModal();
  const formData = new FormData(form);
  const totalDuration = STEP_DURATIONS.reduce((a, b) => a + b, 0);
  
  // Start fetch in background
  fetch(form.action, { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.out_of_scope) {
        // Not a biomedical question: stop the pipeline animation and explain why.
        pipelineCancelled = true;
        hideProcessingModal();
        showScopeNotice(form, data.message);
        return;
      }
      sessionId = data.success ? data.session_id : null;
    })
    .catch(err => console.error('Submission error:', err));
  
  // Run animation sequence
  runAnimationSequence(0, () => {
    if (pipelineCancelled) return;
    if (sessionId) {
      window.location.href = '/user/comparison/' + sessionId;
    } else {
      // Wait a bit more for the request to complete
      setTimeout(() => {
        if (pipelineCancelled) return;
        if (sessionId) {
          window.location.href = '/user/comparison/' + sessionId;
        } else {
          hideProcessingModal();
          alert('Processing failed. Please ensure datasets are uploaded and indexed by the admin.');
        }
      }, 2000);
    }
  });
}

// File upload drag & drop
function initUploadZone(zoneId, inputId) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      input.files = dt.files;
      updateFileLabel(zoneId, files[0].name);
    }
  });
  input.addEventListener('change', (e) => {
    if (e.target.files.length > 0) updateFileLabel(zoneId, e.target.files[0].name);
  });
}

function updateFileLabel(zoneId, filename) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;
  const label = zone.querySelector('.upload-filename');
  if (label) {
    label.textContent = '📄 ' + filename;
    label.style.color = '#2563EB';
    label.style.fontWeight = '700';
  }
}

// Auto-dismiss flash messages
document.addEventListener('DOMContentLoaded', function() {
  const flashMessages = document.querySelectorAll('.flash-message[data-auto-dismiss]');
  flashMessages.forEach(msg => {
    setTimeout(() => {
      msg.style.opacity = '0';
      msg.style.transform = 'translateY(-10px)';
      msg.style.transition = 'all 0.4s ease';
      setTimeout(() => msg.remove(), 400);
    }, 4000);
  });

  // Animate confidence bars
  document.querySelectorAll('.confidence-fill, .betterment-fill, .relevance-fill').forEach(bar => {
    const target = bar.dataset.width || '0';
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = target + '%'; }, 200);
  });
});
