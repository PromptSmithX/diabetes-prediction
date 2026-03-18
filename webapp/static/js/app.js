/* ============================================================
   DiabetesCheck — Frontend Logic
   Handles: multi-step form, BMI bar, sliders, API call, result rendering
   ============================================================ */

/* ── Step navigation ──────────────────────────────────────────
   Shows the target step, marks previous steps as done,
   and updates the progress indicator circles/lines.
   ──────────────────────────────────────────────────────────── */
function goStep(target) {
  const steps     = document.querySelectorAll('.form-step');
  const indicators = document.querySelectorAll('.progress-steps .step');
  const lines      = document.querySelectorAll('.step-line');
  const total      = steps.length;

  // Determine current step
  let current = 1;
  steps.forEach((s, i) => { if (s.classList.contains('active')) current = i + 1; });

  // Validate forward move: check required visible fields (none required — all optional)
  // Nothing is truly required in this form (imputation covers blanks), so always allow.

  // Update form steps
  steps.forEach((s, i) => {
    s.classList.remove('active');
    if (i + 1 === target) s.classList.add('active');
  });

  // Update step indicators
  indicators.forEach((ind, i) => {
    const n = i + 1;
    ind.classList.remove('active', 'done');
    if (n === target)  ind.classList.add('active');
    if (n < target)    ind.classList.add('done');
  });

  // Update connector lines (there are total-1 lines between steps)
  lines.forEach((line, i) => {
    line.classList.remove('done');
    if (i < target - 1) line.classList.add('done');
  });

  // Scroll form panel to top
  const panel = document.querySelector('.form-panel');
  if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


/* ── BMI visual bar ───────────────────────────────────────────
   Moves marker along the gradient bar and shows category label.
   BMI scale: <18.5 underweight, 18.5-24.9 normal,
              25-29.9 overweight, 30+ obese
   Bar maps 10 → 50 as 0% → 100%
   ──────────────────────────────────────────────────────────── */
function updateBMIBar(val) {
  const fill      = document.getElementById('bmiFill');
  const marker    = document.getElementById('bmiMarker');
  const indicator = document.getElementById('bmiIndicator');

  if (!val || val === '') {
    if (marker)    marker.style.display = 'none';
    if (indicator) { indicator.textContent = '–'; indicator.className = 'bmi-indicator'; }
    return;
  }

  const bmi = parseFloat(val);
  if (isNaN(bmi)) return;

  // Map BMI 10-50 → 0-100%  (clamp)
  const pct = Math.max(0, Math.min(100, (bmi - 10) / 40 * 100));

  if (fill)   fill.style.width  = pct + '%';
  if (marker) { marker.style.display = 'block'; marker.style.left = pct + '%'; }

  // Category
  let cat = '', cls = '';
  if      (bmi < 18.5) { cat = 'Thiếu cân';    cls = 'underweight'; }
  else if (bmi < 25)   { cat = 'Bình thường';   cls = 'normal';      }
  else if (bmi < 30)   { cat = 'Thừa cân';      cls = 'overweight';  }
  else if (bmi < 35)   { cat = 'Béo phì I';     cls = 'obese';       }
  else if (bmi < 40)   { cat = 'Béo phì II';    cls = 'obese';       }
  else                 { cat = 'Béo phì III';   cls = 'obese';       }

  if (indicator) {
    indicator.textContent = cat + ' (' + bmi.toFixed(1) + ')';
    indicator.className   = 'bmi-indicator ' + cls;
  }
}


/* ── Slider value display ─────────────────────────────────────
   Called oninput on range sliders to update displayed number.
   ──────────────────────────────────────────────────────────── */
function updateSliderVal(inputId, displayId) {
  const input   = document.getElementById(inputId);
  const display = document.getElementById(displayId);
  if (input && display) display.textContent = input.value;
}


/* ── Collect form data ────────────────────────────────────────
   Reads all named inputs and returns a flat key→value object.
   ──────────────────────────────────────────────────────────── */
function collectFormData() {
  const form    = document.getElementById('diabetesForm');
  const inputs  = {};
  const data    = new FormData(form);

  // Get all named fields
  for (const [key, value] of data.entries()) {
    inputs[key] = value;
  }

  // Also grab number inputs (FormData may miss unchecked radios / empty numbers)
  ['BMI'].forEach(id => {
    const el = document.getElementById(id);
    if (el && el.value !== '') inputs[id] = el.value;
    else if (!(id in inputs))  inputs[id] = '';
  });

  return inputs;
}


/* ── Gauge animation ──────────────────────────────────────────
   Animates the SVG arc from 0 to the target probability.
   Arc perimeter ≈ 251 px (half-circle radius=80).
   stroke-dashoffset trick: offset = perimeter * (1 - fraction)
   ──────────────────────────────────────────────────────────── */
function animateGauge(prob) {
  const PERIMETER = 251;
  const arc  = document.getElementById('gaugeArc');
  const text = document.getElementById('gaugePct');

  if (!arc || !text) return;

  let current = 0;
  const target  = prob;
  const step    = target / 60;   // ~60 frames

  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    const offset = PERIMETER * (1 - current / 100);
    arc.setAttribute('stroke-dashoffset', offset.toFixed(2));
    text.textContent = Math.round(current) + '%';
    if (current >= target) clearInterval(timer);
  }, 16);
}


/* ── Render result ────────────────────────────────────────────
   Populates the result panel with prediction data from the API.
   ──────────────────────────────────────────────────────────── */
function renderResult(data) {
  // Show result content
  document.getElementById('resultPlaceholder').style.display = 'none';
  document.getElementById('resultContent').style.display     = 'flex';

  // ── Risk card ──────────────────────────────────
  const riskCard = document.getElementById('riskCard');
  const level    = (data.risk_level || 'MEDIUM').toUpperCase();

  riskCard.classList.remove('low', 'medium', 'high');
  if      (level === 'LOW')    riskCard.classList.add('low');
  else if (level === 'HIGH')   riskCard.classList.add('high');
  else                         riskCard.classList.add('medium');

  // Risk icon
  const iconMap = { LOW: '🟢', MEDIUM: '🟡', HIGH: '🔴' };
  document.getElementById('riskIcon').textContent    = iconMap[level] || '⚠️';
  document.getElementById('riskTitle').textContent   = 'Nguy cơ: ' + (data.risk_label || '');
  document.getElementById('riskSubtitle').textContent = data.urgency || '';

  // Gauge
  animateGauge(data.probability || 0);

  // Summary text
  document.getElementById('riskSummary').textContent = data.summary || '';

  // Urgency badge
  const urgencyBadge = document.getElementById('urgencyBadge');
  urgencyBadge.textContent = data.urgency || '';
  urgencyBadge.className   = 'urgency-badge';
  if      (level === 'LOW')  urgencyBadge.classList.add('low');
  else if (level === 'HIGH') urgencyBadge.classList.add('high');
  else                       urgencyBadge.classList.add('medium');

  // ── Imputation notice ──────────────────────────
  const imputeNotice = document.getElementById('imputeNotice');
  const imputeList   = document.getElementById('imputeList');
  const imputed      = data.imputed || [];

  if (imputed.length > 0) {
    imputeNotice.style.display = 'block';
    document.getElementById('imputeTitle').textContent =
      imputed.length + ' trường được tự động điền (thiếu thông tin)';

    imputeList.innerHTML = '';
    imputed.forEach(item => {
      const div = document.createElement('div');
      div.className = 'impute-item';
      const stratLabel = item.strategy === 'mode' ? 'mode (phổ biến nhất)' : 'median (trung vị)';
      div.textContent = item.field + ': sử dụng ' + stratLabel + ' = ' + item.imputed_value;
      imputeList.appendChild(div);
    });
  } else {
    imputeNotice.style.display = 'none';
  }

  // ── Advice cards ──────────────────────────────
  const adviceList = document.getElementById('adviceList');
  adviceList.innerHTML = '';

  const advices = data.advices || [];
  advices.forEach(adv => {
    const card = document.createElement('div');
    card.className = 'advice-card priority-' + (adv.priority || 3);

    card.innerHTML =
      '<div class="advice-icon">' + (adv.icon || '💡') + '</div>' +
      '<div class="advice-body">' +
        '<div class="advice-cat">' + escapeHtml(adv.cat || '') + '</div>' +
        '<div class="advice-detail">' + escapeHtml(adv.detail || '') + '</div>' +
        '<div class="advice-action">→ ' + escapeHtml(adv.action || '') + '</div>' +
      '</div>';

    adviceList.appendChild(card);
  });

  // Scroll result panel into view (mobile)
  const resultPanel = document.getElementById('resultPanel');
  if (resultPanel && window.innerWidth < 900) {
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}


/* ── Escape HTML ──────────────────────────────────────────────
   Prevents XSS when inserting server text into innerHTML.
   ──────────────────────────────────────────────────────────── */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#39;');
}


/* ── Reset form ───────────────────────────────────────────────
   Returns the UI to its initial state (step 1, no result).
   ──────────────────────────────────────────────────────────── */
function resetForm() {
  // Reset form fields
  document.getElementById('diabetesForm').reset();

  // Reset BMI bar
  updateBMIBar('');

  // Reset sliders
  updateSliderVal('MentHlth', 'mentVal');
  updateSliderVal('PhysHlth', 'physVal');

  // Reset gauge arc
  const arc  = document.getElementById('gaugeArc');
  const text = document.getElementById('gaugePct');
  if (arc)  arc.setAttribute('stroke-dashoffset', '251');
  if (text) text.textContent = '0%';

  // Show placeholder, hide result
  document.getElementById('resultPlaceholder').style.display = '';
  document.getElementById('resultContent').style.display     = 'none';

  // Go back to step 1
  goStep(1);

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}


/* ── Form submit handler ──────────────────────────────────────
   Collects form data, calls /predict, renders result.
   ──────────────────────────────────────────────────────────── */
document.getElementById('diabetesForm').addEventListener('submit', async function (e) {
  e.preventDefault();

  const overlay = document.getElementById('loadingOverlay');
  overlay.style.display = 'flex';

  try {
    const inputs = collectFormData();

    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ inputs }),
    });

    if (!response.ok) {
      throw new Error('Server trả về lỗi HTTP ' + response.status);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || 'Lỗi không xác định từ server.');
    }

    renderResult(data);

  } catch (err) {
    alert('Đã xảy ra lỗi: ' + err.message +
          '\n\nVui lòng kiểm tra server Flask đang chạy và thử lại.');
    console.error(err);
  } finally {
    overlay.style.display = 'none';
  }
});


/* ── DOMContentLoaded init ────────────────────────────────────
   Initialise slider displays on page load.
   ──────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  updateSliderVal('MentHlth', 'mentVal');
  updateSliderVal('PhysHlth', 'physVal');
});
