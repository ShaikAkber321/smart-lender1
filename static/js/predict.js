let currentStep = 1;

function nextStep(step) {
  if (step === 2 && !validateStep1()) return;
  if (step === 3 && !validateStep2()) { return; }
  if (step === 3) buildSummary();

  document.getElementById(`step-${currentStep}`).classList.add('hidden');
  document.getElementById(`step-${step}`).classList.remove('hidden');

  // Update progress indicators
  document.querySelector(`[data-step="${currentStep}"]`).classList.remove('active');
  document.querySelector(`[data-step="${currentStep}"]`).classList.add('done');
  document.querySelector(`[data-step="${step}"]`).classList.add('active');

  currentStep = step;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function prevStep(step) {
  document.getElementById(`step-${currentStep}`).classList.add('hidden');
  document.getElementById(`step-${step}`).classList.remove('hidden');

  document.querySelector(`[data-step="${currentStep}"]`).classList.remove('active');
  document.querySelector(`[data-step="${currentStep}"]`).classList.remove('done');
  document.querySelector(`[data-step="${step}"]`).classList.remove('done');
  document.querySelector(`[data-step="${step}"]`).classList.add('active');

  currentStep = step;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function validateStep1() {
  // All fields have defaults via checked radio buttons
  return true;
}

function validateStep2() {
  const income = document.getElementById('applicant_income').value;
  if (!income || parseFloat(income) <= 0) {
    showError('applicant_income', 'Please enter a valid applicant income.');
    return false;
  }
  clearError('applicant_income');
  return true;
}

function showError(id, msg) {
  const el = document.getElementById(id);
  el.style.borderColor = '#DC2626';
  let hint = el.nextElementSibling;
  if (hint && hint.classList.contains('form-hint')) {
    hint.textContent = msg;
    hint.style.color = '#DC2626';
  }
}

function clearError(id) {
  const el = document.getElementById(id);
  el.style.borderColor = '';
}

function buildSummary() {
  const form   = document.getElementById('loanForm');
  const data   = new FormData(form);
  const grid   = document.getElementById('summaryGrid');

  const labels = {
    Gender: 'Gender',
    Married: 'Married',
    Dependents: 'Dependents',
    Education: 'Education',
    Self_Employed: 'Employment',
    ApplicantIncome: 'App. Income',
    CoapplicantIncome: 'Co-App. Income',
    Credit_History: 'Credit History',
    LoanAmount: 'Loan Amount (₹K)',
    Loan_Amount_Term: 'Term (months)',
    Property_Area: 'Property Area'
  };

  const formatters = {
    ApplicantIncome: v => `₹${Number(v).toLocaleString()}`,
    CoapplicantIncome: v => `₹${Number(v).toLocaleString()}`,
    LoanAmount: v => `₹${Number(v)}K`,
    Credit_History: v => v === '1' ? '✓ Good Credit' : '✗ Poor / No Credit',
    Self_Employed: v => v === 'Yes' ? 'Self-Employed' : 'Salaried',
  };

  grid.innerHTML = '';
  for (const [key, label] of Object.entries(labels)) {
    const val = data.get(key) || '—';
    const display = formatters[key] ? formatters[key](val) : val;
    grid.innerHTML += `
      <div class="summary-item">
        <span class="s-label">${label}</span>
        <span class="s-value">${display}</span>
      </div>`;
  }
}

// Loading state on submit
document.getElementById('loanForm')?.addEventListener('submit', function() {
  const btn = document.getElementById('submitBtn');
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analysing…';
  btn.disabled = true;
});