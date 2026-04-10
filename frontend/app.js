/* ═══════════════════════════════════════════════════════════════════════
   Promptee — Frontend Application Logic
   Connects to FastAPI backend at /optimize_prompt
   ═══════════════════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:8000";
const RING_CIRCUMFERENCE = 326.73; // 2 * π * 52

// ── DOM References ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const promptInput     = $("prompt-input");
const charCount       = $("char-count");
const optimizeBtn     = $("optimize-btn");
const btnLoader       = $("btn-loader");
const resultsDiv      = $("results");
const errorBanner     = $("error-banner");
const errorText       = $("error-text");
const errorClose      = $("error-close");
const statusDot       = $("status-dot");
const statusText      = $("status-text");

// Prompt display
const rawPromptText       = $("raw-prompt-text");
const optimizedPromptText = $("optimized-prompt-text");

// Score rings (raw)
const rawClarityRing    = $("raw-clarity-ring");
const rawSpecRing       = $("raw-specificity-ring");
const rawSemanticRing   = $("raw-semantic-ring");
const rawTotalRing      = $("raw-total-ring");
const rawClarityVal     = $("raw-clarity-value");
const rawSpecVal        = $("raw-specificity-value");
const rawSemanticVal    = $("raw-semantic-value");
const rawTotalVal       = $("raw-total-value");

// Score rings (optimized)
const optClarityRing    = $("opt-clarity-ring");
const optSpecRing       = $("opt-specificity-ring");
const optSemanticRing   = $("opt-semantic-ring");
const optTotalRing      = $("opt-total-ring");
const optClarityVal     = $("opt-clarity-value");
const optSpecVal        = $("opt-specificity-value");
const optSemanticVal    = $("opt-semantic-value");
const optTotalVal       = $("opt-total-value");

// Improvement badge
const improvementBadge  = $("improvement-badge");
const improvementArrow  = $("improvement-arrow");
const improvementValue  = $("improvement-value");

// LLM responses
const llmResponseRaw    = $("llm-response-raw");
const llmResponseOpt    = $("llm-response-optimized");

// Copy buttons
const copyRawBtn        = $("copy-raw");
const copyOptBtn        = $("copy-optimized");

// ── Character Counter ─────────────────────────────────────────────────
promptInput.addEventListener("input", () => {
  charCount.textContent = `${promptInput.value.length} / 2000`;
});

// ── Error Handling ────────────────────────────────────────────────────
function showError(message) {
  errorText.textContent = message;
  errorBanner.style.display = "flex";
}

function hideError() {
  errorBanner.style.display = "none";
}

errorClose.addEventListener("click", hideError);

// ── Score Ring Animation ──────────────────────────────────────────────
function setRing(ringEl, valueEl, score) {
  const clamped = Math.max(0, Math.min(1, score));
  const offset = RING_CIRCUMFERENCE * (1 - clamped);
  ringEl.style.strokeDashoffset = offset;
  valueEl.textContent = clamped.toFixed(2);

  // Color coding
  let color;
  if (clamped >= 0.7) color = "#10b981";      // green
  else if (clamped >= 0.4) color = "#f59e0b";  // amber
  else color = "#ef4444";                       // red

  if (!ringEl.classList.contains("score-ring__fill--total")) {
    ringEl.style.stroke = color;
  }
}

function resetRings() {
  const rings = document.querySelectorAll(".score-ring__fill");
  const values = document.querySelectorAll(".score-ring__value");
  rings.forEach(r => r.style.strokeDashoffset = RING_CIRCUMFERENCE);
  values.forEach(v => v.textContent = "—");
}

// ── Copy to Clipboard ─────────────────────────────────────────────────
function setupCopy(btn, getTextFn) {
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getTextFn());
      btn.classList.add("copied");
      setTimeout(() => btn.classList.remove("copied"), 1500);
    } catch {
      // Fallback for non-HTTPS
      const ta = document.createElement("textarea");
      ta.value = getTextFn();
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      btn.classList.add("copied");
      setTimeout(() => btn.classList.remove("copied"), 1500);
    }
  });
}

setupCopy(copyRawBtn, () => rawPromptText.textContent);
setupCopy(copyOptBtn, () => optimizedPromptText.textContent);

// ── Loading State ─────────────────────────────────────────────────────
function setLoading(isLoading) {
  if (isLoading) {
    optimizeBtn.classList.add("loading");
    optimizeBtn.disabled = true;
    promptInput.disabled = true;
  } else {
    optimizeBtn.classList.remove("loading");
    optimizeBtn.disabled = false;
    promptInput.disabled = false;
  }
}

// ── API Health Check ──────────────────────────────────────────────────
async function checkApiHealth() {
  try {
    const resp = await fetch(`${API_BASE}/docs`, { method: "HEAD", mode: "no-cors" });
    statusDot.className = "status-dot online";
    statusText.textContent = "API Connected";
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API Offline";
  }
}

// ── Main Optimization Call ────────────────────────────────────────────
async function optimizePrompt() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    showError("Please enter a prompt to optimize.");
    return;
  }

  hideError();
  setLoading(true);
  resetRings();

  try {
    const response = await fetch(`${API_BASE}/optimize_prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError(err.message || "Failed to connect to the optimization API.");
  } finally {
    setLoading(false);
  }
}

// ── Render Results ────────────────────────────────────────────────────
function renderResults(data) {
  resultsDiv.style.display = "block";

  // Scroll into view smoothly
  resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });

  // Prompt texts
  rawPromptText.textContent = data.raw_prompt;
  optimizedPromptText.textContent = data.optimized_prompt;

  // Animate score rings with a slight stagger
  setTimeout(() => {
    setRing(rawClarityRing, rawClarityVal, data.raw_score.clarity);
    setRing(rawSpecRing, rawSpecVal, data.raw_score.specificity);
    setRing(rawSemanticRing, rawSemanticVal, data.raw_score.semantic_preservation);
    setRing(rawTotalRing, rawTotalVal, data.raw_score.total);
  }, 100);

  setTimeout(() => {
    setRing(optClarityRing, optClarityVal, data.optimized_score.clarity);
    setRing(optSpecRing, optSpecVal, data.optimized_score.specificity);
    setRing(optSemanticRing, optSemanticVal, data.optimized_score.semantic_preservation);
    setRing(optTotalRing, optTotalVal, data.optimized_score.total);
  }, 400);

  // Improvement badge
  const imp = data.improvement_score;
  improvementValue.textContent = (imp >= 0 ? "+" : "") + imp.toFixed(4);

  if (imp > 0) {
    improvementArrow.textContent = "↑";
    improvementBadge.className = "improvement-badge positive";
  } else {
    improvementArrow.textContent = "→";
    improvementBadge.className = "improvement-badge neutral";
  }

  // LLM responses
  llmResponseRaw.textContent = data.external_llm_response_raw || "No response available.";
  llmResponseOpt.textContent = data.external_llm_response_optimized || "No response available.";
}

// ── Event Listeners ───────────────────────────────────────────────────
optimizeBtn.addEventListener("click", optimizePrompt);

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    optimizePrompt();
  }
});

// ── Initialize ────────────────────────────────────────────────────────
checkApiHealth();
setInterval(checkApiHealth, 30000);
