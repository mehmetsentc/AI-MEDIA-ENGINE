"use strict";

const PHASES = {
  queued: ["Preparing", "progress"],
  planning: ["Preparing", "progress"],
  waiting_capacity: ["Waiting for available GPU capacity", "waiting"],
  provisioning: ["Starting AI worker", "progress"],
  booting: ["Starting AI worker", "progress"],
  runtime_preparing: ["Starting AI worker", "progress"],
  model_loading: ["Loading image model", "progress"],
  generating: ["Generating image", "progress"],
  saving: ["Saving result", "progress"],
  completed: ["Complete", "done"],
  failed: ["Could not create the image", "error"],
  draft: ["Ready", "idle"],
};

const COST_LABEL = "Actual cost";
const POLL_MS = 4000;

const state = {
  view: "studio",
  dev: false,
  busy: false,
  account: null,
  activeKey: "",
};

function presentJob(job) {
  if (!job) {
    return { phase: "draft", title: PHASES.draft[0], tone: PHASES.draft[1], progress: 0 };
  }
  let status = String(job.status || "");
  const code = job.error_code;
  if (code === "PROVIDER_CAPACITY_UNAVAILABLE" || status === "waiting_capacity") {
    status = "waiting_capacity";
  }
  const pair = PHASES[status] || ["Creating image", "progress"];
  let title = pair[0];
  let tone = pair[1];
  if (status === "failed" && code === "CANCELLED") {
    title = "Cancelled";
    tone = "idle";
  }
  return {
    phase: PHASES[status] ? status : "queued",
    title: title,
    tone: tone,
    progress: job.progress || 0,
  };
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method: method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const content = response.headers.get("Content-Type") || "";
  const payload = content.indexOf("json") >= 0 ? await response.json() : {};
  if (!response.ok) {
    const error = new Error(payload.error || "Request failed");
    error.code = payload.error;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function show(id, visible) {
  document.getElementById(id).hidden = !visible;
}

function renderNav() {
  document.querySelectorAll(".nav-item").forEach(function (button) {
    button.classList.toggle("is-current", button.dataset.view === state.view);
  });
  const signed = Boolean(state.account);
  show("sign-in", !signed);
  show("studio-view", signed && state.view === "studio");
  show("library-view", signed && state.view === "library");
  show("history-view", signed && state.view === "history");
  document.getElementById("sign-out").hidden = !signed;
}

function renderAccount() {
  const pill = document.getElementById("credit-pill");
  const note = document.getElementById("credit-note");
  const generate = document.getElementById("generate");
  const account = state.account;
  if (!account) {
    pill.textContent = "Sign in to see credits";
    return;
  }
  const balance = account.balance == null ? "unavailable" : String(account.balance);
  pill.textContent = account.user_id + " · " + balance + " credits";
  const price = account.image_price;
  if (account.can_generate) {
    note.textContent = "This image costs " + price + " credits. Your balance is " + balance + ".";
    generate.disabled = state.busy;
  } else if (account.block_reason === "INSUFFICIENT_CREDITS") {
    note.textContent = "This image costs " + price + " credits. Your balance is " + balance + ". Generation stays off until the balance covers it.";
    generate.disabled = true;
  } else if (account.block_reason === "AUTHORIZATION_REQUIRED") {
    note.textContent = "A paid GPU authorization is required before this studio can rent a worker. No GPU is rented from this page.";
    generate.disabled = true;
  } else {
    note.textContent = "Credits are unavailable, so generation stays off.";
    generate.disabled = true;
  }
}

function paintStatus(view, extra) {
  const card = document.getElementById("result-card");
  const status = document.getElementById("status");
  const progress = document.getElementById("progress");
  const bar = document.getElementById("bar");
  card.hidden = false;
  status.textContent = extra || view.title;
  status.dataset.tone = view.tone;
  const moving = view.tone === "progress" || view.tone === "waiting";
  progress.hidden = !moving;
  bar.style.width = Math.round((view.progress || 0) * 100) + "%";
}

function showImage(artifactId) {
  const img = document.getElementById("result");
  const download = document.getElementById("download");
  img.hidden = false;
  img.alt = "Generated image";
  img.src = "/v1/artifacts/" + artifactId;
  document.getElementById("dev-flag").hidden = !state.dev;
  download.hidden = false;
  download.href = "/v1/artifacts/" + artifactId + "?download=1";
  download.download = artifactId + ".png";
}

function errorText(err) {
  const payload = err.payload || {};
  if (err.code === "INSUFFICIENT_CREDITS") {
    return "This image costs " + payload.required + " credits. Your balance is " + payload.balance + ".";
  }
  if (err.code === "AUTHORIZATION_REQUIRED") {
    return "Generation is blocked until a paid GPU authorization is bound. No GPU was rented.";
  }
  if (err.code === "PROMPT_REQUIRED") return "Enter a prompt.";
  if (err.code === "DUPLICATE_SUBMISSION") return "That submission is already in progress.";
  if (err.code === "UNKNOWN_CLIENT") return "That identity is not enabled.";
  return "Could not create the image.";
}

async function signIn(event) {
  event.preventDefault();
  const error = document.getElementById("sign-in-error");
  error.hidden = true;
  try {
    state.account = await api("POST", "/v1/session", {
      client_id: document.getElementById("client-id").value.trim(),
    });
    state.view = "studio";
    renderNav();
    renderAccount();
  } catch (err) {
    error.hidden = false;
    error.textContent = err.code === "UNKNOWN_CLIENT" ? "That identity is not enabled." : "Sign-in failed.";
  }
}

async function refreshAccount() {
  state.account = await api("GET", "/v1/account");
  renderAccount();
}

async function generate(event) {
  event.preventDefault();
  if (state.busy || !state.account || !state.account.can_generate) return;
  const prompt = document.getElementById("prompt").value.trim();
  const error = document.getElementById("form-error");
  error.hidden = true;
  if (!prompt) {
    error.hidden = false;
    error.textContent = "Enter a prompt.";
    return;
  }
  const body = {
    prompt: prompt,
    width: 1024,
    height: 1024,
    idempotency_key: state.activeKey || ("studio" + Date.now().toString(16) + Math.random().toString(16).slice(2, 10)),
  };
  const seed = document.getElementById("seed").value.trim();
  if (seed) body.seed = Number(seed);
  state.activeKey = body.idempotency_key;
  state.busy = true;
  renderAccount();
  paintStatus(presentJob({ status: "queued", progress: 0.05 }));
  try {
    const created = await api("POST", "/v1/images/generations", body);
    const job = await poll(created.job_id);
    state.busy = false;
    state.activeKey = "";
    await refreshAccount();
    renderAccount();
    const view = presentJob(job);
    let extra = "";
    if (view.phase === "failed" && job.credit_status === "refunded") {
      extra = "Could not create the image. The unused credit hold was refunded.";
    }
    paintStatus(view, extra);
    if (job.artifact_id) showImage(job.artifact_id);
  } catch (err) {
    state.busy = false;
    state.activeKey = "";
    renderAccount();
    error.hidden = false;
    error.textContent = errorText(err);
    paintStatus({ title: "Could not create the image", tone: "error", progress: 0 });
  }
}

async function poll(jobId) {
  let job = null;
  for (;;) {
    job = await api("GET", "/v1/jobs/" + jobId);
    const view = presentJob(job);
    paintStatus(view);
    if (view.phase === "completed" || view.phase === "failed" || view.phase === "waiting_capacity") return job;
    await new Promise(function (resolve) { setTimeout(resolve, POLL_MS); });
  }
}

function metaLine(item) {
  const parts = [item.width && item.height ? item.width + "×" + item.height : "", item.seed != null ? "seed " + item.seed : "", item.sha256 ? item.sha256.slice(0, 12) : ""];
  return parts.filter(Boolean).join(" · ");
}

async function renderLibrary() {
  const root = document.getElementById("library");
  root.replaceChildren();
  const data = await api("GET", "/v1/library");
  const artifacts = data.artifacts || [];
  if (!artifacts.length) {
    const empty = document.createElement("p");
    empty.textContent = "No images yet.";
    root.appendChild(empty);
    return;
  }
  artifacts.forEach(function (item) {
    const card = document.createElement("article");
    card.className = "media-card";
    const img = document.createElement("img");
    img.src = "/v1/artifacts/" + item.artifact_id;
    img.alt = item.prompt || "Generated image";
    const title = document.createElement("p");
    title.textContent = item.prompt || "Image";
    const meta = document.createElement("p");
    meta.className = "hint";
    meta.textContent = [item.created_at, metaLine(item), item.model || ""].filter(Boolean).join(" · ");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "text-btn";
    open.textContent = "Open";
    open.addEventListener("click", function () { openViewer(item); });
    const download = document.createElement("a");
    download.className = "text-btn";
    download.href = "/v1/artifacts/" + item.artifact_id + "?download=1";
    download.textContent = "Download";
    card.appendChild(img);
    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(open);
    card.appendChild(download);
    root.appendChild(card);
  });
}

function openViewer(item) {
  const dialog = document.getElementById("viewer");
  document.getElementById("viewer-image").src = "/v1/artifacts/" + item.artifact_id;
  document.getElementById("viewer-meta").textContent = (item.prompt || "Image") + " · " + metaLine(item);
  const download = document.getElementById("viewer-download");
  download.href = "/v1/artifacts/" + item.artifact_id + "?download=1";
  dialog.showModal();
}

async function renderHistory() {
  const root = document.getElementById("history");
  root.replaceChildren();
  const data = await api("GET", "/v1/jobs");
  const jobs = data.jobs || [];
  if (!jobs.length) {
    const empty = document.createElement("p");
    empty.textContent = "No jobs yet.";
    root.appendChild(empty);
    return;
  }
  ["pending", "running", "completed", "failed"].forEach(function (group) {
    const matching = jobs.filter(function (job) { return job.group === group; });
    if (!matching.length) return;
    const heading = document.createElement("h2");
    heading.textContent = group.charAt(0).toUpperCase() + group.slice(1);
    root.appendChild(heading);
    matching.forEach(function (job) {
      const row = document.createElement("article");
      row.className = "job-row";
      const view = presentJob(job);
      const title = document.createElement("p");
      title.textContent = view.title + " · " + (job.prompt || "Image");
      const meta = document.createElement("p");
      meta.className = "hint";
      const credit = job.credit_status === "refunded"
        ? "Credits refunded"
        : job.credit_status === "settled"
          ? COST_LABEL + " settled in credits"
          : job.credit_status === "reserved"
            ? "Credits reserved"
            : "";
      meta.textContent = [job.created_at, job.updated_at, job.status, credit, job.error_code || ""].filter(Boolean).join(" · ");
      row.appendChild(title);
      row.appendChild(meta);
      root.appendChild(row);
    });
  });
}

function bind() {
  document.querySelectorAll("[data-view]").forEach(function (button) {
    button.addEventListener("click", function () {
      if (!state.account) return;
      state.view = button.dataset.view;
      renderNav();
      if (state.view === "library") renderLibrary().catch(function () {});
      if (state.view === "history") renderHistory().catch(function () {});
    });
  });
  document.getElementById("sign-in-form").addEventListener("submit", signIn);
  document.getElementById("clone-form").addEventListener("submit", recordConsent);
  document.getElementById("image-form").addEventListener("submit", generate);
  document.getElementById("sign-out").addEventListener("click", function () {
    api("DELETE", "/v1/session").catch(function () {}).then(function () {
      state.account = null;
      state.view = "studio";
      renderNav();
      renderAccount();
    });
  });
  document.getElementById("viewer-close").addEventListener("click", function () {
    document.getElementById("viewer").close();
  });
}

async function recordConsent(event) {
  event.preventDefault();
  const error = document.getElementById("consent-error");
  error.hidden = true;
  try {
    await api("POST", "/v1/voices/consent", {
      speaker_name: document.getElementById("consent-speaker").value.trim(),
      statement: document.getElementById("consent-statement").value.trim(),
      subject_is_owner: true,
      third_party_impersonation: false,
    });
    error.hidden = false;
    error.textContent = "Voice authorization recorded. Speech generation is still unavailable.";
  } catch (err) {
    error.hidden = false;
    error.textContent = err.code === "IMPERSONATION_REFUSED"
      ? "Third-party voice impersonation is refused."
      : "Voice authorization was not recorded.";
  }
}

async function loadCapabilities() {
  const board = document.getElementById("capability-status");
  if (!state.account || !board) return;
  try {
    const catalog = await api("GET", "/v1/capabilities");
    const waiting = (catalog.capabilities || []).filter(function (item) { return !item.executable; });
    board.textContent = waiting.length
      ? "Unavailable until the model is GPU-verified. This screen does not create a result."
      : "Verified capabilities use the image form.";
  } catch (err) {
    board.textContent = "Capability status is unavailable.";
  }
}

async function boot() {
  try {
    const health = await fetch("/health", { credentials: "same-origin" }).then(function (response) { return response.json(); });
    state.dev = health.provider === "fake";
  } catch (err) {
    state.dev = false;
  }
  try {
    state.account = await api("GET", "/v1/account");
  } catch (err) {
    state.account = null;
  }
  renderNav();
  renderAccount();
  if (state.account) loadCapabilities().catch(function () {});
  if (state.dev) document.getElementById("app").dataset.provider = "development";
}

bind();
boot();
