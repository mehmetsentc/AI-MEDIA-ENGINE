const PHASES = {
  queued: ["Creating image", "progress"],
  planning: ["Creating image", "progress"],
  waiting_capacity: ["Waiting for available GPU capacity", "waiting"],
  provisioning: ["Preparing", "progress"],
  booting: ["Preparing", "progress"],
  runtime_preparing: ["Preparing", "progress"],
  model_loading: ["Preparing", "progress"],
  generating: ["Generating", "progress"],
  saving: ["Saving", "progress"],
  completed: ["Complete", "done"],
  failed: ["Could not create the image", "error"],
  draft: ["Ready", "idle"],
};

const TRACKS = ["text", "voice", "music", "image", "video"];
const UPCOMING = { text: "Text", voice: "Voice", music: "Music", video: "Video" };

const state = {
  view: "studio",
  projects: [],
  project: null,
  sceneId: null,
  assetId: null,
  timer: null,
};

function money(value) {
  if (value == null || value === "") return "";
  const raw = String(value);
  if (!/^\d+(\.\d+)?$/.test(raw)) return raw;
  const [whole, frac = ""] = raw.split(".");
  const trimmed = frac.slice(0, 6).replace(/0+$/, "");
  return "$" + whole + (trimmed ? "." + trimmed : "");
}

function presentJob(job) {
  if (!job) return { phase: "draft", title: "Ready", tone: "idle", progress: 0 };
  let status = job.status || "";
  if (job.error_code === "PROVIDER_CAPACITY_UNAVAILABLE" || status === "waiting_capacity") {
    status = "waiting_capacity";
  }
  const known = Object.prototype.hasOwnProperty.call(PHASES, status);
  let title = known ? PHASES[status][0] : "Creating image";
  let tone = known ? PHASES[status][1] : "progress";
  if (status === "failed" && job.error_code === "CANCELLED") {
    title = "Cancelled";
    tone = "idle";
  }
  return { phase: known ? status : "queued", title, tone, progress: job.progress || 0 };
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function scene() {
  if (!state.project) return null;
  return state.project.scenes.find((item) => item.id === state.sceneId) || state.project.scenes[0] || null;
}

function asset() {
  const current = scene();
  if (!current) return null;
  return current.assets.find((item) => item.id === state.assetId)
    || current.assets.find((item) => item.type === "image")
    || null;
}

function setSave(text) {
  document.getElementById("save-state").textContent = text;
}

function showCost(project) {
  const node = document.getElementById("cost");
  const selected = asset();
  const actual = selected && selected.actual_cost;
  const estimated = selected && selected.estimated_cost;
  const shown = money(actual || estimated || (project && project.actual_cost));
  node.textContent = shown ? "Usage " + shown : "Usage";
}

function render() {
  const currentScene = scene();
  const current = asset();
  document.getElementById("project-name").value = state.project ? state.project.name : "Untitled project";
  showCost(state.project);
  document.querySelectorAll(".nav button").forEach((button) => {
    button.classList.toggle("is-current", button.dataset.view === state.view);
  });
  const imageView = state.view === "studio" || state.view === "image" || state.view === "home";
  document.getElementById("quick").hidden = state.view !== "home" && state.view !== "studio" || Boolean(state.project && current && current.artifact_id);
  document.getElementById("preview").hidden = !imageView || !state.project;
  document.getElementById("panel-copy").hidden = imageView || state.view === "projects" || state.view === "assets" || state.view === "history" || state.view === "settings";
  const copy = document.getElementById("panel-copy");
  if (!copy.hidden) {
    const label = UPCOMING[state.view] || "Studio";
    copy.innerHTML = "<h2>" + label + "</h2><p>Coming next. This module shares the same project, scenes, and timeline.</p>";
  }
  if (state.view === "projects") fillList(copy, "Projects", state.projects, (item) => item.name);
  if (state.view === "assets" && currentScene) {
    fillList(copy, "Assets", currentScene.assets, (item) => item.type + " · " + (item.title || "Ready"));
  }
  if (state.view === "history" && current) {
    fillList(copy, "History", current.history || [], (item) => (item.phase || item.status || "job"));
  }
  if (state.view === "settings") {
    copy.hidden = false;
    copy.innerHTML = "<h2>Settings</h2><p>Appearance follows the Light control. Generation uses the same image API as production.</p>";
  }
  const upcoming = !current || current.type !== "image" || (state.view !== "studio" && state.view !== "image" && state.view !== "home");
  document.getElementById("upcoming").hidden = !(state.view in UPCOMING);
  document.getElementById("prop-title").textContent = current && TRACKS.includes(current.type)
    ? current.type[0].toUpperCase() + current.type.slice(1)
    : "Image";
  document.getElementById("image-form").hidden = state.view in UPCOMING;
  if (current && current.type === "image") {
    const prompt = document.getElementById("prompt");
    if (document.activeElement !== prompt) prompt.value = current.prompt || "";
    document.getElementById("seed").value = current.settings && current.settings.seed != null ? current.settings.seed : "";
  }
  const presented = current ? presentJob(current.job || { status: current.phase, error_code: current.error_code, progress: current.progress }) : presentJob(null);
  const status = document.getElementById("status");
  status.textContent = current && current.title ? current.title : presented.title;
  status.dataset.tone = current && current.tone ? current.tone : presented.tone;
  const image = document.getElementById("result");
  const empty = document.getElementById("empty");
  if (current && current.artifact_id && current.phase === "completed") {
    image.hidden = false;
    image.alt = current.prompt || "Generated image";
    image.src = "/v1/artifacts/" + current.artifact_id;
    empty.hidden = true;
    const download = document.getElementById("download");
    download.hidden = false;
    download.href = image.src;
  } else {
    image.hidden = true;
    image.removeAttribute("src");
    empty.hidden = false;
    document.getElementById("download").hidden = true;
  }
  document.getElementById("caption").textContent = current && current.prompt ? current.prompt : "";
  document.getElementById("regenerate").disabled = !(current && current.artifact_id);
  const cancel = document.getElementById("cancel");
  cancel.hidden = !(current && (current.phase === "queued" || current.phase === "planning"));
  const progress = document.getElementById("progress");
  const moving = current && (current.tone === "progress" || current.tone === "waiting");
  progress.hidden = !moving;
  document.getElementById("bar").style.width = Math.round((current && current.progress ? current.progress : 0) * 100) + "%";
  const costs = document.getElementById("costs");
  const hasCost = current && (current.estimated_cost || current.actual_cost);
  costs.hidden = !hasCost;
  document.getElementById("estimated").textContent = current && current.estimated_cost ? money(current.estimated_cost) : "";
  document.getElementById("actual").textContent = current && current.actual_cost ? money(current.actual_cost) : "";
  document.getElementById("estimated").parentElement.hidden = !(current && current.estimated_cost);
  document.getElementById("actual").parentElement.hidden = !(current && current.actual_cost);
  renderTimeline();
}

function fillList(copy, title, items, label) {
  copy.hidden = false;
  const rows = items.map((item) => "<li>" + label(item) + "</li>").join("");
  copy.innerHTML = "<h2>" + title + "</h2><ul>" + (rows || "<li>Nothing here yet.</li>") + "</ul>";
}

function renderTimeline() {
  const scenes = document.getElementById("scenes");
  const tracks = document.getElementById("tracks");
  scenes.replaceChildren();
  tracks.replaceChildren();
  if (!state.project) return;
  state.project.scenes.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.name;
    if (item.id === state.sceneId) button.className = "is-current";
    button.addEventListener("click", () => {
      state.sceneId = item.id;
      state.assetId = null;
      render();
    });
    scenes.appendChild(button);
  });
  const add = document.createElement("button");
  add.type = "button";
  add.textContent = "Add scene";
  add.addEventListener("click", addScene);
  scenes.appendChild(add);
  const current = scene();
  if (!current) return;
  TRACKS.forEach((type) => {
    const item = current.assets.find((asset) => asset.type === type);
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = type[0].toUpperCase() + type.slice(1) + "<small>" + (item && item.title ? item.title : "Empty") + "</small>";
    if (item && item.id === state.assetId) button.className = "is-selected";
    button.addEventListener("click", () => {
      state.assetId = item ? item.id : null;
      state.view = type === "image" ? "image" : type;
      render();
    });
    tracks.appendChild(button);
  });
}

async function ensureProject(brief) {
  if (state.project) return state.project;
  setSave("Saving");
  const name = brief.trim().slice(0, 80) || "Untitled project";
  state.project = await api("POST", "/v1/projects", { name });
  state.sceneId = null;
  setSave("Saved");
  document.getElementById("project-name").value = state.project.name;
  return state.project;
}

async function addScene() {
  if (!state.project) await ensureProject(document.getElementById("brief").value);
  setSave("Saving");
  const created = await api("POST", "/v1/projects/" + state.project.id + "/scenes", {
    name: "Scene " + (state.project.scenes.length + 1),
  });
  state.project = await api("GET", "/v1/projects/" + state.project.id);
  state.sceneId = created.id;
  state.assetId = null;
  setSave("Saved");
  render();
}

async function generateCurrent(regenerate) {
  let current = asset();
  const brief = document.getElementById("brief").value.trim();
  const prompt = (document.getElementById("prompt").value || brief).trim();
  if (!prompt) return;
  if (!state.project) await ensureProject(prompt);
  if (!scene()) await addScene();
  current = scene().assets.find((item) => item.type === "image") || asset();
  if (!current || current.type !== "image") {
    const created = await api("POST", "/v1/scenes/" + scene().id + "/assets", { type: "image", prompt });
    state.assetId = created.id;
    current = created;
  } else {
    state.assetId = current.id;
  }
  setSave("Saving");
  const seedField = document.getElementById("seed").value.trim();
  const body = { prompt, width: 1024, height: 1024 };
  if (!regenerate && seedField) body.seed = Number(seedField);
  await api("PATCH", "/v1/assets/" + current.id, { prompt, ...(seedField && !regenerate ? { seed: Number(seedField) } : {}) });
  const job = await api("POST", "/v1/images/generations", body);
  await api("PATCH", "/v1/assets/" + current.id, { job_id: job.job_id });
  state.project = await api("GET", "/v1/projects/" + state.project.id);
  state.view = "image";
  setSave("Saved");
  render();
  poll(current.id, job.job_id);
}

async function poll(assetId, jobId) {
  clearInterval(state.timer);
  const tick = async () => {
    const job = await api("GET", "/v1/jobs/" + jobId);
    const view = presentJob(job);
    const current = asset();
    if (current && current.id === assetId) {
      current.phase = view.phase;
      current.title = view.title;
      current.tone = view.tone;
      current.progress = view.progress;
      current.error_code = job.error_code;
      if (job.status === "completed") {
        current.artifact_id = job.artifact_id;
        current.phase = "completed";
        current.settings = current.settings || {};
        current.settings.seed = job.seed;
        state.project = await api("GET", "/v1/projects/" + state.project.id);
      }
      render();
    }
    if (view.phase === "completed" || view.phase === "failed" || view.phase === "waiting_capacity") {
      clearInterval(state.timer);
    }
  };
  await tick();
  state.timer = setInterval(tick, 700);
}

async function cancelJob() {
  const current = asset();
  if (!current || !current.job_id) return;
  try {
    await api("POST", "/v1/jobs/" + current.job_id + "/cancel");
  } catch (error) {
    return;
  }
  clearInterval(state.timer);
  current.phase = "failed";
  current.title = "Cancelled";
  current.tone = "idle";
  render();
}

document.getElementById("create-image").addEventListener("click", () => generateCurrent(false));
document.getElementById("generate").addEventListener("click", () => generateCurrent(false));
document.getElementById("form-generate").addEventListener("click", (event) => {
  event.preventDefault();
  generateCurrent(false);
});
document.getElementById("image-form").addEventListener("submit", (event) => {
  event.preventDefault();
  generateCurrent(false);
});
document.getElementById("regenerate").addEventListener("click", () => generateCurrent(true));
document.getElementById("cancel").addEventListener("click", cancelJob);
document.getElementById("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  document.getElementById("theme").textContent = next === "light" ? "Dark" : "Light";
  document.getElementById("theme").setAttribute("aria-pressed", next === "light" ? "true" : "false");
});
document.querySelectorAll(".nav button").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    render();
  });
});
document.getElementById("project-name").addEventListener("change", async (event) => {
  if (!state.project) return;
  setSave("Saving");
  state.project = await api("PATCH", "/v1/projects/" + state.project.id, { name: event.target.value });
  setSave("Saved");
});

if (window.matchMedia("(prefers-color-scheme: light)").matches) {
  document.documentElement.dataset.theme = "light";
  document.getElementById("theme").textContent = "Dark";
}

async function boot() {
  const data = await api("GET", "/v1/projects");
  state.projects = data.projects || [];
  const existing = state.projects.find((project) => (project.scenes || []).some((item) =>
    (item.assets || []).some((asset) => asset.artifact_id)));
  const chosen = existing || state.projects[0];
  if (!chosen) {
    render();
    return;
  }
  state.project = await api("GET", "/v1/projects/" + chosen.id);
  state.sceneId = state.project.scenes[0] ? state.project.scenes[0].id : null;
  const image = state.project.scenes[0] && state.project.scenes[0].assets.find((item) => item.artifact_id);
  state.assetId = image ? image.id : null;
  if (image) state.view = "image";
  render();
}

boot();
