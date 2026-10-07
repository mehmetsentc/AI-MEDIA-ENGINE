"use strict";

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
const TRACK_LABEL = { text: "Text", voice: "Voice", music: "Music", image: "Image", video: "Video" };
const CLIP = {
  text: [8, 28],
  voice: [34, 22],
  music: [4, 90],
  image: [24, 24],
  video: [52, 28],
};

const state = {
  view: "create",
  project: null,
  projects: [],
  sceneId: null,
  assetId: null,
  ratio: localStorage.getItem("studio-ratio") || "16:9",
  musicScope: localStorage.getItem("studio-music") || "scene",
  dev: false,
  busy: false,
  preview: false,
};

const undoStack = [];
const redoStack = [];

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

function shownTitle(view) {
  if (view.phase === "waiting_capacity") return "Waiting for available capacity";
  return view.title;
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method: method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    let detail = {};
    try { detail = await response.json(); } catch (err) { detail = {}; }
    const error = new Error(detail.error || "Request failed");
    error.code = detail.error;
    throw error;
  }
  return response.json();
}

function setSave(text) {
  document.getElementById("save-state").textContent = text;
}

function scene() {
  if (!state.project) return null;
  return state.project.scenes.find(function (item) { return item.id === state.sceneId; }) || state.project.scenes[0] || null;
}

function currentAsset() {
  const current = scene();
  if (!current) return null;
  return current.assets.find(function (item) { return item.id === state.assetId; }) || null;
}

function assetOf(type) {
  const current = scene();
  if (!current) return null;
  return current.assets.find(function (item) { return item.type === type; }) || null;
}

function findAsset(id) {
  if (!state.project) return null;
  for (let i = 0; i < state.project.scenes.length; i += 1) {
    const found = state.project.scenes[i].assets.find(function (item) { return item.id === id; });
    if (found) return found;
  }
  return null;
}

function orderKey() {
  return state.project ? "studio-order-" + state.project.id : "";
}

function applyOrder(project) {
  if (!project || !project.scenes) return;
  let saved = [];
  try { saved = JSON.parse(localStorage.getItem("studio-order-" + project.id) || "[]"); } catch (err) { saved = []; }
  if (!saved.length) return;
  const byId = {};
  project.scenes.forEach(function (item) { byId[item.id] = item; });
  const ordered = saved.map(function (id) { return byId[id]; }).filter(Boolean);
  const rest = project.scenes.filter(function (item) { return saved.indexOf(item.id) < 0; });
  project.scenes = ordered.concat(rest);
  project.scenes.forEach(function (item, index) { item.position = index; });
}

function rememberOrder() {
  if (!state.project) return;
  localStorage.setItem(orderKey(), JSON.stringify(state.project.scenes.map(function (item) { return item.id; })));
}

function reorderScenes(from, to) {
  const scenes = state.project && state.project.scenes;
  if (!scenes || from === to || from < 0 || to < 0 || from >= scenes.length || to >= scenes.length) return;
  const moved = scenes.splice(from, 1)[0];
  scenes.splice(to, 0, moved);
  scenes.forEach(function (item, index) { item.position = index; });
  rememberOrder();
  render();
}

function syncHistoryButtons() {
  document.getElementById("undo").disabled = undoStack.length === 0;
  document.getElementById("redo").disabled = redoStack.length === 0;
}

function rememberPrompt(asset, next) {
  if (!asset || (asset.prompt || "") === next) return;
  undoStack.push({ assetId: asset.id, from: asset.prompt || "", to: next });
  redoStack.length = 0;
  syncHistoryButtons();
}

async function restorePrompt(assetId, prompt) {
  const asset = findAsset(assetId);
  if (!asset) return;
  setSave("Saving");
  const updated = await api("PATCH", "/v1/assets/" + assetId, { prompt: prompt });
  asset.prompt = updated.prompt;
  state.assetId = assetId;
  state.sceneId = asset.scene_id;
  state.view = asset.type === "image" ? "image" : asset.type;
  setSave("Saved");
  syncHistoryButtons();
  render();
}

async function undo() {
  const step = undoStack.pop();
  if (!step) return;
  redoStack.push(step);
  await restorePrompt(step.assetId, step.from);
}

async function redo() {
  const step = redoStack.pop();
  if (!step) return;
  undoStack.push(step);
  await restorePrompt(step.assetId, step.to);
}

function paintStatus(view) {
  const overlay = document.getElementById("overlay");
  const status = document.getElementById("status");
  const progress = document.getElementById("progress");
  const bar = document.getElementById("bar");
  const quiet = view.phase === "draft" || view.phase === "completed";
  overlay.hidden = quiet;
  status.textContent = shownTitle(view);
  status.dataset.tone = view.tone;
  const moving = view.tone === "progress" || view.tone === "waiting";
  progress.hidden = !moving;
  bar.style.width = Math.round((view.progress || 0) * 100) + "%";
}

function renderNav() {
  document.querySelectorAll(".nav-item, .settings-item").forEach(function (button) {
    button.classList.toggle("is-current", button.dataset.view === state.view);
  });
}

function renderStage() {
  const composer = document.getElementById("composer");
  const canvas = document.getElementById("canvas-wrap");
  const library = document.getElementById("library");
  const tool = TRACKS.indexOf(state.view) >= 0;
  composer.hidden = state.view !== "create";
  canvas.hidden = !tool;
  library.hidden = tool || state.view === "create";
  if (!library.hidden) renderLibrary();
  if (tool) renderCanvas();
}

function renderCanvas() {
  const frame = document.getElementById("frame");
  frame.dataset.ratio = state.ratio;
  const asset = currentAsset();
  const empty = document.getElementById("empty");
  const heading = empty.querySelector("h2");
  const note = document.getElementById("empty-prompt");
  const generate = document.getElementById("empty-generate");
  const img = document.getElementById("result");
  const swatch = document.getElementById("swatch");
  const flag = document.getElementById("dev-flag");
  img.hidden = true;
  swatch.hidden = true;
  flag.hidden = true;
  empty.hidden = true;
  if (asset && asset.type === "image" && asset.artifact_id && state.dev) {
    empty.hidden = false;
    swatch.hidden = false;
    swatch.src = "/v1/artifacts/" + asset.artifact_id;
    heading.textContent = "Development preview";
    note.textContent = "This file is local development data, not a finished image.";
    generate.hidden = true;
  } else if (asset && asset.type === "image" && asset.artifact_id) {
    img.hidden = false;
    img.src = "/v1/artifacts/" + asset.artifact_id;
    img.alt = "Scene image";
  } else if (asset && asset.type === "image") {
    empty.hidden = false;
    heading.textContent = "Create your first image";
    note.textContent = asset.prompt || "Add a prompt, then generate this scene.";
    generate.hidden = false;
  } else if (asset) {
    empty.hidden = false;
    heading.textContent = TRACK_LABEL[asset.type] || "Scene";
    note.textContent = asset.prompt || "This part of the scene is still open.";
    generate.hidden = true;
  } else {
    empty.hidden = false;
    heading.textContent = "Open a scene";
    note.textContent = "Create a project to start the first scene.";
    generate.hidden = true;
  }
  if (!state.busy) document.getElementById("overlay").hidden = true;
  const image = asset && asset.type === "image" ? asset : assetOf("image");
  const exp = document.getElementById("export");
  exp.disabled = !(image && image.artifact_id);
}

function renderScenes() {
  const root = document.getElementById("scenes");
  root.replaceChildren();
  const scenes = state.project ? state.project.scenes : [];
  scenes.forEach(function (item, index) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-card" + (item.id === state.sceneId ? " is-current" : "");
    button.draggable = true;
    button.dataset.index = String(index);
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    const image = item.assets.find(function (asset) { return asset.type === "image" && asset.artifact_id; });
    if (image) {
      const pic = document.createElement("img");
      pic.src = "/v1/artifacts/" + image.artifact_id;
      pic.alt = "";
      if (state.dev) thumb.classList.add("is-dev");
      thumb.appendChild(pic);
    } else {
      thumb.textContent = String(index + 1);
    }
    const label = document.createElement("span");
    label.textContent = item.name || ("Scene " + (index + 1));
    button.appendChild(thumb);
    button.appendChild(label);
    button.addEventListener("click", function () { selectScene(item.id); });
    button.addEventListener("dragstart", function (event) {
      event.dataTransfer.setData("text/plain", String(index));
    });
    button.addEventListener("dragover", function (event) { event.preventDefault(); });
    button.addEventListener("drop", function (event) {
      event.preventDefault();
      reorderScenes(Number(event.dataTransfer.getData("text/plain")), index);
    });
    root.appendChild(button);
  });
  const add = document.createElement("button");
  add.type = "button";
  add.className = "add-scene";
  add.id = "add-scene";
  const plus = document.createElement("div");
  plus.className = "thumb";
  plus.textContent = "+";
  const caption = document.createElement("span");
  caption.textContent = "Add";
  add.setAttribute("aria-label", "Add scene");
  add.appendChild(plus);
  add.appendChild(caption);
  add.addEventListener("click", addScene);
  root.appendChild(add);
}

function clipSpan(type) {
  if (type === "music" && state.musicScope === "scene") return [18, 36];
  return CLIP[type];
}

function renderTracks() {
  const root = document.getElementById("tracks");
  root.replaceChildren();
  const current = scene();
  TRACKS.forEach(function (type) {
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.dataset.track = type;
    const label = document.createElement("div");
    label.className = "lane-label";
    label.textContent = TRACK_LABEL[type];
    const body = document.createElement("div");
    body.className = "lane-body";
    const asset = current && current.assets.find(function (item) { return item.type === type; });
    if (asset) {
      const span = clipSpan(type);
      const clip = document.createElement("button");
      clip.type = "button";
      clip.className = "clip" + (asset.id === state.assetId ? " is-selected" : "");
      clip.style.left = span[0] + "%";
      clip.style.width = span[1] + "%";
      const name = (asset.prompt || TRACK_LABEL[type]).trim();
      clip.textContent = name.length > 28 ? name.slice(0, 28) + "…" : name;
      clip.addEventListener("click", function () { selectAsset(asset); });
      body.appendChild(clip);
    }
    lane.appendChild(label);
    lane.appendChild(body);
    root.appendChild(lane);
  });
}

function renderLibrary() {
  const root = document.getElementById("library");
  root.replaceChildren();
  const heading = document.createElement("h2");
  if (state.view === "projects") heading.textContent = "Projects";
  if (state.view === "assets") heading.textContent = "Assets";
  if (state.view === "history") heading.textContent = "History";
  if (state.view === "settings") heading.textContent = "Settings";
  root.appendChild(heading);
  if (state.view === "settings") {
    const note = document.createElement("p");
    note.textContent = state.dev
      ? "Development data is labeled on the canvas. It is not a production image."
      : "Appearance and project files stay on this machine.";
    root.appendChild(note);
    return;
  }
  if (state.view === "projects") {
    state.projects.forEach(function (item) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = item.name || "Untitled project";
      button.addEventListener("click", function () { openProject(item.id); });
      root.appendChild(button);
    });
    return;
  }
  const current = state.project;
  if (!current) return;
  if (state.view === "assets") {
    current.scenes.forEach(function (item) {
      item.assets.forEach(function (asset) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = (item.name || "Scene") + " · " + TRACK_LABEL[asset.type];
        button.addEventListener("click", function () {
          state.sceneId = item.id;
          selectAsset(asset);
        });
        root.appendChild(button);
      });
    });
    return;
  }
  current.scenes.forEach(function (item) {
    const image = item.assets.find(function (asset) { return asset.type === "image"; });
    (image && image.history || []).forEach(function (entry) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = (item.name || "Scene") + " · " + (entry.phase || entry.status || "image");
      button.addEventListener("click", function () {
        state.sceneId = item.id;
        selectAsset(image);
      });
      root.appendChild(button);
    });
  });
}

function renderInspector() {
  const title = document.getElementById("prop-title");
  const upcoming = document.getElementById("upcoming");
  const imageForm = document.getElementById("image-form");
  const other = document.getElementById("other-editor");
  const asset = currentAsset();
  const imageTool = state.view === "image" && asset && asset.type === "image";
  document.querySelectorAll(".ratios button[data-ratio]").forEach(function (button) {
    button.classList.toggle("is-selected", button.dataset.ratio === state.ratio);
  });
  if (!imageTool) {
    imageForm.hidden = true;
    const editable = asset && TRACKS.indexOf(asset.type) >= 0 && asset.type !== "image" && state.view === asset.type;
    other.hidden = !editable;
    upcoming.hidden = false;
    if (state.view === "create") {
      title.textContent = "Create";
      upcoming.textContent = "Write a brief, then open Images to generate the scene.";
    } else if (editable) {
      title.textContent = TRACK_LABEL[asset.type];
      upcoming.textContent = "Coming next. This asset stays in the scene and can be edited on its own.";
      const field = document.getElementById("other-prompt");
      if (document.activeElement !== field) field.value = asset.prompt || "";
      document.getElementById("music-scope").hidden = asset.type !== "music";
      document.querySelectorAll("#music-scope button").forEach(function (button) {
        button.classList.toggle("is-selected", button.dataset.scope === state.musicScope);
      });
    } else {
      title.textContent = TRACK_LABEL[state.view] || "Studio";
      upcoming.textContent = "Choose a scene to edit this part of the project.";
    }
    return;
  }
  title.textContent = "Image";
  upcoming.hidden = true;
  imageForm.hidden = false;
  other.hidden = true;
  const prompt = document.getElementById("prompt");
  const seed = document.getElementById("seed");
  if (document.activeElement !== prompt) prompt.value = asset.prompt || "";
  if (document.activeElement !== seed) {
    seed.value = asset.settings && asset.settings.seed != null ? String(asset.settings.seed) : "";
  }
  const hasImage = Boolean(asset.artifact_id);
  document.getElementById("form-generate").hidden = hasImage;
  document.getElementById("regenerate").hidden = !hasImage;
  const download = document.getElementById("download");
  download.hidden = !hasImage;
  if (hasImage) download.href = "/v1/artifacts/" + asset.artifact_id;
  const costs = document.getElementById("costs");
  const estimate = asset.estimated_cost;
  const realEstimate = !state.dev && estimate != null && estimate !== "";
  costs.hidden = !realEstimate;
  if (realEstimate) document.getElementById("estimated").textContent = String(estimate);
  document.getElementById("cancel").hidden = true;
}

function render() {
  const name = document.getElementById("project-name");
  if (document.activeElement !== name) name.value = state.project ? state.project.name : "Untitled project";
  document.getElementById("app").classList.toggle("is-preview", state.preview);
  document.getElementById("mode").textContent = state.dev ? "Development data" : "Studio";
  renderNav();
  renderStage();
  renderScenes();
  renderTracks();
  renderInspector();
  syncHistoryButtons();
}

function selectScene(id) {
  state.sceneId = id;
  const image = assetOf("image");
  state.assetId = image ? image.id : null;
  state.view = "image";
  render();
}

function selectAsset(asset) {
  state.assetId = asset.id;
  state.sceneId = asset.scene_id;
  state.view = asset.type === "image" ? "image" : asset.type;
  render();
}

async function openProject(id) {
  setSave("Saving");
  state.project = await api("GET", "/v1/projects/" + id);
  applyOrder(state.project);
  const current = state.project.scenes[0];
  state.sceneId = current ? current.id : null;
  const image = current && current.assets.find(function (asset) { return asset.type === "image"; });
  state.assetId = image ? image.id : null;
  state.view = image ? "image" : "create";
  setSave("Saved");
  render();
}

async function createProject() {
  const brief = document.getElementById("brief").value.trim();
  if (!brief || state.busy) return;
  state.busy = true;
  setSave("Saving");
  try {
    const project = await api("POST", "/v1/projects", { name: brief.slice(0, 80) });
    const created = await api("POST", "/v1/projects/" + project.id + "/scenes", { name: "Scene 1" });
    const image = created.assets.find(function (asset) { return asset.type === "image"; });
    if (image) await api("PATCH", "/v1/assets/" + image.id, { prompt: brief });
    state.project = await api("GET", "/v1/projects/" + project.id);
    state.projects = [state.project].concat(state.projects.filter(function (item) { return item.id !== project.id; }));
    state.sceneId = created.id;
    state.assetId = image ? image.id : null;
    state.view = "image";
    setSave("Saved");
    render();
  } catch (err) {
    setSave("Not saved");
  } finally {
    state.busy = false;
  }
}

async function addScene() {
  if (!state.project || state.busy) {
    state.view = "create";
    render();
    return;
  }
  setSave("Saving");
  const index = state.project.scenes.length + 1;
  const created = await api("POST", "/v1/projects/" + state.project.id + "/scenes", { name: "Scene " + index });
  state.project = await api("GET", "/v1/projects/" + state.project.id);
  applyOrder(state.project);
  if (!state.project.scenes.some(function (item) { return item.id === created.id; })) {
    state.project.scenes.push(created);
  }
  state.sceneId = created.id;
  const image = created.assets.find(function (asset) { return asset.type === "image"; });
  state.assetId = image ? image.id : null;
  state.view = "image";
  rememberOrder();
  setSave("Saved");
  render();
}

async function saveName() {
  if (!state.project) return;
  const next = document.getElementById("project-name").value.trim();
  if (!next || next === state.project.name) return;
  setSave("Saving");
  state.project = await api("PATCH", "/v1/projects/" + state.project.id, { name: next });
  applyOrder(state.project);
  setSave("Saved");
  render();
}

async function saveAssetPrompt(asset, next) {
  if (!asset) return;
  rememberPrompt(asset, next);
  if ((asset.prompt || "") === next) return;
  setSave("Saving");
  const updated = await api("PATCH", "/v1/assets/" + asset.id, { prompt: next });
  asset.prompt = updated.prompt;
  setSave("Saved");
  renderTracks();
}

async function generateImage(regenerate) {
  const asset = assetOf("image");
  if (!asset || state.busy) return;
  const prompt = document.getElementById("prompt").value.trim() || asset.prompt || "";
  if (!prompt) return;
  const seedField = document.getElementById("seed").value.trim();
  const body = { prompt: prompt, width: 1024, height: 1024 };
  if (!regenerate && seedField) body.seed = Number(seedField);
  state.busy = true;
  state.view = "image";
  state.assetId = asset.id;
  setSave("Saving");
  render();
  paintStatus({ phase: "queued", title: "Creating image", tone: "progress", progress: 0.05 });
  try {
    const patch = { prompt: prompt };
    if (!regenerate && seedField) patch.seed = Number(seedField);
    await api("PATCH", "/v1/assets/" + asset.id, patch);
    const job = await api("POST", "/v1/images/generations", body);
    state.activeJobId = job.job_id;
    await api("PATCH", "/v1/assets/" + asset.id, { job_id: job.job_id });
    const view = await poll(job.job_id);
    state.busy = false;
    setSave("Saved");
    render();
    if (view && view.phase !== "completed") paintStatus(view);
  } catch (err) {
    state.busy = false;
    paintStatus({ phase: "failed", title: "Could not create the image", tone: "error", progress: 0 });
    setSave("Not saved");
  }
}

async function poll(jobId) {
  let view = null;
  for (;;) {
    const job = await api("GET", "/v1/jobs/" + jobId);
    view = presentJob(job);
    paintStatus(view);
    const cancel = document.getElementById("cancel");
    const early = job.status === "queued" || job.status === "planning";
    cancel.hidden = !early;
    if (view.phase === "completed" || view.phase === "failed" || view.phase === "waiting_capacity") break;
    await new Promise(function (resolve) { setTimeout(resolve, 700); });
  }
  state.project = await api("GET", "/v1/projects/" + state.project.id);
  applyOrder(state.project);
  return view;
}

function bind() {
  document.querySelectorAll("[data-view]").forEach(function (button) {
    button.addEventListener("click", function () {
      const next = button.dataset.view;
      if (TRACKS.indexOf(next) >= 0) {
        const asset = assetOf(next);
        if (!asset) {
          state.view = "create";
          render();
          return;
        }
        selectAsset(asset);
        return;
      }
      state.view = next;
      render();
    });
  });
  document.getElementById("create-project").addEventListener("click", createProject);
  document.getElementById("examples").addEventListener("click", function (event) {
    const example = event.target.closest("[data-example]");
    if (!example) return;
    document.getElementById("brief").value = example.dataset.example;
  });
  document.getElementById("empty-generate").addEventListener("click", function () { generateImage(false); });
  document.getElementById("image-form").addEventListener("submit", function (event) {
    event.preventDefault();
    generateImage(false);
  });
  document.getElementById("regenerate").addEventListener("click", function () { generateImage(true); });
  document.getElementById("prompt").addEventListener("change", function (event) {
    saveAssetPrompt(assetOf("image"), event.target.value.trim());
  });
  document.getElementById("other-prompt").addEventListener("change", function (event) {
    const asset = currentAsset();
    if (!asset || asset.type === "image") return;
    saveAssetPrompt(asset, event.target.value.trim());
  });
  document.getElementById("project-name").addEventListener("change", saveName);
  document.querySelectorAll(".ratios button[data-ratio]").forEach(function (button) {
    button.addEventListener("click", function () {
      state.ratio = button.dataset.ratio;
      localStorage.setItem("studio-ratio", state.ratio);
      render();
    });
  });
  document.querySelectorAll("#music-scope button").forEach(function (button) {
    button.addEventListener("click", function () {
      state.musicScope = button.dataset.scope;
      localStorage.setItem("studio-music", state.musicScope);
      renderTracks();
      renderInspector();
    });
  });
  document.getElementById("undo").addEventListener("click", function () { undo().catch(function () { setSave("Not saved"); }); });
  document.getElementById("redo").addEventListener("click", function () { redo().catch(function () { setSave("Not saved"); }); });
  document.getElementById("preview").addEventListener("click", function () {
    state.preview = !state.preview;
    render();
  });
  document.getElementById("export").addEventListener("click", function () {
    const image = assetOf("image");
    if (!image || !image.artifact_id) return;
    const link = document.createElement("a");
    link.href = "/v1/artifacts/" + image.artifact_id;
    link.download = "image.png";
    link.click();
  });
  document.getElementById("inspector-toggle").addEventListener("click", function () {
    document.getElementById("inspector").classList.toggle("is-open");
  });
  document.getElementById("timeline-toggle").addEventListener("click", function () {
    const timeline = document.getElementById("timeline");
    const open = timeline.classList.toggle("is-collapsed");
    document.getElementById("timeline-toggle").setAttribute("aria-expanded", open ? "false" : "true");
  });
  document.getElementById("account").addEventListener("click", function () {
    const menu = document.getElementById("account-menu");
    menu.hidden = !menu.hidden;
    document.getElementById("account").setAttribute("aria-expanded", menu.hidden ? "false" : "true");
  });
  document.getElementById("theme").addEventListener("click", function () {
    const light = document.documentElement.dataset.theme !== "light";
    document.documentElement.dataset.theme = light ? "light" : "";
    localStorage.setItem("studio-theme", light ? "light" : "dark");
    document.getElementById("theme").textContent = light ? "Use dark appearance" : "Use light appearance";
  });
  document.getElementById("cancel").addEventListener("click", function () {
    if (!state.activeJobId) return;
    api("POST", "/v1/jobs/" + state.activeJobId + "/cancel").catch(function () {});
  });
}

async function boot() {
  try {
    const health = await fetch("/health", { credentials: "same-origin" }).then(function (response) { return response.json(); });
    state.dev = health.provider === "fake";
  } catch (err) {
    state.dev = false;
  }
  if (localStorage.getItem("studio-theme") === "light") {
    document.documentElement.dataset.theme = "light";
    document.getElementById("theme").textContent = "Use dark appearance";
  }
  const data = await api("GET", "/v1/projects");
  state.projects = data.projects || [];
  const existing = state.projects.find(function (project) {
    return (project.scenes || []).some(function (item) {
      return (item.assets || []).some(function (asset) { return asset.artifact_id; });
    });
  });
  const chosen = existing || state.projects[0];
  if (chosen) {
    state.project = chosen.scenes ? chosen : await api("GET", "/v1/projects/" + chosen.id);
    applyOrder(state.project);
    const current = state.project.scenes[0];
    state.sceneId = current ? current.id : null;
    const image = current && current.assets.find(function (asset) { return asset.type === "image"; });
    state.assetId = image ? image.id : null;
    state.view = image && image.artifact_id ? "image" : "create";
  }
  render();
}

bind();
boot().catch(function () { setSave("Not saved"); });
