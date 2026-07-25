const state = {
  reportsPayload: null,
  runsById: new Map(),
  reportById: new Map(),
  activeRunId: null,
  activeTab: "text",
  reportCache: new Map(),
  activeReportData: null,
  allChanges: [],
  filteredChanges: [],
  selectedIndex: null,
  activeFilter: "all",
  visibleCount: 200,
  baselinePage: null,
  revisionPage: null,
  imageView: {
    baseline: {
      scale: 1,
      translateX: 0,
      translateY: 0,
      mode: "zoom",
      dragging: false,
      startX: 0,
      startY: 0,
      initialTranslateX: 0,
      initialTranslateY: 0,
    },
    revision: {
      scale: 1,
      translateX: 0,
      translateY: 0,
      mode: "zoom",
      dragging: false,
      startX: 0,
      startY: 0,
      initialTranslateX: 0,
      initialTranslateY: 0,
    },
  },
};

const elements = {
  runSelect: document.getElementById("run-select"),
  docIds: document.getElementById("doc-ids"),
  status: document.getElementById("status"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  filters: document.getElementById("filters"),
  summary: {
    added: document.getElementById("summary-added"),
    removed: document.getElementById("summary-removed"),
    modified: document.getElementById("summary-modified"),
    unchanged: document.getElementById("summary-unchanged"),
  },
  baselineImage: document.getElementById("baseline-image"),
  baselineOverlay: document.getElementById("baseline-overlay"),
  baselineFrame: document.getElementById("baseline-frame"),
  baselineStage: document.getElementById("baseline-frame")?.closest(".image-stage"),
  revisionImage: document.getElementById("revision-image"),
  revisionOverlay: document.getElementById("revision-overlay"),
  revisionFrame: document.getElementById("revision-frame"),
  revisionStage: document.getElementById("revision-frame")?.closest(".image-stage"),
  changeList: document.getElementById("change-list"),
  changeDetails: document.getElementById("change-details"),
  listMeta: document.getElementById("list-meta"),
  loadMore: document.getElementById("load-more"),
};

const VIEW_NAMES = ["baseline", "revision"];
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;

function getViewElements(viewName) {
  if (viewName === "baseline") {
    return {
      frame: elements.baselineFrame,
      stage: elements.baselineStage,
      image: elements.baselineImage,
    };
  }
  return {
    frame: elements.revisionFrame,
    stage: elements.revisionStage,
    image: elements.revisionImage,
  };
}

function getViewState(viewName) {
  return state.imageView[viewName];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function applyViewTransform(viewName) {
  const { frame } = getViewElements(viewName);
  const view = getViewState(viewName);
  if (!frame) {
    return;
  }
  frame.style.transform = `translate(${view.translateX}px, ${view.translateY}px) scale(${view.scale})`;
}

function resetViewTransform(viewName) {
  const view = getViewState(viewName);
  view.scale = 1;
  view.translateX = 0;
  view.translateY = 0;
  view.dragging = false;
  applyViewTransform(viewName);
}

function resetAllViewTransforms() {
  for (const viewName of VIEW_NAMES) {
    resetViewTransform(viewName);
  }
}

function setViewMode(viewName, mode) {
  const view = getViewState(viewName);
  const { stage } = getViewElements(viewName);
  view.mode = mode;

  document
    .querySelectorAll(`.image-tool-button[data-view="${viewName}"]`)
    .forEach((button) => {
      if (button.dataset.mode === "reset") {
        return;
      }
      button.classList.toggle("active", button.dataset.mode === mode);
    });

  if (stage) {
    stage.classList.remove("mode-zoom", "mode-pan");
    stage.classList.add(mode === "pan" ? "mode-pan" : "mode-zoom");
  }
}

function redrawSelectedChangeIfNeeded() {
  if (state.selectedIndex === null) {
    return;
  }
  drawSelectedChange(state.filteredChanges[state.selectedIndex]);
}

function zoomViewAtPointer(viewName, clientX, clientY, zoomFactor) {
  const { stage } = getViewElements(viewName);
  const view = getViewState(viewName);
  if (!stage) {
    return;
  }

  const stageRect = stage.getBoundingClientRect();
  const pointerX = clientX - stageRect.left;
  const pointerY = clientY - stageRect.top;
  const imageX = (pointerX - view.translateX) / view.scale;
  const imageY = (pointerY - view.translateY) / view.scale;
  const nextScale = clamp(view.scale * zoomFactor, MIN_ZOOM, MAX_ZOOM);

  view.translateX = pointerX - imageX * nextScale;
  view.translateY = pointerY - imageY * nextScale;
  view.scale = nextScale;
  applyViewTransform(viewName);
}

function setupImageViewport(viewName) {
  const { stage } = getViewElements(viewName);
  const view = getViewState(viewName);
  if (!stage) {
    return;
  }

  setViewMode(viewName, view.mode);
  applyViewTransform(viewName);

  document
    .querySelectorAll(`.image-tool-button[data-view="${viewName}"]`)
    .forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.dataset.mode;
        if (mode === "reset") {
          resetViewTransform(viewName);
          redrawSelectedChangeIfNeeded();
          return;
        }
        setViewMode(viewName, mode);
      });
    });

  stage.addEventListener(
    "wheel",
    (event) => {
      if (view.mode !== "zoom") {
        return;
      }
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.12 : 0.89;
      zoomViewAtPointer(viewName, event.clientX, event.clientY, zoomFactor);
    },
    { passive: false }
  );

  stage.addEventListener("mousedown", (event) => {
    if (view.mode !== "pan" || event.button !== 0) {
      return;
    }
    event.preventDefault();
    view.dragging = true;
    view.startX = event.clientX;
    view.startY = event.clientY;
    view.initialTranslateX = view.translateX;
    view.initialTranslateY = view.translateY;
    stage.classList.add("is-dragging");
  });

  window.addEventListener("mousemove", (event) => {
    if (!view.dragging || view.mode !== "pan") {
      return;
    }
    view.translateX = view.initialTranslateX + (event.clientX - view.startX);
    view.translateY = view.initialTranslateY + (event.clientY - view.startY);
    applyViewTransform(viewName);
  });

  window.addEventListener("mouseup", () => {
    if (!view.dragging) {
      return;
    }
    view.dragging = false;
    stage.classList.remove("is-dragging");
  });
}

function setupImageViewports() {
  for (const viewName of VIEW_NAMES) {
    setupImageViewport(viewName);
  }
}

function setStatus(message) {
  elements.status.textContent = message;
}

function reportKindForTab(tab) {
  const kindByTab = {
    text: "text",
    geometry: "geometry",
    vlm: "vlm",
    "vlm-comparison": "vlm_comparison",
  };
  return kindByTab[tab] || tab;
}

function reportIdForKind(run, tab) {
  return run?.reports?.[reportKindForTab(tab)] || null;
}

function isNormalizedBox(box) {
  if (!box) return false;
  const values = [box.x0, box.y0, box.x1, box.y1];
  return values.every((value) => typeof value === "number" && value >= 0 && value <= 1.01);
}

function normalizeBbox(raw) {
  if (!raw) {
    return null;
  }
  if (Array.isArray(raw) && raw.length >= 4) {
    const [x0, y0, x1, y1] = raw;
    if (![x0, y0, x1, y1].every((value) => Number.isFinite(value))) {
      return null;
    }
    return { x0, y0, x1, y1 };
  }
  if (typeof raw === "object") {
    const { x0, y0, x1, y1 } = raw;
    if (![x0, y0, x1, y1].every((value) => Number.isFinite(value))) {
      return null;
    }
    return { x0, y0, x1, y1 };
  }
  return null;
}

function metricNumber(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "-";
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function imageApiPath(source) {
  if (!source || typeof source !== "string") return null;
  if (source.startsWith("/api/images/")) return source;

  let relative = source.trim();
  const artifactToken = "/artifacts/";
  if (relative.includes(artifactToken)) {
    relative = relative.split(artifactToken)[1];
  }
  relative = relative.replace(/^\/+/, "");

  const encoded = relative
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");

  return encoded ? `/api/images/${encoded}` : null;
}

function clearOverlay(canvas) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function syncOverlayCanvas(canvas, imageElement) {
  if (!imageElement.naturalWidth || !imageElement.naturalHeight) {
    return false;
  }
  canvas.width = imageElement.clientWidth;
  canvas.height = imageElement.clientHeight;
  return canvas.width > 0 && canvas.height > 0;
}

function pageBoxToDisplayRect(box, pageSize, displayWidth, displayHeight) {
  const scaleX = displayWidth / pageSize.width;
  const scaleY = displayHeight / pageSize.height;

  let x0 = box.x0 * scaleX;
  let y0 = box.y0 * scaleY;
  let x1 = box.x1 * scaleX;
  let y1 = box.y1 * scaleY;

  let left = Math.min(x0, x1);
  let top = Math.min(y0, y1);
  let width = Math.abs(x1 - x0);
  let height = Math.abs(y1 - y0);

  const minSize = 2;
  if (width < minSize) {
    const center = left + width / 2;
    left = center - minSize / 2;
    width = minSize;
  }
  if (height < minSize) {
    const center = top + height / 2;
    top = center - minSize / 2;
    height = minSize;
  }

  return { left, top, width, height };
}

function imageBoxToDisplayRect(box, imageElement) {
  const scaleX = imageElement.clientWidth / imageElement.naturalWidth;
  const scaleY = imageElement.clientHeight / imageElement.naturalHeight;

  let left = box.x0 * scaleX;
  let top = box.y0 * scaleY;
  let width = (box.x1 - box.x0) * scaleX;
  let height = (box.y1 - box.y0) * scaleY;

  const minSize = 2;
  if (Math.abs(width) < minSize) {
    left -= (minSize - Math.abs(width)) / 2;
    width = minSize;
  }
  if (Math.abs(height) < minSize) {
    top -= (minSize - Math.abs(height)) / 2;
    height = minSize;
  }

  return { left, top, width, height };
}

function normalizedBoxToDisplayRect(box, pageSize, displayWidth, displayHeight) {
  return pageBoxToDisplayRect(
    {
      x0: box.x0 * pageSize.width,
      y0: box.y0 * pageSize.height,
      x1: box.x1 * pageSize.width,
      y1: box.y1 * pageSize.height,
    },
    pageSize,
    displayWidth,
    displayHeight
  );
}

function drawRect(context, rect, color) {
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.strokeRect(rect.left, rect.top, rect.width, rect.height);
  context.fillStyle = `${color}22`;
  context.fillRect(rect.left, rect.top, rect.width, rect.height);
}

function drawPageBox(canvas, imageElement, box, color, pageSize) {
  if (!box || !pageSize || !syncOverlayCanvas(canvas, imageElement)) {
    return;
  }

  const context = canvas.getContext("2d");
  const rect = pageBoxToDisplayRect(
    box,
    pageSize,
    imageElement.clientWidth,
    imageElement.clientHeight
  );
  drawRect(context, rect, color);
}

function drawImageSpaceBox(canvas, imageElement, box, color) {
  if (!box || !syncOverlayCanvas(canvas, imageElement)) {
    return;
  }

  const context = canvas.getContext("2d");
  const rect = imageBoxToDisplayRect(box, imageElement);
  drawRect(context, rect, color);
}

function drawNormalizedBox(canvas, imageElement, box, color, pageSize) {
  if (!box || !pageSize || !syncOverlayCanvas(canvas, imageElement)) {
    return;
  }

  const context = canvas.getContext("2d");
  const rect = normalizedBoxToDisplayRect(
    box,
    pageSize,
    imageElement.clientWidth,
    imageElement.clientHeight
  );
  drawRect(context, rect, color);
}

function getActiveRun() {
  return state.runsById.get(state.activeRunId) || null;
}

function normalizeTextOrGeometry(data) {
  const summary = data?.summary || {};
  const changes = Array.isArray(data?.changes) ? data.changes : [];
  const baselinePage = data?.revision_a
    ? { width: data.revision_a.page_width, height: data.revision_a.page_height }
    : null;
  const revisionPage = data?.revision_b
    ? { width: data.revision_b.page_width, height: data.revision_b.page_height }
    : null;

  return {
    summary,
    changes,
    baselinePage,
    revisionPage,
    baselineImage: null,
    revisionImage: null,
  };
}

function normalizeVlm(data, run) {
  const summary = data?.vlm_analysis?.summary || {
    has_meaningful_changes: false,
    number_of_meaningful_changes: 0,
    overall_summary: "No VLM summary available",
  };

  let changes = Array.isArray(data?.vlm_analysis?.changes) ? data.vlm_analysis.changes : [];
  if (changes.length === 0 && Array.isArray(data?.change_regions)) {
    changes = data.change_regions.map((region) => ({
      change_type: "unknown",
      description: "Detected visual difference region",
      location_description: region.id,
      confidence: null,
      baseline_bbox: null,
      revision_bbox: region.bbox || null,
      _bbox_space: "image",
    }));
  }

  return {
    summary: {
      added: changes.filter((item) => item.change_type === "added").length,
      removed: changes.filter((item) => item.change_type === "removed").length,
      modified:
        changes.filter((item) => item.change_type === "modified").length ||
        (summary.number_of_meaningful_changes ?? changes.length),
      unchanged: 0,
      overall_summary: summary.overall_summary,
      number_of_meaningful_changes: summary.number_of_meaningful_changes ?? changes.length,
    },
    changes,
    baselinePage: null,
    revisionPage: null,
    baselineImage: run ? `/api/images/${encodeURIComponent(run.baseline_document_id)}/original_render.png` : null,
    revisionImage: run ? `/api/images/${encodeURIComponent(run.revision_document_id)}/original_render.png` : null,
  };
}

function normalizeVlmComparison(data, run) {
  const summary = data?.summary || {};
  const changes = (Array.isArray(data?.changes) ? data.changes : []).map((change) => ({
    ...change,
    baseline_bbox: normalizeBbox(change.baseline_bbox),
    revision_bbox: normalizeBbox(change.revision_bbox),
    _bbox_space: "normalized",
  }));

  return {
    summary: {
      added: changes.filter((item) => item.change_type === "added").length,
      removed: changes.filter((item) => item.change_type === "removed").length,
      modified: changes.filter((item) => item.change_type === "modified").length,
      unchanged: changes.filter((item) => item.change_type === "unchanged").length,
      overall_summary: summary.overall_summary,
      number_of_changes: summary.number_of_changes ?? changes.length,
    },
    changes,
    baselinePage: null,
    revisionPage: null,
    baselineImage: run ? `/api/images/${encodeURIComponent(run.baseline_document_id)}/original_render.png` : null,
    revisionImage: run ? `/api/images/${encodeURIComponent(run.revision_document_id)}/original_render.png` : null,
  };
}

async function loadReportsList() {
  setStatus("Loading available reports...");
  const payload = await fetchJson("/api/reports");
  state.reportsPayload = payload;

  state.runsById.clear();
  state.reportById.clear();

  for (const report of payload.reports || []) {
    state.reportById.set(report.id, report);
  }
  for (const run of payload.runs || []) {
    state.runsById.set(run.id, run);
  }

  renderRunOptions(payload.runs || []);
}

function renderRunOptions(runs) {
  elements.runSelect.innerHTML = "";
  if (runs.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No report runs found";
    elements.runSelect.append(option);
    elements.runSelect.disabled = true;
    setStatus("No reports found under data/artifacts/delta.");
    return;
  }

  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.baseline_document_id} -> ${run.revision_document_id}`;
    elements.runSelect.append(option);
  }

  elements.runSelect.disabled = false;
  state.activeRunId = runs[0].id;
  elements.runSelect.value = state.activeRunId;
}

function isChangeListSelectionEnabled() {
  return state.activeTab !== "vlm-comparison";
}

function setChangeDetailsContent(content, { isSummary = false } = {}) {
  elements.changeDetails.textContent = content;
  elements.changeDetails.classList.toggle("change-details--summary", isSummary);
}

function resetSelection() {
  state.selectedIndex = null;
  if (state.activeTab === "vlm-comparison") {
    const summary = state.activeReportData?.summary?.overall_summary;
    setChangeDetailsContent(
      summary || "Change selection is disabled for VLM Comparison.",
      { isSummary: true }
    );
  } else {
    setChangeDetailsContent("Select a change to inspect details.");
  }
  clearOverlay(elements.baselineOverlay);
  clearOverlay(elements.revisionOverlay);
}

function updateSummaryCards(summary) {
  elements.summary.added.textContent = summary.added ?? summary.number_of_meaningful_changes ?? 0;
  elements.summary.removed.textContent = summary.removed ?? 0;
  elements.summary.modified.textContent = summary.modified ?? summary.number_of_meaningful_changes ?? 0;
  elements.summary.unchanged.textContent = summary.unchanged ?? 0;
}

function changeTypeBadge(type) {
  const safe = type || "unknown";
  return `<span class="badge ${safe}">${safe}</span>`;
}

function summarizeChange(change, index) {
  const type = change.change_type || "unknown";
  if (state.activeTab === "text") {
    const beforeText = change?.revision_a?.text || "";
    const afterText = change?.revision_b?.text || "";
    return `${index + 1}. ${beforeText || "(none)"} -> ${afterText || "(none)"}`;
  }
  if (state.activeTab === "geometry") {
    const geomA = change?.revision_a?.geometry_type || "-";
    const geomB = change?.revision_b?.geometry_type || "-";
    return `${index + 1}. ${geomA} -> ${geomB}`;
  }
  if (state.activeTab === "vlm-comparison" || state.activeTab === "vlm") {
    return `${index + 1}. ${change.description || change.location_description || "Visual change"}`;
  }
  return `${index + 1}. ${change.description || change.location_description || "Visual change"}`;
}

function renderFilters() {
  const base = ["all", "added", "removed", "modified", "unchanged", "moved", "unknown"];
  elements.filters.innerHTML = "";

  const availableTypes = new Set(state.allChanges.map((change) => change.change_type || "unknown"));
  const active = ["all", ...base.filter((item) => item !== "all" && availableTypes.has(item))];

  for (const filter of active) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `filter-chip ${state.activeFilter === filter ? "active" : ""}`;
    button.textContent = filter;
    button.addEventListener("click", () => {
      state.activeFilter = filter;
      state.visibleCount = state.activeTab === "geometry" ? 200 : 400;
      applyFilterAndRender();
    });
    elements.filters.append(button);
  }
}

function applyFilterAndRender() {
  const filter = state.activeFilter;
  state.filteredChanges =
    filter === "all"
      ? state.allChanges
      : state.allChanges.filter((change) => (change.change_type || "unknown") === filter);

  renderFilters();
  renderChangeList();
  resetSelection();
}

function renderChangeList() {
  const total = state.filteredChanges.length;
  const shown = Math.min(state.visibleCount, total);
  const selectionEnabled = isChangeListSelectionEnabled();
  elements.changeList.classList.toggle("change-list--readonly", !selectionEnabled);
  elements.listMeta.textContent = selectionEnabled
    ? `${shown} shown of ${total}`
    : `${total} changes (read-only)`;
  elements.changeList.innerHTML = "";

  for (let index = 0; index < shown; index += 1) {
    const change = state.filteredChanges[index];
    const item = document.createElement("li");
    item.className = "change-row";

    const button = document.createElement("button");
    button.type = "button";
    button.className = selectionEnabled
      ? "change-button"
      : "change-button change-button--readonly";
    button.disabled = !selectionEnabled;
    button.innerHTML = `
      <div>${changeTypeBadge(change.change_type || "unknown")} ${summarizeChange(change, index)}</div>
      <div class="change-meta">confidence=${metricNumber(change.confidence)}</div>
    `;

    if (selectionEnabled) {
      button.addEventListener("click", () => selectChange(index));
    }
    item.append(button);
    elements.changeList.append(item);
  }

  elements.loadMore.style.display = shown < total ? "block" : "none";
}

async function ensureRunPageDimensions(run) {
  if (state.baselinePage && state.revisionPage) {
    return;
  }
  const candidateReportId = run?.reports?.text || run?.reports?.geometry;
  if (!candidateReportId) {
    return;
  }
  let candidate = state.reportCache.get(candidateReportId);
  if (!candidate) {
    try {
      candidate = await fetchJson(`/api/reports/${encodeURIComponent(candidateReportId)}`);
      state.reportCache.set(candidateReportId, candidate);
    } catch (_error) {
      return;
    }
  }
  if (candidate?.revision_a && candidate?.revision_b) {
    state.baselinePage = {
      width: candidate.revision_a.page_width,
      height: candidate.revision_a.page_height,
    };
    state.revisionPage = {
      width: candidate.revision_b.page_width,
      height: candidate.revision_b.page_height,
    };
  }
}

function selectChange(index) {
  if (!isChangeListSelectionEnabled()) {
    return;
  }

  state.selectedIndex = index;

  const buttons = Array.from(elements.changeList.querySelectorAll(".change-button"));
  buttons.forEach((button, buttonIndex) => {
    button.classList.toggle("selected", buttonIndex === index);
  });

  const change = state.filteredChanges[index];
  setChangeDetailsContent(JSON.stringify(change, null, 2));
  drawSelectedChange(change);
}

function bboxFromChangeSide(side) {
  if (!side || !side.bbox) {
    return null;
  }
  const { x0, y0, x1, y1 } = side.bbox;
  if (![x0, y0, x1, y1].every((value) => Number.isFinite(value))) {
    return null;
  }
  return side.bbox;
}

function drawSelectedChange(change) {
  clearOverlay(elements.baselineOverlay);
  clearOverlay(elements.revisionOverlay);

  if (!change) {
    return;
  }

  if (state.activeTab === "text" || state.activeTab === "geometry") {
    const baselineBox = bboxFromChangeSide(change.revision_a);
    const revisionBox = bboxFromChangeSide(change.revision_b);

    if (baselineBox && state.baselinePage) {
      drawPageBox(
        elements.baselineOverlay,
        elements.baselineImage,
        baselineBox,
        "#2978b5",
        state.baselinePage
      );
    }

    if (revisionBox && state.revisionPage) {
      drawPageBox(
        elements.revisionOverlay,
        elements.revisionImage,
        revisionBox,
        "#d86f2a",
        state.revisionPage
      );
    }
    return;
  }

  const baselineBox = normalizeBbox(change.baseline_bbox) || bboxFromChangeSide(change.revision_a);
  const revisionBox = normalizeBbox(change.revision_bbox) || bboxFromChangeSide(change.revision_b);

  if (baselineBox) {
    if (change._bbox_space === "image") {
      drawImageSpaceBox(
        elements.baselineOverlay,
        elements.baselineImage,
        baselineBox,
        "#2978b5"
      );
    } else if (isNormalizedBox(baselineBox) && state.baselinePage) {
      drawNormalizedBox(
        elements.baselineOverlay,
        elements.baselineImage,
        baselineBox,
        "#2978b5",
        state.baselinePage
      );
    } else if (state.baselinePage) {
      drawPageBox(
        elements.baselineOverlay,
        elements.baselineImage,
        baselineBox,
        "#2978b5",
        state.baselinePage
      );
    }
  }

  if (revisionBox) {
    console.log("revisionBox", revisionBox);
    const bboxSpace = change._bbox_space || (isNormalizedBox(revisionBox) ? "normalized" : "page");
    if (bboxSpace === "image") {
      drawImageSpaceBox(
        elements.revisionOverlay,
        elements.revisionImage,
        revisionBox,
        "#d86f2a"
      );
    } else if (bboxSpace === "normalized" && state.revisionPage) {
      drawNormalizedBox(
        elements.revisionOverlay,
        elements.revisionImage,
        revisionBox,
        "#d86f2a",
        state.revisionPage
      );
    } else if (state.revisionPage) {
      drawPageBox(
        elements.revisionOverlay,
        elements.revisionImage,
        revisionBox,
        "#d86f2a",
        state.revisionPage
      );
    }
  }
}

async function loadActiveReport() {
  const run = getActiveRun();
  if (!run) {
    setStatus("No run selected.");
    return;
  }

  elements.docIds.textContent = `${run.baseline_document_id} -> ${run.revision_document_id}`;

  const reportId = reportIdForKind(run, state.activeTab);
  if (!reportId) {
    state.activeReportData = null;
    state.allChanges = [];
    updateSummaryCards({ added: 0, removed: 0, modified: 0, unchanged: 0 });
    elements.changeList.innerHTML = "";
    setChangeDetailsContent("This run does not have the selected report type.");
    setImages(`/api/images/${encodeURIComponent(run.baseline_document_id)}/original_render.png`, `/api/images/${encodeURIComponent(run.revision_document_id)}/original_render.png`);
    return;
  }

  try {
    setStatus(`Loading ${state.activeTab} report...`);

    let reportData = state.reportCache.get(reportId);
    if (!reportData) {
      reportData = await fetchJson(`/api/reports/${encodeURIComponent(reportId)}`);
      state.reportCache.set(reportId, reportData);
    }

    let normalized;
    if (state.activeTab === "vlm") {
      await ensureRunPageDimensions(run);
      normalized = normalizeVlm(reportData, run);
    } else if (state.activeTab === "vlm-comparison") {
      await ensureRunPageDimensions(run);
      normalized = normalizeVlmComparison(reportData, run);
      normalized.baselinePage = state.baselinePage;
      normalized.revisionPage = state.revisionPage;
    } else {
      normalized = normalizeTextOrGeometry(reportData);
      normalized.baselineImage = `/api/images/${encodeURIComponent(run.baseline_document_id)}/original_render.png`;
      normalized.revisionImage = `/api/images/${encodeURIComponent(run.revision_document_id)}/original_render.png`;
    }

    state.activeReportData = normalized;
    state.allChanges = normalized.changes;
    state.activeFilter = state.activeTab === "geometry" ? "modified" : "all";
    state.visibleCount = state.activeTab === "geometry" ? 200 : 400;
    state.baselinePage = normalized.baselinePage;
    state.revisionPage = normalized.revisionPage;

    updateSummaryCards(normalized.summary || {});
    setImages(normalized.baselineImage, normalized.revisionImage);
    applyFilterAndRender();

    const tailMessage = normalized.summary?.overall_summary
      ? ` ${normalized.summary.overall_summary}`
      : "";
    setStatus(`Loaded ${state.activeTab} report with ${state.allChanges.length} changes.${tailMessage}`);
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load ${state.activeTab} report: ${error.message}`);
    setChangeDetailsContent("Failed to load report JSON.");
  }
}

function setImages(baselineSrc, revisionSrc) {
  resetAllViewTransforms();
  elements.baselineImage.src = baselineSrc || "";
  elements.revisionImage.src = revisionSrc || "";

  const redrawIfSelected = () => {
    if (state.selectedIndex !== null && isChangeListSelectionEnabled()) {
      drawSelectedChange(state.filteredChanges[state.selectedIndex]);
    }
  };

  elements.baselineImage.onload = redrawIfSelected;
  elements.revisionImage.onload = redrawIfSelected;
}

function setupEventHandlers() {
  setupImageViewports();

  elements.runSelect.addEventListener("change", async (event) => {
    state.activeRunId = event.target.value;
    await loadActiveReport();
  });

  for (const tab of elements.tabs) {
    tab.addEventListener("click", async () => {
      state.activeTab = tab.dataset.tab;
      for (const node of elements.tabs) {
        node.classList.toggle("is-active", node === tab);
      }
      await loadActiveReport();
    });
  }

  elements.loadMore.addEventListener("click", () => {
    state.visibleCount += state.activeTab === "geometry" ? 200 : 400;
    renderChangeList();
  });

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      if (state.selectedIndex !== null) {
        drawSelectedChange(state.filteredChanges[state.selectedIndex]);
      }
    });
    observer.observe(elements.baselineImage);
    observer.observe(elements.revisionImage);
  }
  window.addEventListener("resize", () => {
    if (state.selectedIndex !== null) {
      drawSelectedChange(state.filteredChanges[state.selectedIndex]);
    }
  });
}

async function bootstrap() {
  setupEventHandlers();
  await loadReportsList();
  if (state.activeRunId) {
    await loadActiveReport();
  }
}

bootstrap();
