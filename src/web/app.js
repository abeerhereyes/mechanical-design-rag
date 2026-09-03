const chat = document.querySelector("#chat");
const welcome = document.querySelector("#welcome");
const form = document.querySelector("#composer");
const question = document.querySelector("#question");
const send = document.querySelector("#send");
const courseSelect = document.querySelector("#course-select");
const courseTitle = document.querySelector("#course-title");
const includeGeneral = document.querySelector("#include-general");
const template = document.querySelector("#message-template");
const dialog = document.querySelector("#source-dialog");
const storageKey = "course-notes-chat-v1";
const coursePrompts = {
  "mechanical-design": [
    "When should I check a spring for buckling?",
    "Explain how preload improves bolt fatigue life.",
    "Calculate spring rate for G=79300, wire 3 mm, mean D 25 mm, 8 active coils."
  ],
  aerodynamics: [
    "Explain potential flow around a circular cylinder.",
    "How does circulation create lift in the course notes?",
    "Calculate pressure coefficient for local speed 30 m/s and freestream speed 20 m/s."
  ],
  qrm: [
    "Explain producer risk and consumer risk in acceptance sampling.",
    "When should a p chart be used?",
    "Calculate ARL for signal probability 0.0027."
  ]
};

let history = readHistory();

function readHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) || "[]");
    return Array.isArray(stored) ? stored.slice(-50) : [];
  } catch {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem(storageKey, JSON.stringify(history.slice(-50)));
}

function text(element, value) {
  element.textContent = value == null ? "" : String(value);
}

function formatValue(value) {
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}

function addDetail(container, label, value, warning = false) {
  if (value == null || value === "" || (Array.isArray(value) && !value.length)) return;
  const row = document.createElement("div");
  row.className = `detail${warning ? " warning" : ""}`;
  const title = document.createElement("strong");
  title.textContent = `${label}:`;
  row.append(title, document.createTextNode(
    Array.isArray(value) ? value.join(" · ") : formatValue(value)
  ));
  container.append(row);
}

function renderMessage(entry, persist = false) {
  welcome?.remove();
  const node = template.content.firstElementChild.cloneNode(true);
  text(node.querySelector(".question-text"), entry.query);

  if (entry.loading) {
    text(node.querySelector(".intent-badge"), "Thinking");
    text(node.querySelector(".course-answer"), "Searching your course notes…");
    node.dataset.loading = "true";
    chat.append(node);
    return node;
  }

  const result = entry.result || {};
  text(node.querySelector(".intent-badge"), result.intent || "answer");
  text(
    node.querySelector(".course-answer"),
    result.course_answer || result.answer || "No course-note answer was returned."
  );

  const generalSection = node.querySelector(".general-section");
  if (result.general_answer) {
    text(node.querySelector(".general-answer"), result.general_answer);
  } else {
    generalSection.classList.add("empty");
  }

  const citations = node.querySelector(".citations");
  for (const citation of result.citations || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation";
    button.textContent = `${citation.source} · ${citation.section || "source"} · p. ${citation.page}`;
    button.addEventListener("click", () => openSource(citation));
    citations.append(button);
  }

  const detailsSection = node.querySelector(".engineering-details");
  const details = node.querySelector(".detail-grid");
  addDetail(details, "Calculation", result.calculation);
  addDetail(
    details,
    "Parameters",
    Object.entries(result.parameters || {}).map(([key, value]) => `${key} = ${formatValue(value)}`)
  );
  addDetail(details, "Needed parameters", result.missing_parameters, true);
  addDetail(details, "Assumptions", result.assumptions);
  addDetail(details, "Warnings", result.warnings, true);
  if (details.childElementCount) detailsSection.hidden = false;

  chat.append(node);
  if (persist) {
    history.push(entry);
    saveHistory();
  }
  return node;
}

async function openSource(citation) {
  const content = document.querySelector("#source-content");
  text(document.querySelector("#source-title"), "Loading source…");
  text(document.querySelector("#source-meta"), "");
  content.replaceChildren();
  dialog.showModal();
  try {
    const response = await fetch(citation.preview_url);
    if (!response.ok) throw new Error("This source page is not available.");
    const source = await response.json();
    text(document.querySelector("#source-title"), source.title || source.section);
    text(
      document.querySelector("#source-meta"),
      `${source.source} · ${source.section} · page ${source.page}`
    );
    if (source.kind === "pdf" && source.document_url) {
      const frame = document.createElement("iframe");
      frame.className = "pdf-preview";
      frame.title = `${source.source}, page ${source.page}`;
      frame.src = source.document_url;
      content.append(frame);
    } else {
      text(content, source.content);
    }
  } catch (error) {
    text(document.querySelector("#source-title"), "Source unavailable");
    text(content, error.message);
  }
}

async function loadCourses() {
  try {
    const response = await fetch("/courses");
    if (!response.ok) return;
    const data = await response.json();
    const courses = data.courses || [];
    if (!courses.length) return;
    courseSelect.replaceChildren();
    for (const course of courses) {
      const option = document.createElement("option");
      option.value = course.id;
      option.textContent = course.name;
      option.disabled = course.available === false;
      courseSelect.append(option);
    }
    const remembered = localStorage.getItem("course-notes-current-course");
    if (remembered && courses.some((course) => course.id === remembered)) {
      courseSelect.value = remembered;
    }
    updateCourseTitle();
  } catch {
    // The default course remains usable when metadata is unavailable.
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/ingestion/status");
    const status = await response.json();
    const count = status.documents;
    text(
      document.querySelector("#ingestion-status"),
      count == null ? String(status.status || "Available") : `${count} source${count === 1 ? "" : "s"} available`
    );
    if (status.status !== "empty") document.querySelector("#status-dot").classList.add("ready");
  } catch {
    text(document.querySelector("#ingestion-status"), "Status unavailable");
  }
}

function updateCourseTitle() {
  const option = courseSelect.selectedOptions[0];
  text(courseTitle, option?.textContent || "Course Notes");
  localStorage.setItem("course-notes-current-course", courseSelect.value);
  const prompts = coursePrompts[courseSelect.value] || [];
  document.querySelectorAll(".prompt").forEach((button, index) => {
    if (prompts[index]) button.textContent = prompts[index];
  });
}

async function ask(queryText) {
  const pending = renderMessage({ query: queryText, loading: true });
  send.disabled = true;
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryText,
        params: {},
        course_id: courseSelect.value,
        include_general: includeGeneral.checked,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The assistant could not answer.");
    pending.remove();
    renderMessage({ query: queryText, result: body }, true);
  } catch (error) {
    text(pending.querySelector(".intent-badge"), "Error");
    const answer = pending.querySelector(".course-answer");
    answer.classList.add("error");
    text(answer, error.message);
  } finally {
    send.disabled = false;
    question.focus();
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const queryText = question.value.trim();
  if (!queryText || send.disabled) return;
  question.value = "";
  question.style.height = "";
  ask(queryText);
});

question.addEventListener("input", () => {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight, 160)}px`;
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".prompt").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.textContent;
    form.requestSubmit();
  });
});

document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
courseSelect.addEventListener("change", updateCourseTitle);

function clearChat() {
  history = [];
  localStorage.removeItem(storageKey);
  window.location.reload();
}

document.querySelector("#new-chat").addEventListener("click", clearChat);
document.querySelector("#clear-history").addEventListener("click", clearChat);

history.forEach((entry) => renderMessage(entry));
loadCourses();
loadStatus();
