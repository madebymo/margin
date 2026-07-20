"use strict";

import { mount, unmount } from "svelte";

import "./app.css";
import { api } from "./api.js";
import WidgetHost from "./widgets/WidgetHost.svelte";

let sessionId = null;
const widgetRoots = [];
const el = (id) => document.getElementById(id);
const log = el("log");
const answer = el("answer");
const send = el("send");
const hintBtn = el("hintBtn");
const phase = el("phase");
const summary = el("summary");

function add(kind, text) {
  const div = document.createElement("div");
  div.className = "bubble " + kind;
  if (!["you", "tutor", "hint", "error"].includes(kind)) {
    const tag = document.createElement("span");
    tag.className = "kind-tag";
    tag.textContent = kind;
    div.appendChild(tag);
  }
  div.appendChild(document.createTextNode(text));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function destroyWidgets() {
  for (const root of widgetRoots.splice(0)) {
    unmount(root);
  }
}

function renderWidget(item) {
  const target = document.createElement("div");
  target.className = "widget-mount";
  log.appendChild(target);
  const root = mount(WidgetHost, {
    target,
    props: {
      item,
      sessionId,
      onError: (error) => add("error", error.message),
    },
  });
  widgetRoots.push(root);
  log.scrollTop = log.scrollHeight;
}

function applyTurn(turn) {
  sessionId = turn.session_id;
  for (const item of turn.interactions) {
    add(item.kind === "message" ? "tutor" : item.kind, item.text);
    if (item.widget) {
      renderWidget(item);
    }
  }
  phase.textContent = turn.phase;
  if (turn.llm_enabled === false) {
    add("tutor", "(LLM unavailable — using template content.)");
  }
  const active = turn.phase !== "done" && turn.phase !== "stopped";
  answer.disabled = send.disabled = hintBtn.disabled = !active;
  if (!active) {
    summary.style.display = "block";
    summary.textContent =
      "Session summary\n" + JSON.stringify(turn.summary, null, 2);
  } else {
    answer.focus();
  }
}

async function guard(fn) {
  try {
    await fn();
  } catch (error) {
    add("error", error.message);
  }
}

el("start").onclick = () =>
  guard(async () => {
    destroyWidgets();
    log.innerHTML = "";
    summary.style.display = "none";
    phase.textContent = "starting…";
    applyTurn(
      await api("/sessions", {
        target_kc: el("target").value.trim(),
        llm: el("llm").value === "on",
      }),
    );
  });

async function submitAnswer() {
  const value = answer.value.trim();
  if (!value || !sessionId) {
    return;
  }
  add("you", value);
  answer.value = "";
  await guard(async () =>
    applyTurn(
      await api(`/sessions/${sessionId}/answer`, { answer: value }),
    ),
  );
}

send.onclick = submitAnswer;
answer.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    submitAnswer();
  }
});

hintBtn.onclick = () =>
  guard(async () => {
    const data = await api(`/sessions/${sessionId}/hint`, {});
    add("hint", data.hint || "No more hints — give it your best try.");
  });
