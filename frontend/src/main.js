"use strict";

import { mount } from "svelte";

import "@fontsource/space-grotesk/latin-400.css";
import "@fontsource/space-grotesk/latin-500.css";
import "@fontsource/space-grotesk/latin-600.css";
import "@fontsource/space-grotesk/latin-700.css";

import App from "./App.svelte";
import "./app.css";
import "./margin-theme.css";

mount(App, {
  target: document.getElementById("app"),
});
