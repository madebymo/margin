import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import elm from "vite-plugin-elm-watch";

export default defineConfig({
  base: "/static/",
  plugins: [
    svelte(),
    elm({ mode: "auto" }),
  ],
  build: {
    outDir: "../backend/src/tutor/api/static/dist",
    assetsDir: "assets",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/sessions": "http://127.0.0.1:8000",
      "/healthz": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.js"],
  },
});
