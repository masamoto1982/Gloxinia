import { defineConfig } from "vite";

// GitHub Pages serves a project site from https://<user>.github.io/<repo>/,
// so assets must be requested under "/Gloxinia/". Override with BASE_PATH=/ for
// local `vite preview` at the root if desired.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/Gloxinia/",
  build: {
    target: "es2020",
    outDir: "dist",
  },
});
