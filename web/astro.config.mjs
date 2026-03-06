import { defineConfig } from "astro/config";

function normalizeBase(value) {
  if (!value || value === "/") return "/";
  const withLeading = value.startsWith("/") ? value : `/${value}`;
  return withLeading.endsWith("/") ? withLeading : `${withLeading}/`;
}

const repository = process.env.GITHUB_REPOSITORY || "";
const repoName = repository.includes("/") ? repository.split("/")[1] : "";
const defaultBase = process.env.GITHUB_ACTIONS && repoName ? `/${repoName}/` : "/";
const base = normalizeBase(process.env.SITE_BASE || defaultBase);

export default defineConfig({
  output: "static",
  site: process.env.SITE_URL || "https://example.com",
  base,
});
