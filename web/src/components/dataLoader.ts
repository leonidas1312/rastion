export function siteBase(): string {
  const candidate = (window as any).__RASTION_BASE__;
  if (typeof candidate !== "string" || !candidate.length) return "/";
  return candidate.endsWith("/") ? candidate : `${candidate}/`;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${siteBase()}${path.replace(/^\/+/, "")}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${path}`);
  }
  return (await response.json()) as T;
}

export function formatIso(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "n/a";
  return date.toLocaleString();
}
