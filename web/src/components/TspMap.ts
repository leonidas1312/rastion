export type TspNode = {
  id: number;
  x: number;
  y: number;
};

function bounds(nodes: TspNode[]): { minX: number; maxX: number; minY: number; maxY: number } {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const node of nodes) {
    minX = Math.min(minX, node.x);
    maxX = Math.max(maxX, node.x);
    minY = Math.min(minY, node.y);
    maxY = Math.max(maxY, node.y);
  }

  if (!nodes.length) {
    minX = 0;
    maxX = 1;
    minY = 0;
    maxY = 1;
  }

  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }

  return { minX, maxX, minY, maxY };
}

export function renderTspMap(
  canvas: HTMLCanvasElement,
  nodes: TspNode[],
  depot: number,
  routes: Array<{ name: string; route: number[] }>,
  focusedSolverName: string | null,
): void {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 960;
  const height = canvas.clientHeight || 420;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const palette = ["#0d7f73", "#c4472d", "#255cb8", "#8e5eb8", "#1b8a4a", "#bd8b13"];
  const margin = 28;
  const box = bounds(nodes);
  const sx = (width - margin * 2) / (box.maxX - box.minX);
  const sy = (height - margin * 2) / (box.maxY - box.minY);

  const mapPoint = (node: TspNode) => {
    const x = margin + (node.x - box.minX) * sx;
    const y = height - margin - (node.y - box.minY) * sy;
    return { x, y };
  };

  const byId = new Map(nodes.map((node) => [node.id, node]));

  for (let idx = 0; idx < routes.length; idx += 1) {
    const row = routes[idx];
    if (!row.route || row.route.length < 2) continue;

    const dim = focusedSolverName && row.name !== focusedSolverName;
    const alpha = dim ? 0.18 : 0.88;

    ctx.strokeStyle = `${palette[idx % palette.length]}${Math.floor(alpha * 255)
      .toString(16)
      .padStart(2, "0")}`;
    ctx.lineWidth = dim ? 1.2 : 2.0;
    ctx.beginPath();

    let moved = false;
    for (const nodeId of row.route) {
      const node = byId.get(nodeId);
      if (!node) continue;
      const p = mapPoint(node);
      if (!moved) {
        ctx.moveTo(p.x, p.y);
        moved = true;
      } else {
        ctx.lineTo(p.x, p.y);
      }
    }
    if (moved) {
      ctx.stroke();
    }
  }

  for (const node of nodes) {
    const p = mapPoint(node);
    const isDepot = node.id === depot;
    ctx.fillStyle = isDepot ? "#c4472d" : "#1f2d39";
    ctx.beginPath();
    ctx.arc(p.x, p.y, isDepot ? 4.6 : 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  const depotNode = byId.get(depot);
  if (depotNode) {
    const p = mapPoint(depotNode);
    ctx.fillStyle = "#c4472d";
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.fillText("depot", p.x + 7, p.y - 7);
  }
}
