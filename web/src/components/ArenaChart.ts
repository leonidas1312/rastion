export type ArenaEvent = {
  t_ms: number;
  iter: number;
  best_value: number;
};

export type ArenaSeries = {
  name: string;
  events: ArenaEvent[];
};

export type ArenaReference = {
  label: string;
  distance: number;
  url?: string;
} | null;

const PALETTE = ["#0d7f73", "#c4472d", "#255cb8", "#8e5eb8", "#1b8a4a", "#bd8b13"];

function range(values: number[]): [number, number] {
  if (!values.length) return [0, 1];
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (min === max) {
    return [min - 1, max + 1];
  }
  return [min, max];
}

export function renderArenaChart(
  canvas: HTMLCanvasElement,
  series: ArenaSeries[],
  focusedSolverName: string | null = null,
  reference: ArenaReference = null,
): void {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 900;
  const height = canvas.clientHeight || 340;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(10, 22, 26, 0.05)";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 62, right: 24, top: 20, bottom: 44 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;

  const allEvents = series.flatMap((row) => row.events);
  const [minX, maxX] = range(allEvents.map((event) => Number(event.t_ms || 0)));
  const yValues = allEvents.map((event) => Number(event.best_value || 0));
  if (reference) {
    yValues.push(Number(reference.distance));
  }
  const [minY, maxY] = range(yValues);

  const mapX = (value: number) => margin.left + ((value - minX) / (maxX - minX || 1)) * chartW;
  const mapY = (value: number) => margin.top + chartH - ((value - minY) / (maxY - minY || 1)) * chartH;

  ctx.strokeStyle = "rgba(120, 137, 156, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + chartH);
  ctx.lineTo(margin.left + chartW, margin.top + chartH);
  ctx.stroke();

  ctx.fillStyle = "rgba(98, 113, 129, 0.9)";
  ctx.font = "12px 'IBM Plex Mono', monospace";
  ctx.fillText(`${minY.toFixed(2)}`, 8, margin.top + chartH);
  ctx.fillText(`${maxY.toFixed(2)}`, 8, margin.top + 12);
  ctx.fillText(`${minX.toFixed(0)}ms`, margin.left, height - 14);
  ctx.fillText(`${maxX.toFixed(0)}ms`, margin.left + chartW - 60, height - 14);

  if (reference) {
    const y = mapY(Number(reference.distance));
    ctx.save();
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = "rgba(139, 95, 32, 0.9)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + chartW, y);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "rgba(139, 95, 32, 0.95)";
    ctx.fillText(`${reference.label} ${Number(reference.distance).toFixed(2)}`, margin.left + 8, Math.max(y - 8, 14));
  }

  series.forEach((row, idx) => {
    if (!row.events.length) return;
    const isFocused = !focusedSolverName || row.name === focusedSolverName;
    const alpha = focusedSolverName && row.name !== focusedSolverName ? 0.2 : 0.95;

    ctx.strokeStyle = `${PALETTE[idx % PALETTE.length]}${Math.floor(alpha * 255)
      .toString(16)
      .padStart(2, "0")}`;
    ctx.lineWidth = isFocused ? 2.8 : 2.0;

    ctx.beginPath();
    row.events.forEach((event, eventIdx) => {
      const x = mapX(Number(event.t_ms));
      const y = mapY(Number(event.best_value));
      if (eventIdx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
}
