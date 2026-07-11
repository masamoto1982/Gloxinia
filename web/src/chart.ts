// Minimal dependency-free line chart on a 2D canvas. Handles high-DPI displays
// and reads its light/dark colors from CSS custom properties so it always
// matches the page theme.

export interface Series {
  color: string;
  points: number[]; // y-values, indexed the same as `xs`
  /** If set, y is scaled to [0, autoMaxRef.value]; otherwise to [0, yMax]. */
  auto?: boolean;
}

export class LineChart {
  private ctx: CanvasRenderingContext2D;
  private cssVar(name: string, fallback: string): string {
    const v = getComputedStyle(this.canvas).getPropertyValue(name).trim();
    return v || fallback;
  }

  constructor(
    private canvas: HTMLCanvasElement,
    private yMax = 1,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    this.ctx = ctx;
  }

  /** Draw the frame. `xs` are step numbers; each series shares them. */
  draw(xs: number[], series: Series[]): void {
    const dpr = window.devicePixelRatio || 1;
    const cw = this.canvas.clientWidth || 900;
    const ch = this.canvas.clientHeight || 260;
    if (this.canvas.width !== cw * dpr || this.canvas.height !== ch * dpr) {
      this.canvas.width = cw * dpr;
      this.canvas.height = ch * dpr;
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const padL = 44,
      padR = 12,
      padT = 12,
      padB = 24;
    const plotW = cw - padL - padR;
    const plotH = ch - padT - padB;

    const grid = this.cssVar("--grid", "#2a2f3a");
    const axis = this.cssVar("--axis", "#8a93a6");

    // gridlines + y labels (0..yMax)
    ctx.strokeStyle = grid;
    ctx.fillStyle = axis;
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      const val = this.yMax * (1 - i / 4);
      ctx.fillText(fmt(val, this.yMax), padL - 6, y);
    }

    const xmax = xs.length ? xs[xs.length - 1] : 1;
    const xmin = xs.length ? xs[0] : 0;
    const xspan = Math.max(1, xmax - xmin);

    // x labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= 4; i++) {
      const x = padL + (plotW * i) / 4;
      const step = Math.round(xmin + (xspan * i) / 4);
      ctx.fillText(String(step), x, padT + plotH + 6);
    }

    if (xs.length < 2) return;

    for (const s of series) {
      // Per-series scale: auto series fill the plot to their own running max.
      let scale = this.yMax;
      if (s.auto) {
        let m = 1e-9;
        for (const v of s.points) if (v > m) m = v;
        scale = m;
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const px = padL + (plotW * (xs[i] - xmin)) / xspan;
        const norm = Math.min(1, Math.max(0, s.points[i] / scale));
        const py = padT + plotH * (1 - norm);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }
}

function fmt(v: number, max: number): string {
  if (max <= 1.001) return v.toFixed(2);
  if (max >= 100) return Math.round(v).toString();
  return v.toFixed(1);
}
