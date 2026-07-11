import init, { Trainer, Encoding } from "./wasm/gloxinia.js";
import { LineChart, type Series } from "./chart.ts";
import "./style.css";

// ---- element helpers -------------------------------------------------------
const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

// ---- config from the controls ---------------------------------------------
interface Config {
  p: number;
  trainFrac: number;
  hidden: number;
  lr: number;
  wd: number;
  encoding: Encoding;
  nf: number;
  seed: number;
}

function readConfig(): Config {
  const encBtn = document.querySelector<HTMLButtonElement>("#enc-seg .active")!;
  return {
    p: +($("p") as HTMLInputElement).value,
    trainFrac: +($("tf") as HTMLInputElement).value,
    hidden: +($("hidden") as HTMLInputElement).value,
    lr: +($("lr") as HTMLInputElement).value,
    wd: +($("wd") as HTMLInputElement).value,
    encoding: +encBtn.dataset.enc! === 1 ? Encoding.Fourier : Encoding.OneHot,
    nf: 4,
    seed: +($("seed") as HTMLInputElement).value,
  };
}

// ---- history for the charts -----------------------------------------------
interface History {
  xs: number[];
  trainAcc: number[];
  valAcc: number[];
  valLoss: number[];
  weightNorm: number[];
}
const MAX_POINTS = 600;
function freshHistory(): History {
  return { xs: [], trainAcc: [], valAcc: [], valLoss: [], weightNorm: [] };
}

// ---- app state -------------------------------------------------------------
let trainer: Trainer | null = null;
let running = false;
let raf = 0;
let history = freshHistory();
let grokked = false;

let accChart: LineChart;
let auxChart: LineChart;

// theme colors are read live from CSS variables so charts match light/dark.
const color = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function rebuild(): void {
  stop();
  const c = readConfig();
  if (trainer) trainer.free();
  trainer = new Trainer(c.p, c.trainFrac, c.hidden, c.lr, c.wd, c.encoding, c.nf, c.seed);
  history = freshHistory();
  grokked = false;
  pushMetrics();
  setPhase("init", "Idle — press Run");
  render();
}

function pushMetrics(): void {
  if (!trainer) return;
  const m = trainer.metrics();
  history.xs.push(m.step);
  history.trainAcc.push(m.train_acc);
  history.valAcc.push(m.val_acc);
  history.valLoss.push(m.val_loss);
  history.weightNorm.push(m.weight_norm);
  if (history.xs.length > MAX_POINTS) {
    for (const k of Object.keys(history) as (keyof History)[]) history[k].shift();
  }
  updateStats(m);
  m.free();
}

function updateStats(m: {
  step: number;
  train_acc: number;
  val_acc: number;
  val_loss: number;
  weight_norm: number;
}): void {
  $("s-step").textContent = m.step.toLocaleString();
  $("s-tra").textContent = m.train_acc.toFixed(3);
  $("s-vaa").textContent = m.val_acc.toFixed(3);
  $("s-val").textContent = m.val_loss.toFixed(2);
  $("s-wn").textContent = m.weight_norm.toFixed(1);

  // phase logic
  if (!grokked && m.val_acc >= 0.9) {
    grokked = true;
    setPhase("grok", "⚡ GROKKED — validation generalized");
  } else if (!grokked) {
    if (m.train_acc >= 0.99) setPhase("mem", "Memorizing — train perfect, val at chance");
    else setPhase("warm", "Fitting the training set…");
  }
}

function setPhase(kind: string, text: string): void {
  const el = $("phase");
  el.className = `phase phase-${kind}`;
  el.textContent = text;
}

// ---- the training loop -----------------------------------------------------
function tick(): void {
  if (!running || !trainer) return;
  const stepsPerFrame = +($("speed") as HTMLInputElement).value;
  trainer.step(stepsPerFrame);
  pushMetrics();
  render();
  raf = requestAnimationFrame(tick);
}

function start(): void {
  if (running || !trainer) return;
  running = true;
  $("run").textContent = "⏸ Pause";
  $("run").classList.add("running");
  raf = requestAnimationFrame(tick);
}
function stop(): void {
  running = false;
  cancelAnimationFrame(raf);
  const btn = document.getElementById("run");
  if (btn) {
    btn.textContent = "▶ Run";
    btn.classList.remove("running");
  }
}

// ---- rendering -------------------------------------------------------------
function render(): void {
  const trainC = color("--c-train", "#f4a742");
  const valC = color("--c-val", "#3bd6c6");
  const lossC = color("--c-loss", "#e5679b");
  const normC = color("--c-norm", "#7c9cff");

  const acc: Series[] = [
    { color: trainC, points: history.trainAcc },
    { color: valC, points: history.valAcc },
  ];
  accChart.draw(history.xs, acc);

  const aux: Series[] = [
    { color: lossC, points: history.valLoss, auto: true },
    { color: normC, points: history.weightNorm, auto: true },
  ];
  auxChart.draw(history.xs, aux);
}

// ---- wiring ----------------------------------------------------------------
function bindSlider(id: string, out: string, fmt: (v: number) => string): void {
  const input = $(id) as HTMLInputElement;
  const output = document.getElementById(out);
  const sync = () => {
    if (output) output.textContent = fmt(+input.value);
  };
  input.addEventListener("input", sync);
  sync();
}

function wire(): void {
  bindSlider("wd", "wd-out", (v) => v.toFixed(1));
  bindSlider("lr", "lr-out", (v) => v.toFixed(3));
  bindSlider("tf", "tf-out", (v) => v.toFixed(2));
  bindSlider("p", "p-out", (v) => String(v));
  bindSlider("hidden", "hidden-out", (v) => String(v));
  bindSlider("speed", "speed-out", (v) => String(v));

  // encoding segmented control
  document.querySelectorAll<HTMLButtonElement>("#enc-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#enc-seg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      rebuild();
    });
  });

  // structural knobs rebuild the trainer; wd/lr are read live on next tick,
  // but changing them mid-run is confusing, so rebuild on change too.
  ["p", "hidden", "seed", "tf", "lr", "wd"].forEach((id) => {
    $(id).addEventListener("change", rebuild);
  });

  $("run").addEventListener("click", () => (running ? stop() : start()));
  $("reset").addEventListener("click", rebuild);
}

// ---- boot ------------------------------------------------------------------
async function main(): Promise<void> {
  await init();
  accChart = new LineChart($<HTMLCanvasElement>("chart-acc"), 1);
  auxChart = new LineChart($<HTMLCanvasElement>("chart-aux"), 1);
  wire();
  rebuild();
  window.addEventListener("resize", render);
}

main().catch((err) => {
  console.error(err);
  const phase = document.getElementById("phase");
  if (phase) {
    phase.className = "phase phase-error";
    phase.textContent = "Failed to load the WASM module — see console.";
  }
});
