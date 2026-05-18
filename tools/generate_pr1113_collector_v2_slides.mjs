#!/usr/bin/env node

import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pptxgen = require("pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "NVIDIA";
pptx.subject = "ai-dynamo/aiconfigurator PR #1113";
pptx.title = "Collector v2: model-centric support-matrix healing";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "LAYOUT_WIDE";
pptx.margin = 0;
pptx.pageLayout = { name: "LAYOUT_WIDE", width: 13.333, height: 7.5 };
pptx.layout = "LAYOUT_WIDE";
pptx.slideWidth = 13.333;
pptx.slideHeight = 7.5;

const C = {
  bg: "F8FAFC",
  ink: "172033",
  muted: "5B6578",
  line: "CBD5E1",
  slate: "334155",
  blue: "2563EB",
  blueSoft: "DBEAFE",
  teal: "0F766E",
  tealSoft: "CCFBF1",
  amber: "B45309",
  amberSoft: "FEF3C7",
  red: "B91C1C",
  redSoft: "FEE2E2",
  green: "15803D",
  greenSoft: "DCFCE7",
  purple: "6D28D9",
  purpleSoft: "EDE9FE",
  white: "FFFFFF",
};

const W = 13.333;
const H = 7.5;
const SLIDE_FOOTER = "ai-dynamo/aiconfigurator PR #1113";

function addSlide(title, kicker = SLIDE_FOOTER) {
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.08,
    fill: { color: C.blue },
    line: { color: C.blue },
  });
  if (title) {
    slide.addText(title, {
      x: 0.55,
      y: 0.35,
      w: 9.5,
      h: 0.45,
      fontFace: "Aptos Display",
      fontSize: 24,
      bold: true,
      color: C.ink,
      margin: 0,
      fit: "shrink",
    });
  }
  slide.addText(kicker, {
    x: 0.55,
    y: 7.12,
    w: 6,
    h: 0.2,
    fontSize: 8,
    color: "94A3B8",
    margin: 0,
  });
  return slide;
}

function addTitleSlide() {
  const slide = pptx.addSlide();
  slide.background = { color: "EEF6FF" };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: H,
    fill: { color: "EEF6FF" },
    line: { color: "EEF6FF" },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.13,
    fill: { color: C.blue },
    line: { color: C.blue },
  });
  slide.addText("Collector v2", {
    x: 0.72,
    y: 1.6,
    w: 7.8,
    h: 0.75,
    fontFace: "Aptos Display",
    fontSize: 43,
    bold: true,
    color: C.ink,
    margin: 0,
    fit: "shrink",
  });
  slide.addText("Model-centric support-matrix healing", {
    x: 0.75,
    y: 2.45,
    w: 8.2,
    h: 0.45,
    fontSize: 22,
    color: C.slate,
    margin: 0,
    fit: "shrink",
  });
  slide.addText("ai-dynamo/aiconfigurator PR #1113", {
    x: 0.78,
    y: 3.05,
    w: 5.8,
    h: 0.3,
    fontSize: 13,
    color: C.muted,
    margin: 0,
  });
  addBadge(slide, "Problem", 8.9, 1.45, 2.3, 0.48, C.redSoft, C.red);
  addBadge(slide, "Design", 9.55, 2.25, 2.3, 0.48, C.blueSoft, C.blue);
  addBadge(slide, "Workflow", 8.95, 3.05, 2.3, 0.48, C.greenSoft, C.green);
  addBadge(slide, "How-to", 9.5, 3.85, 2.3, 0.48, C.amberSoft, C.amber);
  slide.addText("May 2026", {
    x: 0.78,
    y: 6.75,
    w: 2,
    h: 0.25,
    fontSize: 10,
    color: C.muted,
    margin: 0,
  });
}

function addBadge(slide, text, x, y, w, h, fill, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
  slide.addText(text, {
    x: x + 0.14,
    y: y + 0.11,
    w: w - 0.28,
    h: h - 0.18,
    fontSize: 13,
    bold: true,
    color,
    align: "center",
    margin: 0,
    fit: "shrink",
  });
}

function bullets(slide, items, x, y, w, h, opts = {}) {
  const fontSize = opts.fontSize || 17;
  slide.addText(
    items.map((t) => ({ text: t, options: { bullet: { indent: 14 }, hanging: 4, breakLine: true } })),
    {
      x,
      y,
      w,
      h,
      fontSize,
      color: opts.color || C.ink,
      fit: "shrink",
      valign: "mid",
      margin: 0.03,
      breakLine: false,
      paraSpaceAfterPt: opts.paraSpaceAfterPt || 6,
    },
  );
}

function label(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontSize: opts.size || 12,
    bold: opts.bold || false,
    color: opts.color || C.ink,
    align: opts.align || "left",
    valign: opts.valign || "mid",
    margin: opts.margin === undefined ? 0.06 : opts.margin,
    fit: "shrink",
  });
}

function codeBox(slide, text, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: opts.fill || C.white },
    line: { color: opts.line || C.line, width: 0.8 },
  });
  slide.addText(text, {
    x: x + 0.18,
    y: y + 0.16,
    w: w - 0.36,
    h: h - 0.25,
    fontFace: "Aptos Mono",
    fontSize: opts.size || 10,
    color: opts.color || C.slate,
    breakLine: false,
    fit: "shrink",
    margin: 0,
  });
}

function card(slide, title, body, x, y, w, h, fill, color = C.ink) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
  label(slide, title, x + 0.18, y + 0.14, w - 0.36, 0.26, { size: 11, bold: true, color });
  label(slide, body, x + 0.18, y + 0.48, w - 0.36, h - 0.58, { size: 9.5, color: C.slate, valign: "top" });
}

function node(slide, text, x, y, w, h, fill, color = C.ink) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: C.line, width: 0.6 },
  });
  label(slide, text, x + 0.1, y + 0.08, w - 0.2, h - 0.16, {
    size: 10.5,
    bold: true,
    color,
    align: "center",
  });
}

function arrow(slide, x1, y1, x2, y2, color = C.slate) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: { color, width: 1.2, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function slidePainpoints() {
  const slide = addSlide("Why this PR exists");
  bullets(
    slide,
    [
      "Support-matrix gaps remain after repeated full collector runs.",
      "Agents cannot heal the matrix easily because the collector is op-centric.",
      "Adding a model spreads code edits across model lists, generators, registries, and version forks.",
      "Missing perf points are hard to collect as a targeted one-off case.",
      "WideEP and DSV4 special-image requirements were not first-class metadata.",
      "Perf outputs still need better compression and organization.",
    ],
    0.75,
    1.2,
    7.2,
    4.8,
    { fontSize: 16 },
  );
  card(slide, "Core mismatch", "The matrix gap is usually model + GPU + framework, but the collector asked agents to reason in op buckets.", 8.25, 1.35, 3.85, 1.25, C.redSoft, C.red);
  card(slide, "Agent cost", "No centralized view means more Python archaeology, more token use, and more fragile fixes.", 8.25, 2.95, 3.85, 1.25, C.amberSoft, C.amber);
  card(slide, "Operational cost", "Broad reruns hide the exact missing case and make expected failures look like new breakage.", 8.25, 4.55, 3.85, 1.25, C.blueSoft, C.blue);
}

function slideBeforeArchitecture() {
  const slide = addSlide("Previous architecture");
  node(slide, "Support-matrix healer", 0.7, 1.25, 2.1, 0.64, C.redSoft, C.red);
  node(slide, "collect.py", 3.25, 1.25, 1.35, 0.64, C.white);
  node(slide, "--ops / backend registry", 5.05, 1.25, 2.1, 0.64, C.amberSoft, C.amber);
  node(slide, "sglang/*.py", 7.85, 0.65, 1.65, 0.55, C.white);
  node(slide, "trtllm/*.py", 7.85, 1.42, 1.65, 0.55, C.white);
  node(slide, "vllm/*.py", 7.85, 2.19, 1.65, 0.55, C.white);
  node(slide, "common_test_cases.py", 10.05, 1.42, 2.25, 0.65, C.redSoft, C.red);
  node(slide, "perf result files", 10.05, 3.25, 2.25, 0.65, C.white);
  node(slide, "WideEP mixed into stock registries", 7.45, 4.35, 2.9, 0.72, C.amberSoft, C.amber);
  arrow(slide, 2.8, 1.57, 3.25, 1.57);
  arrow(slide, 4.6, 1.57, 5.05, 1.57);
  arrow(slide, 7.15, 1.57, 7.85, 0.93);
  arrow(slide, 7.15, 1.57, 7.85, 1.69);
  arrow(slide, 7.15, 1.57, 7.85, 2.46);
  arrow(slide, 9.5, 0.93, 10.05, 1.62);
  arrow(slide, 9.5, 1.69, 10.05, 1.7);
  arrow(slide, 9.5, 2.46, 10.05, 1.8);
  arrow(slide, 11.15, 2.07, 11.15, 3.25);
  arrow(slide, 8.9, 2.74, 8.9, 4.35, C.amber);
  label(slide, "Collection unit: op bucket", 0.75, 5.72, 3.6, 0.28, { size: 12, bold: true, color: C.red });
  label(slide, "Support-matrix gap: model/GPU/framework", 5.15, 5.72, 4.5, 0.28, { size: 12, bold: true, color: C.blue });
  label(slide, "Mismatch creates broad reruns and scattered fixes.", 0.75, 6.08, 10.5, 0.32, { size: 13, color: C.muted });
}

function slideBeforeCode() {
  const slide = addSlide("Previous code shape");
  codeBox(
    slide,
    `collector/
  common_test_cases.py
    # model dimensions + shared sweeps in Python
  collect.py
    # op-centric execution
  sglang/
    collect_*.py
    registry.py
      # stock ops + WideEP mixed together
  trtllm/
    collect_*_v1.py
    collect_*_v2.py
    collect_*_v3.py
    registry.py
  vllm/
    collect_*_v1.py
    collect_*_v2.py
    registry.py`,
    0.78,
    1.08,
    5.7,
    5.35,
    { size: 11 },
  );
  card(slide, "Scattered model intent", "Model dimensions lived in Python lists and collector-specific helpers.", 6.95, 1.2, 4.9, 0.9, C.redSoft, C.red);
  card(slide, "Scattered exclusions", "Hardware and framework constraints appeared as guards, special cases, or runtime failures.", 6.95, 2.45, 4.9, 0.9, C.amberSoft, C.amber);
  card(slide, "Scattered runtimes", "Special images were not tied to a manifest or isolated collector namespace.", 6.95, 3.7, 4.9, 0.9, C.blueSoft, C.blue);
  card(slide, "Hard to review", "A reviewer cannot quickly answer: which model gained which cases on which GPU?", 6.95, 4.95, 4.9, 0.9, C.purpleSoft, C.purple);
}

function slidePrinciple() {
  const slide = addSlide("New design principle");
  label(slide, "Make the model the unit of collection intent", 0.75, 1.2, 8.7, 0.62, {
    size: 28,
    bold: true,
    color: C.ink,
  });
  bullets(
    slide,
    [
      "FPM and support-matrix healing both start from a model.",
      "Collector v2 resolves model + GPU/SM into a planned case set.",
      "Base op sweeps stay reusable.",
      "Model dimensions stay in model YAML.",
      "Hardware and framework exclusions stay in SM exception YAML.",
      "Python collectors focus on generating and running cases.",
    ],
    0.9,
    2.25,
    7.2,
    3.9,
    { fontSize: 17 },
  );
  node(slide, "model", 9.0, 1.45, 1.4, 0.62, C.greenSoft, C.green);
  node(slide, "GPU/SM", 10.75, 1.45, 1.4, 0.62, C.blueSoft, C.blue);
  node(slide, "case plan", 9.86, 2.75, 1.65, 0.62, C.white);
  node(slide, "collectors", 9.86, 4.05, 1.65, 0.62, C.amberSoft, C.amber);
  arrow(slide, 9.7, 2.07, 10.35, 2.75, C.green);
  arrow(slide, 11.45, 2.07, 10.75, 2.75, C.blue);
  arrow(slide, 10.68, 3.37, 10.68, 4.05, C.slate);
}

function slideNewArchitecture() {
  const slide = addSlide("New architecture");
  node(slide, "FPM / healer", 0.5, 1.0, 1.5, 0.55, C.greenSoft, C.green);
  node(slide, "model path + GPU", 2.35, 1.0, 1.75, 0.55, C.blueSoft, C.blue);
  node(slide, "collect.py", 4.45, 1.0, 1.25, 0.55, C.white);
  node(slide, "model_cases.py planner", 6.0, 1.0, 2.05, 0.55, C.white);
  node(slide, "filtered op case plan", 8.45, 1.0, 2.1, 0.55, C.blueSoft, C.blue);
  node(slide, "collectors", 11.05, 1.0, 1.45, 0.55, C.white);
  arrow(slide, 2.0, 1.28, 2.35, 1.28, C.green);
  arrow(slide, 4.1, 1.28, 4.45, 1.28, C.blue);
  arrow(slide, 5.7, 1.28, 6.0, 1.28);
  arrow(slide, 8.05, 1.28, 8.45, 1.28);
  arrow(slide, 10.55, 1.28, 11.05, 1.28);
  node(slide, "base_ops/<op>.yaml", 2.1, 2.5, 2.0, 0.52, C.white);
  node(slide, "models/<Architecture>_cases.yaml", 4.55, 2.5, 2.95, 0.52, C.greenSoft, C.green);
  node(slide, "sm_exceptions/sm<version>.yaml", 7.95, 2.5, 2.9, 0.52, C.amberSoft, C.amber);
  arrow(slide, 3.1, 2.5, 6.7, 1.55, C.slate);
  arrow(slide, 6.0, 2.5, 6.95, 1.55, C.green);
  arrow(slide, 9.35, 2.5, 7.4, 1.55, C.amber);
  node(slide, "framework_manifest.yaml", 4.15, 4.0, 2.35, 0.52, C.purpleSoft, C.purple);
  node(slide, "stock registries", 7.05, 4.0, 1.65, 0.52, C.white);
  node(slide, "wideep/* registries", 9.05, 4.0, 1.9, 0.52, C.tealSoft, C.teal);
  arrow(slide, 5.35, 4.0, 5.05, 1.55, C.purple);
  arrow(slide, 9.5, 1.55, 7.9, 4.0, C.slate);
  arrow(slide, 9.5, 1.55, 10.0, 4.0, C.teal);
  label(slide, "The planned case set is visible before collection via --plan-only.", 0.75, 6.08, 9.5, 0.35, {
    size: 14,
    bold: true,
    color: C.slate,
  });
}

function slideNewCode() {
  const slide = addSlide("New code shape");
  codeBox(
    slide,
    `collector/
  framework_manifest.yaml
  framework_manifest.py
  collect.py
  model_cases.py
  case_generator.py
  cases/
    base_op_cases.yaml
    base_ops/<op>.yaml
    models/<Architecture>_cases.yaml
    sm_exceptions/sm<version>_exceptions.yaml
  sglang/
  trtllm/
  vllm/
  wideep/
    sglang/
    trtllm/
  network/`,
    0.78,
    1.06,
    5.1,
    5.6,
    { size: 11 },
  );
  card(slide, "Central review surface", "YAML shows base sweeps, model-specific cases, and SM-specific exceptions.", 6.35, 1.1, 5.2, 0.9, C.greenSoft, C.green);
  card(slide, "Cleaner collector code", "Collectors generate runnable cases from YAML-backed specs instead of owning policy.", 6.35, 2.35, 5.2, 0.9, C.blueSoft, C.blue);
  card(slide, "Isolated WideEP", "Special-image collectors live under collector/wideep and are requested by the plan.", 6.35, 3.6, 5.2, 0.9, C.tealSoft, C.teal);
  card(slide, "Full or narrow", "The same planner supports full model-centric runs and support-matrix healing slices.", 6.35, 4.85, 5.2, 0.9, C.amberSoft, C.amber);
}

function slideYamlContract() {
  const slide = addSlide("Centralized YAML contract");
  codeBox(
    slide,
    `# base_ops/gemm.yaml
all_frameworks_op_cases:
  gemm:
    cases:
      - id: base_transformer_gemm_shape_sweep
        token_counts: [1, 2, 4, 8, 16]
        feature_sizes: [128, 256, 512]`,
    0.72,
    1.15,
    5.75,
    2.35,
    { size: 10 },
  );
  codeBox(
    slide,
    `# models/DeepseekV4ForCausalLM_cases.yaml
architecture: DeepseekV4ForCausalLM
model_paths:
  - sgl-project/DeepSeek-V4-Flash-FP8
all_frameworks_op_cases:
  moe:
    cases: all
framework_specific_op_cases:
  sglang:
    wideep_moe:
      cases: all`,
    6.85,
    1.15,
    5.75,
    2.9,
    { size: 10 },
  );
  codeBox(
    slide,
    `# sm_exceptions/sm120_exceptions.yaml
framework_specific_op_exceptions:
  sglang:
    moe:
      rules:
        - reason_type: framework_version_unsupported
          version_prefixes: ["0.5.10"]
          match:
            moe_type: nvfp4`,
    0.72,
    4.05,
    5.75,
    2.15,
    { size: 10 },
  );
  card(slide, "Planning merge order", "Base op cases + model cases + SM exceptions -> filtered per-op plan.", 6.85, 4.35, 5.75, 0.85, C.blueSoft, C.blue);
  card(slide, "Stable selectors", "cases: all, case_ids, contains, indices, ranges, limit, and structured rules.", 6.85, 5.45, 5.75, 0.85, C.greenSoft, C.green);
}

function slideHealingFlow() {
  const slide = addSlide("Targeted healing flow");
  const steps = [
    ["Gap", "model X missing perf point on GPU Y", C.redSoft, C.red],
    ["Command", "collect.py --model-path X --gpu Y", C.blueSoft, C.blue],
    ["Plan", "merge base + model + SM exceptions", C.white, C.slate],
    ["Select", "exact IDs, contains, ranges, indices, limit", C.amberSoft, C.amber],
    ["Run", "collectors execute only needed cases", C.greenSoft, C.green],
  ];
  const xs = [0.65, 3.0, 5.45, 7.9, 10.35];
  for (let i = 0; i < steps.length; i += 1) {
    const [t, b, fill, color] = steps[i];
    node(slide, t, xs[i], 1.45, 1.65, 0.55, fill, color);
    label(slide, b, xs[i] - 0.25, 2.18, 2.15, 0.8, { size: 10.5, align: "center", color: C.slate });
    if (i < steps.length - 1) arrow(slide, xs[i] + 1.65, 1.72, xs[i + 1], 1.72, C.slate);
  }
  codeBox(
    slide,
    `python3 collect.py --backend sglang \\
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \\
  --gpu b200_sxm \\
  --plan-only`,
    1.0,
    4.35,
    5.6,
    1.25,
    { size: 12, fill: "F1F5F9" },
  );
  bullets(
    slide,
    [
      "Whole-model collection and one-off missing cases use the same planner.",
      "Expected SM/framework gaps are skipped before collection where possible.",
      "Runtime-only expected failures are checkpointed as expected_failed.",
    ],
    7.0,
    4.25,
    4.9,
    1.55,
    { fontSize: 13 },
  );
}

function slideCollectorBehavior() {
  const slide = addSlide("Collector behavior");
  bullets(
    slide,
    [
      "Op collectors can receive model_path directly.",
      "Legacy collectors still use COLLECTOR_MODEL_PATH while they are migrated.",
      "Model-specific cases may overlap with base cases.",
      "Full mode aggregates base cases plus every model YAML.",
      "Subset support comes from central selectors, not per-collector ad hoc logic.",
    ],
    0.8,
    1.2,
    6.3,
    4.0,
    { fontSize: 16 },
  );
  codeBox(
    slide,
    `# Full model-centric refresh
python3 collect.py --backend trtllm --model-cases-full

# Narrow support-matrix healing
python3 collect.py --backend sglang \\
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \\
  --gpu b200_sxm`,
    7.2,
    1.45,
    5.25,
    2.55,
    { size: 11.5, fill: C.white },
  );
  card(slide, "Result", "Collectors can collect a very specific set of cases for a model or update a single missing point.", 7.2, 4.55, 5.25, 1.0, C.greenSoft, C.green);
}

function slideWideEp() {
  const slide = addSlide("WideEP and images");
  node(slide, "stock sglang registry", 0.8, 1.35, 2.3, 0.6, C.white);
  node(slide, "stock trtllm registry", 0.8, 2.4, 2.3, 0.6, C.white);
  node(slide, "collector/wideep/sglang", 4.25, 1.35, 2.45, 0.6, C.tealSoft, C.teal);
  node(slide, "collector/wideep/trtllm", 4.25, 2.4, 2.45, 0.6, C.tealSoft, C.teal);
  arrow(slide, 3.1, 1.65, 4.25, 1.65, C.teal);
  arrow(slide, 3.1, 2.7, 4.25, 2.7, C.teal);
  codeBox(
    slide,
    `wideep:
  sglang:
    version: "0.5.10"
    images:
      default: "deepseek-v4-blackwell"
      grace_blackwell: "deepseek-v4-grace-blackwell"
    collector_dir: "collector/wideep/sglang"`,
    7.3,
    1.2,
    5.1,
    2.35,
    { size: 10.3 },
  );
  bullets(
    slide,
    [
      "WideEP is a first-class namespace instead of a stock-registry surprise.",
      "Special images are declared in framework_manifest.yaml.",
      "DSV4/WideEP runtime requirements become reviewable metadata.",
    ],
    1.0,
    4.35,
    10.4,
    1.45,
    { fontSize: 15 },
  );
}

function slideHowToModelGpu() {
  const slide = addSlide("How to add model or GPU support");
  card(slide, "New model", "1. Create cases/models/<Architecture>_cases.yaml\n2. Add architecture, model_path, model_paths\n3. Set include_base if shared sweeps apply\n4. Add model_case_values\n5. Add all/framework op cases\n6. Add a new op only if existing ops cannot generate it\n7. Validate with --plan-only", 0.75, 1.2, 5.65, 4.85, C.greenSoft, C.green);
  card(slide, "New GPU", "1. Add or reuse system YAML mapping GPU to SM\n2. Create sm_exceptions/sm<version>_exceptions.yaml for a new SM\n3. Add gpu_types for concrete hardware\n4. Add all/framework exception rules\n5. Use reason_type for hardware vs framework gaps\n6. Use known_exceptions for subprocess-only failures", 6.95, 1.2, 5.65, 4.85, C.blueSoft, C.blue);
}

function slideAgentImpact() {
  const slide = addSlide("What changes for agents");
  const items = [
    ["Question", "Which model, GPU, framework, and missing case?", C.blueSoft, C.blue],
    ["Answer", "Inspect a small YAML set instead of reconstructing Python behavior.", C.greenSoft, C.green],
    ["Action", "Run a narrow, reproducible healing command.", C.amberSoft, C.amber],
    ["Outcome", "Less broad rerunning, more intentional perf data.", C.purpleSoft, C.purple],
  ];
  for (let i = 0; i < items.length; i += 1) {
    const x = 0.85 + (i % 2) * 6.05;
    const y = 1.35 + Math.floor(i / 2) * 2.0;
    card(slide, items[i][0], items[i][1], x, y, 5.25, 1.1, items[i][2], items[i][3]);
  }
  label(slide, "The same plan can serve FPM, support-matrix repair, and full framework-version refreshes.", 1.0, 5.75, 11.1, 0.35, { size: 15, bold: true, color: C.slate, align: "center" });
}

function slideFollowUps() {
  const slide = addSlide("Follow-ups");
  bullets(
    slide,
    [
      "Compress and organize perf result files by model, GPU, framework, op, and run timestamp.",
      "Add more examples of arbitrary case specs for one-off debugging.",
      "Continue migrating collectors to accept model_path directly.",
      "Keep PR review focused on whether YAML intent matches support-matrix needs.",
    ],
    1.0,
    1.35,
    7.8,
    3.6,
    { fontSize: 17 },
  );
  node(slide, "Cleaner review", 9.45, 1.55, 2.25, 0.62, C.blueSoft, C.blue);
  node(slide, "Narrower healing", 9.45, 2.8, 2.25, 0.62, C.greenSoft, C.green);
  node(slide, "Better outputs", 9.45, 4.05, 2.25, 0.62, C.amberSoft, C.amber);
  arrow(slide, 10.58, 2.17, 10.58, 2.8, C.slate);
  arrow(slide, 10.58, 3.42, 10.58, 4.05, C.slate);
}

addTitleSlide();
slidePainpoints();
slideBeforeArchitecture();
slideBeforeCode();
slidePrinciple();
slideNewArchitecture();
slideNewCode();
slideYamlContract();
slideHealingFlow();
slideCollectorBehavior();
slideWideEp();
slideHowToModelGpu();
slideAgentImpact();
slideFollowUps();

await pptx.writeFile({ fileName: "docs/pr1113_collector_v2_slides.pptx" });
