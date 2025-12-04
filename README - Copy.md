# 🧩 Keypoint Estimation on COCO – Project Roadmap

## 🎯 Goal
Develop a modular and extensible **PyTorch framework** for human keypoint estimation on the **COCO dataset**, focused on research flexibility, reproducibility, and ONNX export for deployment.

---

## 🪜 Milestones

### 🧩 Milestone 1 — Project Framework
**Goal:** Define a clean, research-oriented codebase layout.
- [X] Integrate a configuration system (e.g., `yaml` or `omegaconf`) for experiment management.
- [X] Add logging utilities (TensorBoard or WandB-ready).
- [X] Provide an entry point for reproducible training runs.

**Branch:** `framework/setup`

---

### 🧮 Milestone 2 — Dataset & Preprocessing
**Goal:** Implement flexible COCO keypoint data interface.
- [ ] Implement dataset class handling images, annotations, and visibility flags.
- [ ] Generate target heatmaps and keypoint masks.
- [ ] Design augmentation and preprocessing pipeline (flip, affine transforms, color jitter).
- [ ] Include visualization utilities for sanity-checking augmentations and annotations.

**Branch:** `data/pipeline`

---

### 🧠 Milestone 3 — Model Architecture
**Goal:** Build and abstract the model design.
- [ ] Implement a configurable baseline (e.g., ResNet + deconv head).
- [ ] Support easy swapping of backbones and heads.
- [ ] Optionally experiment with HRNet or MobileNet backbones.
- [ ] Document design rationale (tradeoff between accuracy and speed).

**Branch:** `model/architecture`

---

### 🔁 Milestone 4 — Training & Optimization
**Goal:** Design a reusable training loop.
- [ ] Implement generic training and validation loops with clear interfaces.
- [ ] Support flexible loss registration (e.g., MSE, Wing, OKS-based losses).
- [ ] Add callbacks for checkpointing, early stopping, LR scheduling.
- [ ] Enable experiment logging and metric tracking.

**Deliverable:** Training runs reproducibly on a COCO subset.  
**Branch:** `train/core`

---

### 📊 Milestone 5 — Evaluation & Visualization
**Goal:** Implement robust model evaluation and qualitative analysis.
- [ ] Integrate COCO evaluation API for mAP and OKS.
- [ ] Add visual evaluation: predicted vs. GT keypoints overlay.
- [ ] Include performance summaries per joint type or difficulty level.
- [ ] Export evaluation results in structured format (CSV/JSON).

**Branch:** `eval/metrics`

---

### 📦 Milestone 6 — Model Export & ONNX Inference
**Goal:** Enable model deployment and lightweight inference.
- [ ] Implement ONNX export with dynamic input support.
- [ ] Validate exported model with `onnxruntime`.
- [ ] Benchmark inference speed on CPU/GPU.
- [ ] Provide minimal inference API for downstream integration.

**Branch:** `export/onnx`

---

### 🧾 Milestone 7 — Codebase Refinement & Documentation
**Goal:** Finalize research-grade usability and clarity.
- [ ] Refine module interfaces (data/model/train/eval).
- [ ] Add docstrings, type hints, and architectural diagrams.
- [ ] Update `README` with training commands, results, and visualization examples.
- [ ] Ensure reproducibility (seeds, configs, versions).

**Branch:** `docs/refinement`

---

### 🚀 Milestone 8 — Experimental Extensions
**Goal:** Explore extensions and performance improvements.
- [ ] Introduce domain randomization or weather-based augmentations for robustness.
- [ ] Implement lightweight models for edge deployment.
- [ ] Compare alternative losses (e.g., Adaptive Wing, OKS-based).
- [ ] Add unit tests for core utilities and key modules.

**Branch examples:**
- `research/robustness`
- `research/loss-experiments`
- `research/lightweight-model`

---

### 📦 Milestone 9 — Submission & Packaging
**Goal:** Prepare professional project delivery.
- [ ] Generate `setup_instructions.md` and `experiments_overview.md`.
- [ ] Include a minimal validation dataset and sample visualizations.
- [ ] Create reproducible zip export (no large data).
- [ ] Tag `v1.0` release.

**Branch:** `release/v1.0`

---

## 🌱 Optional 2-Week Extension Ideas
- Add self-supervised pretraining on human datasets.
- Integrate a transformer-based head for keypoint refinement.
- Conduct cross-dataset evaluation (e.g., MPII, CrowdPose).
- Add mixed-precision training and distributed support.

---

## 🧭 Progress Tracker
| Milestone | Status | Branch |
|------------|---------|--------|
| 1. Framework | ☐ | framework/setup |
| 2. Data Pipeline | ☐ | data/pipeline |
| 3. Model Architecture | ☐ | model/architecture |
| 4. Training | ☐ | train/core |
| 5. Evaluation | ☐ | eval/metrics |
| 6. ONNX Export | ☐ | export/onnx |
| 7. Documentation | ☐ | docs/refinement |
| 8. Research Extensions | ☐ | research/* |
| 9. Submission | ☐ | release/v1.0 |

---

## 💡 Notes
- Follow semantic commits:  
  `feat(train): add OKS loss`  
  `refactor(model): modularize backbone config`  
  `docs(readme): update usage section`
- Keep each milestone in a separate branch → merge into `main` via PRs.
- Maintain a clean experiment log (`logs/experiments.json`) for reproducibility.

---