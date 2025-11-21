# LocalCat – Autonomous, Self-Learning Transaction Categorisation System

LocalCat is a fully self-hosted AI/ML engine for categorising raw financial
transactions without relying on third-party APIs. It combines a hybrid transformer +
LightGBM ensemble, per-user personalisation adapters, cold-start semantic fallback,
robust preprocessing, and confidence-driven routing.

This repository contains the full source code for training, evaluating, and
deploying LocalCat, along with a reproducible dockerised environment.

---

## 🚀 Project Structure (initial)

localcat/
├─ src/ # All source code (preprocess, training, inference, adapters)
├─ models/ # Saved models (transformer, LightGBM, ensemble, ONNX)
├─ data/ # Input datasets (train/val/test)
├─ docker/ # Docker & environment files
├─ notebooks/ # Evaluation & experiment notebooks
├─ demo/ # Demo assets (GIF, screenshots)
├─ run_demo.sh # 1-command demo script
└─ run_eval.sh # Reproducible evaluation script

---

## ⚙️ Phase Progress

- **Phase 0 – Repo & Infra Bootstrap:** ✅ (initialised)
- **Phase 1 – Data Pipeline & Baseline Model:** ⏳ next
- **Phase 2 – Transformer + Ensemble:** pending
- **Phase 3 – Personalisation:** pending
- **Phase 4 – Explainability & Routing:** pending
- **Phase 5 – Robustness & Quantisation:** pending
- **Phase 6 – Packaging & Final PDF/Video:** pending

---

## 🐳 Docker (initial placeholder)

A working Docker build will be added in later phases. For now, a placeholder file
is included to validate structure.
