# Session Context: MLOps Chest X-Ray Project Setup
Date: 2026-04-09

## Project Goal
Implement a professional MLOps pipeline for Chest X-Ray classification (COVID, Pneumonia, Normal) using TensorFlow 2.15 and MLflow, optimized for hardware with limited VRAM (GTX 1050).

## Technical Architecture Analyzed
- **Data Pipeline:** `src/data_pipeline.py` (Asynchronous loading, stratified splits, grayscale conversion).
- **Model:** `src/model_pipeline.py` (Custom CNN Backbone, Focal Loss for class imbalance, 2-phase training strategy).
- **Experimentation:** `notebooks/chest_xray_experiment.py` (GridSearch for LR and Dropout, MLflow tracking).
- **Inference API:** `app/main.py` (FastAPI with Pydantic validation, MLflow metrics retrieval).
- **Config:** `src/config.py` (Centralized hyperparameters and paths).

## Implementation Progress
- [x] Project Analysis: Completed.
- [ ] Step 1: Python 3.11 installation (User reported completed, but system not detecting it).
- [ ] Step 2: Create VENV `xray_env` (Pending - blocked by Python runtime detection).
- [ ] Step 3: Install Clinical Dependencies (TensorFlow 2.15, etc.) (Pending).
- [ ] Step 4: Launch Machinery (MLflow UI, FastAPI, Experiments) (Pending).

## Current Blocker
The system is not recognizing the Python 3.11 installation via the `py` launcher.
**Required Action:** Restart VS Code/Terminal to refresh Environment Variables (PATH).

## Planned Next Steps
1. Verify Python 3.11 detection.
2. Run: `py -3.11 -m venv xray_env`
3. Run: `.\xray_env\Scripts\pip install tensorflow==2.15.0 mlflow fastapi uvicorn pydantic scikit-learn pillow pytest`
4. Execute the ML pipeline.
