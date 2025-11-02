# Project Plan: Emotion-Aware Customer Feedback Analysis

| Week | Milestone | Key Tasks | Primary Assets |
| --- | --- | --- | --- |
| 1️⃣ Data Understanding & Setup | Download GoEmotions; clean text; perform exploratory data analysis; add Docker scaffolding | `data/raw/`, `notebooks/01_data_exploration.ipynb`, `docker/` |
| 2️⃣ Baseline Modeling | Implement preprocessing utilities; train TF-IDF + Logistic Regression baseline; evaluate F1/ROC-AUC | `src/preprocess.py`, `src/baseline_model.py` |
| 3️⃣ Transformer Fine-Tuning | Fine-tune `distilroberta-base` on GoEmotions with BCE loss and weighted sampling; persist best model | `src/train_transformer.py`, `notebooks/02_finetuning.ipynb` |
| 4️⃣ Explainability & Interpretation | Add SHAP/LIME explanations; visualize attention weights | `src/explain_model.py`, `notebooks/03_explainability.ipynb` |
| 5️⃣ Clustering & Theme Discovery | Generate sentence embeddings; run UMAP + HDBSCAN to uncover themes | `src/cluster_feedback.py`, `notebooks/04_clustering_analysis.ipynb` |
| 6️⃣ Streamlit App & Deployment | Build Streamlit UI with emotion charts and SHAP visuals; add Docker deployment workflow; finalize documentation | `app/streamlit_app.py`, `README.md`, Docker image |
