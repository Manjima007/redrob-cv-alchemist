# 🔮 CV Alchemist | RedRob Hackathon v4 Submission

 **Team Name:** Team Bellpaper

CV Alchemist is an AI-powered candidate ranking pipeline designed to evaluate 100,000+ technical resumes and identify the top 100 fits for a Senior AI Engineer role. It utilizes deterministic feature engineering, strict temporal logic validation, and a **LightGBM LambdaRank** model to operate entirely locally—bypassing the need for slow, expensive LLM APIs at inference time while comfortably executing within a 5-minute CPU budget.

---

## 🚀 Sandbox Demonstration
As per the Stage 1 sandbox requirements, you can interact with a lightweight visualization of our ranking model using our Streamlit Community Cloud deployment. 

🔗 **Live Streamlit App:** `[https://redrob-cv-alchemist-2w6r6sxfrvzw8zamsf3k5m.streamlit.app/]`

> **Note to Judges:** The Streamlit app is intended as a visual sandbox for small datasets (`sample_candidates.json`). For the Stage 3 automated reproduction against the full 100K+ `candidates.jsonl` dataset, please use the headless `generate_submission.py` script detailed below.

---

## 🛠️ How to Reproduce the Submission (Stage 3)

The pipeline is entirely self-contained. It extracts the zipped model, streams the JSONL file, computes features, detects honeypots, ranks the candidates, and outputs the final CSV.

### 1. Prerequisites
Ensure you are running **Python 3.11+** (compatible with modern `pandas` and `numpy` versions, without `distutils` dependencies).

### 2. Install Dependencies
```bash
pip install -r requirements.txt

### 3. Run this command to generate the submission CSV:
```bash
python generate_submission.py --input candidates.jsonl --output submission.csv

### Project File structure:
redrob-cv-alchemist/
├── app.py                     # Streamlit UI (Sandbox Demo)
├── generate_submission.py     # Headless execution script for Stage 3
├── extract_features.py        # JSON parser and feature engineering logic
├── logical_validation.py      # Temporal consistency & Honeypot detection
├── outputs/
│   └── ltr_model.zip          # Zipped LightGBM model weights
├── submission_metadata.yaml   # Required portal metadata
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file