# CV Alchemist - Deployment Guide

This guide explains how to deploy CV Alchemist to a sandbox environment as required for the RedRob Hackathon submission.

## **Quick Start - Streamlit Cloud (Recommended)**

### 1. Push code to GitHub
```bash
git add .
git commit -m "Add submission code"
git push origin main
```

### 2. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "New app" and connect your GitHub repository
3. Set repository, branch, and main file path to `app.py`
4. Click "Deploy"

The app will be live at `https://share.streamlit.io/<username>/<repo>/app.py`

## **Alternative: HuggingFace Spaces**

### 1. Create a new Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose Streamlit as the SDK

### 2. Upload your code
- Upload `app.py`, `extract_features.py`, `disqualifier_detection.py`, `logical_validation.py`, `rank.py`
- Upload `outputs/ltr_model.txt`
- Upload `requirements.txt`

### 3. The Space will auto-deploy
Your app will be live at `https://huggingface.co/spaces/<username>/cv-alchemist`

## **Alternative: Replit**

### 1. Create new Replit project
1. Go to https://replit.com
2. Click "Create Repl" → Python
3. Upload all project files

### 2. Create `.replit` file
```
run = "streamlit run app.py"
```

### 3. Run
Click "Run" button — app will be live at `https://cv-alchemist.replit.dev`

## **Alternative: Heroku (Free tier no longer available, but here for reference)**

Use Streamlit Cloud or HuggingFace Spaces instead.

## **Testing Locally**

Before deploying, test locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then visit: http://localhost:8501

## **Key Features to Demo**

1. **Upload a candidate JSON** — app scores and ranks them
2. **Use sample_candidates.json** — shows full pipeline on 50 candidates
3. **View top candidates** — displays reasoning strings
4. **Download submission.csv** — generates submission for top 100

## **Troubleshooting**

### "ModuleNotFoundError"
- Ensure `requirements.txt` is in root directory
- Platform will auto-install on deploy

### "Model not found"
- Ensure `outputs/ltr_model.txt` is committed and pushed to GitHub
- If using HuggingFace, upload the model file manually

### "Slow scoring"
- First run downloads model (30s). Subsequent runs use cache.
- For 100 candidates: ~10-15 seconds to score and rank

## **Submission Checklist**

- [ ] Deployment live and accessible
- [ ] Can upload JSON or use sample
- [ ] Top candidates display with reasoning
- [ ] Download submission.csv works
- [ ] Model loads successfully (green ✓ message)
- [ ] No errors on sample data
- [ ] Copy sandbox URL for submission metadata
