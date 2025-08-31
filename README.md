# CIRS Risk Categorization with GBERT

Transformer-based explainable classification of Critical Incident Reports (CIRS) using German BERT (GBERT) and SHAP interpretability.

---

## Background
Critical Incident Reporting Systems (CIRS) are widely used in healthcare to improve patient safety. However, narrative free-text reports are difficult to analyze systematically.  
This project investigates whether transformer-based language models can classify anonymized CIRS reports into predefined risk categories and provide interpretable explanations that support clinical risk managers.

---

## Data
- Dataset: 617 anonymized CIRS reports collected at a German university hospital (2018–2024).  
- Labels: Organization, Treatment, Documentation, and Consent/Communication.  
- Annotation: Reports were categorized by the hospital’s risk management team. Interrater agreement: κ = 0.75.  
- **Note:** Due to confidentiality, the dataset cannot be shared. 

---

## Methods
- **Baseline:** TF-IDF + Logistic Regression.  
- **Transformer:** Fine-tuned `deepset/gbert-base` for multi-class sequence classification.  
- **Cross-validation:** Stratified 5-fold CV.  
- **Loss:** Class-weighted cross-entropy with label smoothing (factor 0.1), weights recomputed for each fold.  
- **Hyperparameters:**  
  - Learning rates: {2e−5, 5e−5}  
  - Epochs: {3, 5}  
  - Batch size: 8  
- **Evaluation metrics:** Accuracy, per-class precision/recall/F1, macro-F1, weighted-F1, macro-AUPRC, weighted-AUPRC.  
- **Interpretability:** SHAP values (50 samples per fold), aggregated to word level with perturbation checks for stability.

---

## Results
- **GBERT:** macro-F1 = 0.44, weighted-F1 = 0.75.  
- **TF-IDF baseline:** much lower performance on minority classes.  
- **SHAP analysis:** Highlighted clinically meaningful token contributions  
  
