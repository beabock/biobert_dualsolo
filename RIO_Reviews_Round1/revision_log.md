# Revision Log: Bock RIO 2025 BioBERT Paper

This document tracks all changes made in response to reviewer comments for the R2R (Response to Reviewers) document.

---

## Revision Summary

| Date | Change Description | Reviewer Comment(s) Addressed |
|------|-------------------|-------------------------------|
| 2024-12-30 | Added four-model comparison (BERT-uncased, BERT-cased, BioBERT, BiodivBERT) | R1-2a |
| 2024-12-30 | Implemented stratified 5-fold cross-validation | R1-2c, R2-4 |
| 2024-12-30 | Added timing instrumentation for each model | — |
| 2024-12-30 | Added fold-level predictions export | — |
| 2024-12-30 | Added error analysis with multi-model misclassification flagging | R1-5b |
| 2024-12-30 | Clarified validation set approach (k-fold eliminates ambiguity) | R2-5 |
| 2024-12-30 | Documented actual epoch setting: 20 max with early stopping (patience=3) | R1-2b |

---

## Detailed Change Log

### 1. Four-Model Baseline Comparison (R1-2a)

**Reviewer Comment:** "I believe the inclusion of a standard BERT (i.e., bert-base-uncased from Hugging Face) or preferably a BiodivBERT as a baseline comparison for BioBERT would be useful."

**Changes Made:**
- Refactored `generate_report.py` to support multiple models
- Added four models for comparison:
  1. `google-bert/bert-base-uncased` — Standard BERT baseline (uncased)
  2. `google-bert/bert-base-cased` — Standard BERT baseline (cased, for direct BioBERT comparison)
  3. `monologg/biobert_v1.1_pubmed` — Domain-specific biomedical model
  4. `NoYo25/BiodivBERT` — Domain-specific biodiversity model
- All models trained with identical hyperparameters for fair comparison

---

### 2. Stratified 5-Fold Cross-Validation (R1-2c, R2-4)

**Reviewer Comments:**
- R1: "Given the limited number of instances, have you tried performing a n-fold cross-validation?"
- R2: "Authors should consider the implementation of a cross-validation strategy"

**Changes Made:**
- Combined train.csv and test.csv (56 total samples) for pure k-fold evaluation
- Implemented `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Each model evaluated on identical fold splits for fair comparison
- Results reported as mean ± standard deviation across folds

---

### 3. Validation Set Clarification (R2-5)

**Reviewer Comment:** "The text says that only training and testing sets were used... However, the model learning curves show that there is also a validation set."

**Changes Made:**
- K-fold cross-validation eliminates ambiguity
- Each fold: 4/5 of data for training (with internal validation split), 1/5 for testing
- Clear documentation of fold structure in methodology

---

### 4. Epochs Discrepancy (R1-2b)

**Reviewer Comment:** "The manuscript mentions 10 epochs... but the code trains with 20 epochs."

**Changes Made:**
- Documented actual training configuration:
  - Maximum epochs: 20
  - Early stopping: patience=3 on eval_loss
  - Best model selected based on lowest eval_loss
- Paper will be updated to reflect actual settings

---

### 5. Error Analysis (R1-5b)

**Reviewer Comment:** "Consider... manually inspecting the misclassified abstracts to determine why they were misclassified."

**Changes Made:**
- Added error analysis output to `figures/error_analysis.csv`
- Columns: abstract_text, true_label, predicted_label, fold, model
- Flagged samples misclassified by multiple models for priority review
- Enables manual inspection of problematic abstracts

---

## Pending Changes (To Be Addressed)

| Reviewer Comment | Status | Notes |
|-----------------|--------|-------|
| R1-1a: State search terms for Web of Science | **Completed** | Added to Methods section with full Boolean queries and result counts |
| R1-1b: Clarify >512 token handling | **Completed** | Clarified that truncation is used, but all abstracts fell within limit |
| R1-1c: Provide trophic mode label list | Pending | Supplementary material |
| R1-1d: State labeler count/disagreement process | **Completed** | Single labeler (BMB) noted in Methods |
| R1-3a: Move Dataset Characteristics to Methods | Pending | Paper restructure |
| R1-4a: Clarify macro vs binary averaging | Pending | Code uses macro; document in paper |
| R1-5a: Move lit review to Introduction | Pending | Paper restructure |
| R1-5c: Expand Future Work sections | Pending | Paper edit |
| R2-1: Separate Discussion and Conclusions | Pending | Paper restructure |
| R2-2: State sample selection method | Pending | Paper edit |
| R2-3: Add pipeline diagram | Pending | Create figure |

---

## Files Modified

| File | Modification Type | Description |
|------|------------------|-------------|
| `generate_report.py` | Major refactor | Multi-model support, k-fold CV, timing, error analysis |
| `bioBERT_dual_lifestyle_classifier.ipynb` | Update | Fix bug, add comparison visualizations |
| `RIO_Reviews_Round1/revision_log.md` | Created | This tracking document |

---

## Hyperparameters (Standardized Across All Models)

```python
{
    "num_train_epochs": 20,
    "early_stopping_patience": 3,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 8,
    "weight_decay": 0.05,
    "hidden_dropout_prob": 0.2,
    "attention_probs_dropout_prob": 0.2,
    "metric_for_best_model": "eval_loss",
    "lr_scheduler_type": "linear",
    "random_seed": 42
}
```
