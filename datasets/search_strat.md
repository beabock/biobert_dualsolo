# Search Strategy Documentation

**Date:** September 11, 2025  
**Database:** Web of Science Core Collection  
**Purpose:** Curate training dataset for BioBERT trophic mode classification

---

## Search Queries

### Solo Trophic Mode (single lifestyle)
```
("obligate mycorrhizal" OR "strictly endophytic" OR "exclusive saprotroph") AND fungus
```
**Results:** 119 articles

### Dual Trophic Mode (multiple lifestyles)
```
("dual lifestyle" OR "facultative lifestyle" OR "dual trophic mode" OR "lifestyle switching" OR "endophyte-saprotroph" OR "plant-associated saprotroph") AND fungi
```
**Results:** 70 articles

**Total candidate articles:** 189

---

## Selection Criteria

From the 189 candidate articles, abstracts were manually reviewed and 56 were selected based on:

1. **Unambiguous trophic mode description** — Abstract explicitly states trophic mode classification
2. **English language** — Only English-language abstracts included
3. **No duplicates** — Manually verified no overlap between solo and dual searches

Abstracts without explicit trophic mode statements were excluded.

---

## Curation Process

1. Selected articles organized in **Zotero** by class (solo folder, dual folder)
2. Exported as BibTeX files (`solo.bib`, `dual.bib`)
3. Processed with `parse_bib.py` to extract abstracts and create `abstracts.csv`
4. Evaluated using stratified 5-fold cross-validation (rather than single train/test split) to maximize statistical robustness with small dataset

---

## Final Dataset

| Class | Count | Description |
|-------|-------|-------------|
| Solo (0) | 28 | Single trophic mode only |
| Dual (1) | 28 | Multiple trophic modes |
| **Total** | **56** | |

---

## Model Training & Evaluation

**Models Compared (5-fold CV):**
1. BERT-base-uncased (`google-bert/bert-base-uncased`)
2. BERT-base-cased (`google-bert/bert-base-cased`)
3. BioBERT v1.1 (`monologg/biobert_v1.1_pubmed`)
4. BiodivBERT (`NoYo25/BiodivBERT`)

**Best Performance:**
- BioBERT and BERT-cased: ~89% accuracy (F1 = 0.892)
- Key finding: Case sensitivity matters more than domain-specific pre-training

**Evaluation Approach:**
- Stratified 5-fold cross-validation (seed=42)
- All 56 samples used for both training and testing across folds
- Results reported as mean ± standard deviation

---

## Additional Resources Consulted

- **FunGuild** (funguild.org) — Trophic mode definitions (saprotroph, symbiotroph, pathotroph)

---
