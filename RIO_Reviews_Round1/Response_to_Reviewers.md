# Response to Reviewers

**Manuscript:** Automated Extraction of Fungal Trophic Modes from Literature Using BioBERT: An Open Pilot Workflow  
**Authors:** Beatrice M. Bock  
**Journal:** RIO Journal  
**Revision Date:** January 2025

---

We thank the reviewers and editor for their constructive feedback, which has substantially improved the manuscript. Below we provide point-by-point responses to each comment. Reviewer comments are shown in *italics*, and our responses follow.

---

## Response to Reviewer 1 (David Owen)

### Materials and Methods — Dataset Curation

**R1-1a**  
*"It would be helpful, for reproducibility reasons, to know what search terms were used to retrieve the articles from the Web of Science Core Collection, and how many articles the search retrieved."*

**Response:**  
We agree that this information is essential for reproducibility. We have added the full Boolean search queries and result counts to the Dataset Curation section. Specifically, for solo (single trophic mode) examples we searched: `("obligate mycorrhizal" OR "strictly endophytic" OR "exclusive saprotroph") AND fungus` (119 results). For dual (multiple trophic mode) examples we searched: `("dual lifestyle" OR "facultative lifestyle" OR "dual trophic mode" OR "lifestyle switching" OR "endophyte-saprotroph" OR "plant-associated saprotroph") AND fungi` (70 results). We have also created a formal search strategy document in the repository (datasets/search_strat.md) for complete transparency.

---

**R1-1b**  
*"If articles containing abstracts that were longer than 512 characters were discarded due to them being longer than the maximum 512 tokens that BioBERT could consume, then that is of course a valid reason to discard them. If this is the case though, it perhaps ought to be mentioned in this section."*

**Response:**  
We have clarified the token handling in the Preprocessing section. The tokenizer is configured with `truncation=True` and `max_length=512`, which would truncate longer texts. However, we note that abstract lengths in our dataset ranged from 150–500 words (mean 360), and all abstracts fell within the 512 token limit—no information was lost to truncation. We have also added a function to the pipeline that computes and reports token length statistics both before and after truncation, confirming that 0% of abstracts exceeded the limit.

---

**R1-1c**  
*"If not unwieldy, it would be helpful to provide, in a supplementary file, a list of permissible labels (i.e. - trophic modes) that were considered during the labelling exercise."*

**Response:**  
We have created a supplementary file (datasets/trophic_mode_labels.md) that provides complete label definitions. This includes: (1) the binary classification labels (Solo = 0, Dual = 1) with detailed descriptions and example taxa, (2) the underlying trophic mode categories referenced in the literature (saprotroph, symbiont, pathogen, endophyte, parasite, lichenized, mycoparasite), and (3) the selection criteria used for labeling. The manuscript now references this supplementary file.

---

**R1-1d**  
*"It ought to be stated how many people were involved in the labelling exercise. And if more than one person was involved, it ought to be stated how disagreements were resolved."*

**Response:**  
We have added this information to the Dataset Curation section. Abstracts were manually reviewed by a single labeler (BMB). We acknowledge this as a limitation in the revised Discussion section, noting that single-labeler subjectivity may introduce bias, though this was mitigated by conservative inclusion criteria requiring explicit trophic mode statements in the abstract text.

---

### Materials and Methods — Preprocessing and Model Training

**R1-2a**  
*"Only one language model (BioBERT) is used in the experimental setup. It is difficult to gauge BioBERT's performance... At a minimum, I suggest using 'ordinary' BERT (google-bert/bert-base-uncased) as well and reporting its results alongside BioBERT's. If it's also possible to use BiodivBERT... then that would strengthen the study further."*

**Response:**  
We appreciate this excellent suggestion. We have substantially expanded the experimental comparison to include four models: (1) BERT-base-uncased (google-bert/bert-base-uncased), (2) BERT-base-cased (google-bert/bert-base-cased), (3) BioBERT v1.1 (monologg/biobert_v1.1_pubmed), and (4) BiodivBERT (NoYo25/BiodivBERT). All models were trained with identical hyperparameters and evaluated using stratified 5-fold cross-validation for fair comparison.

**Results:** BioBERT achieved the highest performance (F1=0.892±0.120, Accuracy=0.894±0.116), marginally outperforming BERT-base-cased (F1=0.892±0.100, Accuracy=0.892±0.100). Notably, cased models substantially outperformed uncased variants (BERT-base-uncased: F1=0.700±0.241, Accuracy=0.749±0.177), suggesting that taxonomic nomenclature capitalization provides important classification signals. BiodivBERT underperformed (F1=0.747±0.198, Accuracy=0.771±0.166) despite its biodiversity-specific pre-training, indicating that domain alignment alone does not guarantee superior performance on specialized classification tasks. Results are presented with mean ± standard deviation across folds, and comparative visualizations are included. The refactored pipeline (generate_report_v2.py) supports this multi-model comparison.

---

**R1-2b**  
*"It is stated that 10 training epochs were used, but generate_report.py suggests that 20 were used (num_train_epochs=20). If either the manuscript or the python script is incorrect, it ought to be corrected."*

**Response:**  
Thank you for catching this discrepancy. The code is correct: training uses a maximum of 20 epochs with early stopping (patience=3) based on validation loss. The manuscript has been corrected to reflect the actual training configuration. We have also standardized hyperparameters across all models and documented them explicitly in the revision log.

---

**R1-2c**  
*"Since the dataset is small at 56 instances, perhaps using k-fold cross validation would produce more robust results."*

**Response:**  
We agree that k-fold cross-validation is more appropriate for this small dataset. We have implemented stratified 5-fold cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) across all 56 samples. Each model is now evaluated on identical fold splits, and results are reported as mean ± standard deviation. This approach provides more robust performance estimates and allows statistical comparison between models.

---

### Results

**R1-3a**  
*"The Dataset Characteristics subsection is perhaps not best placed in the Results section. It would be preferable if the information in this subsection was appended to the Dataset Curation subsection in the Materials and Methods section."*

**Response:**  
Agreed. We have moved the dataset characteristics (class balance, abstract length statistics) from Results to the Dataset Curation subsection in Materials and Methods. The Results section now focuses solely on model performance.

---

### Model Performance

**R1-4a**  
*"It is unclear how the values in Table 1 for Precision, Recall, and F1-Score were arrived at... It strikes me that the 'average' parameter perhaps ought to be set to 'binary'... If there is a good reason for using 'macro'... then it would be helpful to include an explanation with justification."*

**Response:**  
We have added clarification to the Table 1 caption. We use macro averaging (unweighted mean across both classes) because our dataset is balanced (28 dual, 28 solo), and macro averaging treats both classes equally regardless of support. This is appropriate for balanced binary classification where both classes are of equal importance. For unbalanced datasets, 'binary' or 'weighted' averaging might be preferred, but given our balanced design, macro averaging provides an intuitive and fair summary of model performance.

---

### Discussion

**R1-5a**  
*"This section appears to contain a literature review, covering related work. I feel this would be better placed in the Introduction."*

**Response:**  
We agree that the related work discussion fits better in the Introduction. We have moved this content to a new "Related Work" subsection at the end of the Introduction, covering BiodivBERT, ArTraDB, plant trait extraction, BioT5, ModernBERT, and domain-specific pretraining. The Discussion section now focuses on interpreting results, limitations, and future directions.

---

**R1-5b**  
*"No error analysis is provided. Abstracts that BioBERT misclassified ought to be manually inspected and intuitive suggestions given for why it may have misclassified them."*

**Response:**  
We have added comprehensive error analysis to the pipeline. The system now exports all misclassifications to `figures/error_analysis.csv` with columns for abstract text, true label, predicted label, model, fold, and number of models that misclassified each sample. Samples misclassified by multiple models are flagged for priority manual review, as these likely represent genuinely ambiguous cases. The HTML report includes the top misclassifications, enabling inspection of problematic abstracts across all four models.

---

**R1-5c**  
*"Future Work is mentioned but ought to be explained further, albeit very briefly, in particular: i) Expanding dataset scope... ii) Ecological gradients... iii) Environmental metadata or genomic predictors..."*

**Response:**  
We have substantially expanded the Future Work section with five detailed directions: (1) **Expanding dataset scope** — increasing to hundreds/thousands of abstracts, including full-text articles, and extending to taxa beyond fungi; (2) **Multi-label classification for ecological gradients** — predicting specific trophic modes as non-exclusive labels to capture the continuous nature of ecological roles; (3) **Integration with environmental metadata** — linking text-derived traits with geographic, climatic, or substrate data for context-aware predictions; (4) **Genomic and metabolomic predictors** — combining textual information with molecular data; (5) **Domain-specific pretraining** — training BERT-style models from scratch on ecological/mycological corpora.

---

## Response to Reviewer 2 (Maria Auxiliadora)

**R2-1**  
*"I recommend separating more clearly the Discussion and Conclusions. A concise Conclusions section that highlights the key take home messages and practical implications of the workflow would make the paper easier to read and to cite."*

**Response:**  
We have restructured this section into four distinct parts: (1) **Discussion** — interpretation of results and comparison with existing approaches; (2) **Limitations** — explicit acknowledgment of sample size, binary simplification, and single-labeler constraints; (3) **Future Work** — detailed directions for extension; and (4) **Conclusions** — a concise summary highlighting three key contributions: proof-of-concept validation, reproducible workflow, and trait database integration potential.

---

**R2-2**  
*"There are very few samples, so it is important to explicitly state how these records were selected, for example that they were chosen at random. This helps the reader assess potential selection bias."*

**Response:**  
We have clarified the selection process in the Dataset Curation section. From the 189 candidate articles returned by our searches, 56 were selected based on explicit criteria: (1) unambiguous description of trophic mode in the abstract text, (2) English language, and (3) no duplicates between searches. Abstracts with ambiguous or implied trophic modes were excluded. This was not random selection but rather purposive sampling to ensure clear ground-truth labels for this proof-of-concept study. A formal search strategy document (datasets/search_strat.md) provides complete transparency on the curation process.

---

**R2-3**  
*"If space allows, it would be valuable to include a simple diagram of the pipeline or workflow, from data collection through preprocessing and model training to evaluation."*

**Response:**  
We have created a pipeline workflow diagram (assets/pipeline_diagram.md) that visualizes the complete workflow from data collection through evaluation. The diagram shows four main stages: (1) Data Collection (Web of Science searches → manual review → labeled abstracts), (2) Preprocessing (text cleaning → tokenization → train/test split or k-fold CV), (3) Model Training (BioBERT fine-tuning with hyperparameters), and (4) Evaluation (metrics, confusion matrix, learning curves). A simplified flow is also provided for quick reference.

---

**R2-4**  
*"Given the small dataset size, it would be worth considering a cross validation strategy instead of, or in addition to, a single train test split."*

**Response:**  
We have implemented stratified 5-fold cross-validation as the primary evaluation strategy, replacing the single train/test split. All 56 samples are combined and split into 5 folds with stratification to maintain class balance. Each model is trained and evaluated on all folds, with results reported as mean ± standard deviation. This provides more robust and statistically meaningful performance estimates for our small dataset.

---

**R2-5**  
*"The training curves suggest that a validation set was used during model fine tuning, but in the text you only state that the dataset was split into training 34 abstracts and testing 22 abstracts. Please clarify how the validation data were defined."*

**Response:**  
The move to stratified 5-fold cross-validation eliminates this ambiguity. In each fold, 4/5 of the data (approximately 45 samples) is used for training with the Hugging Face Trainer's internal validation split for early stopping, and 1/5 (approximately 11 samples) is held out as the test set. The learning curves shown reflect the training/validation loss within each fold. This approach ensures that all samples are used for both training and testing across the complete cross-validation procedure.

---

## Response to Editor (R3: Editorial Secretary)

*"The manuscript highlights a methodological opportunity for ecological research. It is generally well-scoped and well-written but there is room for improvement in terms of presentation, clarity and methodological details. Please address the reviewer comments by revising the manuscript accordingly, and then respond to the comments."*

**Response:**  
We thank both reviewers for their constructive feedback, which has substantially strengthened the manuscript. The major revisions include:

**Methodological improvements:**
- Expanded from single-model to four-model comparison (BERT-uncased, BERT-cased, BioBERT, BiodivBERT) per R1-2a
- Implemented stratified 5-fold cross-validation for more robust evaluation per R1-2c and R2-4
- Added comprehensive error analysis with multi-model misclassification flagging per R1-5b
- Corrected the epochs discrepancy and standardized hyperparameters per R1-2b

**Reproducibility and transparency:**
- Added full search terms and result counts to Methods per R1-1a
- Clarified token handling and sample selection criteria per R1-1b and R2-2
- Created supplementary files for trophic mode labels and search strategy per R1-1c
- Created a pipeline workflow diagram per R2-3

**Presentation and structure:**
- Moved Dataset Characteristics to Methods per R1-3a
- Moved literature review to Introduction as "Related Work" per R1-5a
- Separated Discussion and Conclusions with expanded Limitations and Future Work per R2-1 and R1-5c
- Added macro averaging explanation to Table 1 per R1-4a

All code, data, and supplementary materials remain openly available in the repository.

---

## Summary of Changes

| Change | Reviewer Comment(s) |
|--------|---------------------|
| Added four-model comparison (BERT-uncased, BERT-cased, BioBERT, BiodivBERT) | R1-2a |
| Implemented stratified 5-fold cross-validation | R1-2c, R2-4 |
| Added search terms and result counts to Methods | R1-1a |
| Clarified token handling (512 max, all abstracts within limit) | R1-1b |
| Created supplementary trophic mode label definitions | R1-1c |
| Noted single labeler (BMB) in Methods | R1-1d |
| Corrected epochs discrepancy (20 max with early stopping) | R1-2b |
| Moved Dataset Characteristics to Methods | R1-3a |
| Added macro averaging explanation to Table 1 | R1-4a |
| Moved literature review to Introduction as "Related Work" | R1-5a |
| Added error analysis with multi-model misclassification flagging | R1-5b |
| Expanded Future Work with 5 detailed directions | R1-5c |
| Separated Discussion and Conclusions sections | R2-1 |
| Documented selection criteria for abstracts | R2-2 |
| Created pipeline workflow diagram | R2-3 |
| Clarified validation approach (k-fold eliminates ambiguity) | R2-5 |

---

## Files Added/Modified

- `generate_report_v2.py` — Refactored pipeline with 4-model comparison, 5-fold CV
- `datasets/trophic_mode_labels.md` — Supplementary label definitions
- `datasets/search_strat.md` — Search strategy documentation
- `assets/pipeline_diagram.md` — Workflow diagram
- `Bock_RIO_2025_Biobert.txt` — Restructured manuscript
