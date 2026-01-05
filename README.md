
# BioBERT Dual Lifestyle Classifier

This project implements a pilot workflow for automated classification of fungal trophic modes from scientific abstracts using transformer-based language models. The pipeline now supports four models (BioBERT, BERT-base-cased, BERT-base-uncased, BiodivBERT) and stratified 5-fold cross-validation, with robust error analysis and reproducibility features.

## Overview

Fungi exhibit diverse lifestyles, from solitary saprotrophs to complex symbionts. Accurately classifying fungal lifestyles is crucial for ecological research and trait database development. This project demonstrates that fine-tuned transformer models can automate this process, with BioBERT and BERT-cased achieving ~89% accuracy in cross-validation.

## Methodology

### Data Collection
- **Source**: Curated abstracts from Web of Science using documented Boolean queries (see `datasets/search_strat.md`)
- **Categories**:
  - Dual (n=28): Fungi occupying multiple trophic modes
  - Solo (n=28): Fungi restricted to a single trophic mode
- **Total abstracts**: 56 (after curation and deduplication)

### Preprocessing
- Parsed and cleaned abstracts using `parse_bib.py` and `nlp_prep.py`
- Tokenized with model-specific tokenizers (max length 512 tokens)
- Token length statistics and truncation analysis included for reproducibility

### Model Training & Evaluation
- **Models compared**:
  - `monologg/biobert_v1.1_pubmed` (BioBERT)
  - `google-bert/bert-base-cased`
  - `google-bert/bert-base-uncased`
  - `NoYo25/BiodivBERT`
- **Cross-validation**: Stratified 5-fold CV (all samples used for both training and testing)
- **Hyperparameters**: Standardized across all models (20 epochs max, early stopping, batch size 8, learning rate 5e-5)
- **Scripts**: Use `generate_report_v2.py` for full pipeline, including multi-model comparison, error analysis, and visualizations

### Results

| Model               | F1 Score         | Accuracy         | Training Time |
|---------------------|------------------|------------------|--------------|
| **BioBERT**         | **0.892 ± 0.120**| **0.894 ± 0.116**| 10.3 min      |
| BERT-base-cased     | 0.892 ± 0.100    | 0.892 ± 0.100    | 11.1 min      |
| BiodivBERT          | 0.747 ± 0.198    | 0.771 ± 0.166    | 35.3 min      |
| BERT-base-uncased   | 0.700 ± 0.241    | 0.749 ± 0.177    | 35.3 min      |

#### Key Findings
- BioBERT and BERT-cased perform equally well; case sensitivity is important for taxonomic text
- BiodivBERT and uncased BERT underperform, showing domain adaptation alone is not sufficient
- Error analysis and confusion matrices are included for transparency

## Usage

### Prerequisites
- Python 3.8+
- Required packages: `transformers`, `datasets`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
1. **Parse and prepare data**:
   ```bash
   python parse_bib.py
   python nlp_prep.py
   ```

2. **Run full pipeline (multi-model CV, evaluation, report generation)**:
   ```bash
   python generate_report_v2.py
   ```

3. **Interactive exploration**: Open `bioBERT_dual_lifestyle_classifier.ipynb` in Jupyter for step-by-step analysis and visualization

### Monitoring & Outputs
- Training curves, model comparison plots, confusion matrices, and error analysis are saved in `figures/`
- All summary metrics and per-fold predictions are in `results/`
- Supplementary files: `datasets/trophic_mode_labels.md`, `datasets/search_strat.md`, `assets/pipeline_diagram.md`

## File Structure
- `datasets/`: Labeled abstracts, search strategy, and label definitions
- `parse_bib.py`, `nlp_prep.py`: Data extraction and tokenization
- `generate_report_v2.py`: Main pipeline for multi-model CV and reporting
- `bioBERT_dual_lifestyle_classifier.ipynb`: Interactive notebook
- `results/`: Metrics, predictions, and token stats
- `figures/`: Training curves, model comparison, confusion matrices, error analysis
- `project_report.html`, `project_report_comparison.html`: Generated reports

## Limitations and Future Work

This pilot study demonstrates proof-of-concept but has several limitations:
- Small dataset size (56 abstracts)
- Manual curation required for training data
- Binary classification (solo/dual) oversimplifies ecological reality
- Domain-specific performance may vary

Future directions include:
- Expanding to larger, more diverse datasets
- Multi-label classification for ecological gradients
- Integration with environmental metadata and genomic data
- Active learning for efficient annotation
- Further enhancements to monitoring and reporting

## Citations

For grant applications, cite this work as:

Beatrice Bock. (2025). *BioBERT Dual Lifestyle Classifier: A Pilot Study for Automated Classification of Fungal Lifestyles from Scientific Abstracts* [Computer software]. Available at: https://github.com/beabock/biobert_dualsolo

### References
- Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers).* *arXiv preprint arXiv:1810.04805*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Beatrice Bock
bmb646@nau.edu

### File Structure
- `datasets/`: Source BibTeX files (dual.bib, solo.bib)
- `parse_bib.py`: Data extraction and preprocessing
- `nlp_prep.py`: Tokenization script
- `generate_report.py`: Full pipeline automation with real-time monitoring
- `bioBERT_dual_lifestyle_classifier.ipynb`: Interactive notebook
- `train.csv`, `test.csv`: Processed datasets
- `figures/`: Training curves and evaluation plots
- `project_report.html`: Generated evaluation report

## Limitations and Future Work

This pilot study demonstrates proof-of-concept but has several limitations:
- Small dataset size (58 abstracts total)
- Manual curation required for training data
- Domain-specific performance may vary
- No multi-label classification for complex lifestyles

Future directions include:
- Expanding to larger, more diverse datasets
- Incorporating multi-modal data (images, phylogenetic trees)
- Developing active learning approaches for efficient annotation
- Exploring other transformer architectures optimized for scientific text
- Further enhancements to real-time monitoring and automated early stopping

## Citations

For grant applications, cite this work as:

Beatrice Bock. (2025). *BioBERT Dual Lifestyle Classifier: A Pilot Study for Automated Classification of Fungal Lifestyles from Scientific Abstracts with Real-time Training Monitoring* [Computer software]. Available at: https://github.com/beabock/biobert_dualsolo

### References
- Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers).* *arXiv preprint arXiv:1810.04805*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Beatrice Bock
bmb646@nau.edu