# BioBERT Dual Lifestyle Classifier

This project implements a pilot experiment using BioBERT (a biomedical language model) to classify scientific abstracts into two categories: "dual" (fungi that can occupy more than one trophic mode, e.g., both saprotrophic and pathogenic) and "solo" (fungi that can only occupy one trophic mode, e.g., strictly saprotrophic or pathogenic).

## Overview

Fungi exhibit diverse lifestyles ranging from solitary saprotrophs to complex symbionts. Accurately classifying research papers on fungal lifestyles is crucial for advancing mycological research, especially in the context of a Saprotrophy-Symbiosis Continuum (Martin and Tan, 2025). This pilot study demonstrates the feasibility of using fine-tuned BioBERT models for automated classification of fungal lifestyle descriptions from scientific abstracts.

The model achieves approximately 86% accuracy on a test set of manually curated abstracts, showing promise for scaling to larger datasets and broader mycological literature.

## Methodology

### Data Collection
- **Source**: Curated BibTeX files containing fungal research papers
- **Categories**:
  - Dual (n=28): Papers describing fungi that can occupy multiple trophic modes
  - Solo (n=28): Papers describing fungi limited to a single trophic mode
- **Total abstracts**: 56 (after preprocessing)

### Preprocessing
- Parsed BibTeX files using `parse_bib.py` to extract titles, abstracts, and keywords
- Cleaned HTML/LaTeX formatting and special characters
- Tokenized abstracts using BioBERT tokenizer (max length 512 tokens)
- Train/test split: 60% training, 40% testing

### Model Fine-tuning
- **Base Model**: `monologg/biobert_v1.1_pubmed` (BioBERT v1.1 trained on PubMed)
- **Task**: Binary sequence classification
- **Training Parameters**:
  - Learning rate: 5e-5
  - Batch size: 8
  - Epochs: 10
  - Optimizer: AdamW with weight decay
- **Hardware**: Trained on CPU/GPU (depending on availability)

### Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Analysis of prediction errors
- **Training Curves**: Loss and accuracy monitoring during fine-tuning

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.8636 |
| Precision | 0.8675 |
| Recall | 0.8583 |
| F1 Score | 0.8611 |

### Key Findings
- The model performs well at distinguishing single vs. multiple trophic mode fungal descriptions
- Highest accuracy on abstracts with clear trophic mode indicators
- Some confusion between borderline cases requiring domain expertise
- Training converged within 10 epochs, indicating efficient learning

### Example Predictions
- **Dual**: "This fungus can switch between saprotrophic decomposition and pathogenic infection depending on environmental conditions."
- **Solo**: "This fungus is strictly saprotrophic, specializing in the decomposition of dead plant material."
- **Dual**: "The endophytic fungus forms mutualistic associations with plants but can become pathogenic under stress."
- **Solo**: "The mycorrhizal fungus exclusively forms symbiotic relationships with plant roots."

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
   ```

2. **Run full pipeline (data loading, training, evaluation, report generation)**:
   ```bash
   python generate_report.py
   ```

3. **Interactive exploration**: Open `bioBERT_dual_lifestyle_classifier.ipynb` in Jupyter

### File Structure
- `datasets/`: Source BibTeX files (dual.bib, solo.bib)
- `parse_bib.py`: Data extraction and preprocessing
- `nlp_prep.py`: Tokenization script
- `generate_report.py`: Full pipeline automation
- `bioBERT_dual_lifestyle_classifier.ipynb`: Interactive notebook
- `train.csv`, `test.csv`: Processed datasets
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