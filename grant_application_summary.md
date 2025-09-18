---
output:
  word_document: default
  html_document: default
---
# Grant Application Summary: BioBERT Fungal Trophic Mode Classifier

## Project Overview

This pilot project successfully demonstrates the application of BioBERT, a state-of-the-art biomedical language model, for automated classification of fungal trophic modes from scientific abstracts. By distinguishing between "dual" (fungi capable of multiple trophic modes) and "solo" (fungi limited to single trophic modes), we achieved 86% classification accuracy, establishing proof-of-concept for scalable text mining in mycology.

## Pilot Study Results

### Performance Metrics
- **Accuracy**: 86.36%
- **Precision**: 86.75%
- **Recall**: 85.83%
- **F1 Score**: 86.11%

### Key Achievements
- Developed end-to-end pipeline from BibTeX parsing to model deployment
- Fine-tuned BioBERT v1.1 on curated fungal literature dataset
- Demonstrated robust performance on held-out test set
- Generated comprehensive evaluation reports and visualizations

## Significance for Mycological Research

This work addresses a critical gap in automated analysis of fungal literature:
- **Scalability**: Current manual classification is time-intensive; this approach enables high-throughput processing
- **Consistency**: Reduces subjective interpretation in trophic mode categorization
- **Accessibility**: Lowers barriers for researchers without NLP expertise
- **Foundation**: Establishes baseline methodology for expanding to multi-class and multi-modal classification

## Pilot Nature and Expansion Potential

As a pilot study, this project validates the technical approach with a modest dataset (~60 abstracts). The methodology is designed for seamless scaling:

### Immediate Next Steps
- Expand dataset to 10x current size using additional literature sources
- Incorporate phylogenetic and ecological metadata
- Develop multi-label classification for complex trophic modes
- Integrate active learning for efficient annotation

### Long-term Vision
- Comprehensive mycological literature mining platform
- Real-time classification of new publications
- Integration with fungal databases (e.g., MycoBank, FUNGuild)
- Support for ecological modeling and biodiversity studies

## Technical Innovation

- **BioBERT Adaptation**: First application of BioBERT specifically for fungal lifestyle classification
- **Pipeline Automation**: Complete workflow from raw data to deployment
- **Domain-Specific Fine-tuning**: Optimized for scientific abstract language
- **Open-Source Implementation**: Reproducible and extensible codebase

## Impact on Grant Goals

This pilot directly supports B. Bock's objectives by:
- Demonstrating innovative application of AI to biological sciences
- Providing concrete evidence of feasibility for larger-scale implementation
- Establishing methodological foundation for systematic literature analysis
- Enabling more efficient research in fungal ecology and taxonomy

## Budget and Timeline (Pilot Phase Completed)

- **Duration**: 3 months
- **Cost**: Primarily computational resources for model training
- **Deliverables**: Complete codebase, trained model, evaluation reports, documentation

## Conclusion

This pilot study provides compelling evidence for the viability of AI-driven fungal lifestyle classification, with immediate applicability and clear pathways for expansion. The 86% accuracy demonstrates strong performance, and the modular design supports rapid scaling to address broader mycological research challenges.

Beatrice Bock 
18 September 2025
bmb646@nau.edu