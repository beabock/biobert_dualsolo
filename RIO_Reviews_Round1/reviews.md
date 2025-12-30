R1: David Owen

The manuscript describes a language model-driven machine learning classification task. The aim of the task is to distinguish between article abstracts that refer to single or multiple fungal trophic modes. In machine learning terms, this is framed as a binary classification task.

The manuscript's Abstract and Introduction provide good motivation for the study. I have a number of suggestions to help strengthen the manuscript and the findings reported in it.

Materials and MethodsDataset Curation

1)
a) It would be helpful, for reproducibility reasons, to know what search terms were used to retrieve the articles from the Web of Science Core Collection, and how many articles the search retrieved.

b) If articles containing abstracts that were longer than 512 characters were discarded due to them being longer than the maximum 512 tokens that BioBERT could consume, then that is of course a valid reason to discard them. If this is the case though, it perhaps ought to be mentioned in this section.

c) If not unwieldy, it would be helpful to provide, in a supplementary file, a list of permissible labels (i.e. - trophic modes) that were considered during the labelling exercise.

d) It ought to be stated how many people were involved in the labelling exercise. And if more than one person was involved, it ought to be stated how disagreements were resolved (i.e. - how the final label for each abstract was finally agreed).

Preprocessing and Model Training

2)
a) Only one language model (BioBERT) is used in the experimental setup. It is difficult to gauge BioBERT's performance (thus its suitability for this binary classification task) when there are no results using other language models that it can be compared with. At a minimum, I suggest using "ordinary" BERT (google-bert/bert-base-uncased) as well and reporting its results alongside BioBERT's. This would provide a crucial baseline against which to compare the performance of BioBERT. If it's also possible to use BiodivBERT (which is mentioned in the Discussion section) then that would strengthen the study further.

b) It is stated that 10 training epochs were used, but generate_report.py suggests that 20 were used (num_train_epochs=20). If either the manuscript or the python script is incorrect, it ought to be corrected.

c) Since the dataset is small at 56 instances, perhaps using k-fold cross validation would produce more robust results. If not to be done in this study, I suggest it be mentioned as a possibility in Future Work.

Results

3)

a) The Dataset Characteristics subsection is perhaps not best placed in the Results section. It would be preferable if the information in this subsection was appended to the Dataset Curation subsection in the Materials and Methods section.

Model Performance

4)a) It is unclear how the values in Table 1 for Precision, Recall, and F1-Score were arrived at. Using the confusion matrix (Figure 1), I am only able to arrive at the following values: Precision 100%, Recall 70%, F1-Score 82.4%. In generate_report.py, it appears that the macro average of these values was computed, which may provide an explanation. It strikes me that the "average" parameter perhaps ought to be set to "binary" (which is the default) as it would be apt for this binary classification exercise. The "macro" setting is perhaps more suited to multi-class classification exercises. If there is a good reason for using "macro" (or these results have been arrived at in some other way) then it would be helpful to include an explanation with justification.

Discussion

5)

a) This section appears to contain a literature review, covering related work. I feel this would be better placed in the Introduction.

b) No error analysis is provided. Abstracts that BioBERT misclassified ought to be manually inspected and intuitive suggestions given for why it may have misclassified them. The same ought to be done for other classifiers that I have suggested should be used (i.e. - BERT, at least) and suggestions for why, possibly, BERT may classify some abstracts correctly, say, that BioBERT does not.

c) Future Work is mentioned but ought to be explained further, albeit very briefly, in particular:

i) Expanding dataset scope - in what way? Including longer abstracts perhaps?

ii) Ecological gradients - what labels might be assigned with respect to these and why might they be useful?

iii) Environmental metadata or genomic predictors - again, what labels might be worth considering and why?


R2: Maria Auxiliadora

The manuscript presents a reproducible text mining workflow to extract trophic mode information from literature. Overall, the manuscript is well written and logically structured. The methods are relevant and adequately described, and the results support the main conclusions. The author follows open science practices by sharing data and code, and by documenting their workflow so that it can be reused and extended.

I particularly appreciate the following strengths:

- The focus on an open and reproducible NLP workflow using BioBERT for a concrete ecological problem, namely fungal trophic modes.

- The availability of data and code in standard, well documented formats that facilitate reuse.

- The automatic extraction of this information contributes directly to the structuring and standardization of biodiversity data. It helps transform scattered information in the literature into machine readable formats that can be integrated into trait databases and biodiversity information systems.

I have a few suggestions that could help to improve the document:

- I recommend separating more clearly the Discussion and Conclusions. A concise Conclusions section that highlights the key take home messages and practical implications of the workflow would make the paper easier to read and to cite.

- There are very few samples, so it is important to explicitly state how these records were selected, for example that they were chosen at random. This helps the reader assess potential selection bias.

- If space allows, it would be valuable to include a simple diagram of the pipeline or workflow, from data collection through preprocessing and model training to evaluation. A visual summary would make the methodology easier to follow.

- Given the small dataset size, it would be worth considering a cross validation strategy instead of, or in addition to, a single train test split. This could provide more stable and reliable estimates of model performance.

- The training curves suggest that a validation set was used during model fine
tuning, but in the text you only state that the dataset was split into training 34 abstracts and testing 22 abstracts. Please clarify how the validation data were defined and whether they overlap with the training or test sets.


R3: Editorial Secretary

The manuscript highlights a methodological opportunity for ecological research. It is generally well-scoped and well-written but there is room for improvement in terms of presentation, clarity and methodological details. Please address the reviewer comments by revising the manuscript accordingly, and then respond to the comments.