
---

# Model Card for r07539ym-w7572sb-ED: Approach B (GROUP 5)

<!-- Provide a quick summary of what the model is/does. -->

## Model Details
This is a Bidirectional LSTM with Attention classification model for evidence detection. Given an evidence and claim pair, it determines if the evidence supports or refutes the claim.

### Model Description

This is a model based on the Bidirectional Long Short-Term Memory (BiLSTM) with Attention. The model has been trained on 21K claim-evidence pairs as part of the Evidence Detection (ED) task and is intended for the task of pairwise sequence classification.

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Yash Mahajan and Shannon Barretto 
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Bidirectional LSTM with Attention
- **Finetuned from model [optional]:** N/A (trained from scratch)

### Model Resources

<!-- Provide links where applicable. -->

- **Pre-trained embeddings:** GloVe 6B 300d word embeddings  https://nlp.stanford.edu/data/wordvecs/glove.6B.zip
- **Attention mechanism:** Based on attention mechanisms described in "Attention Is All You Need" (Vaswani et al., 2017), https://arxiv.org/pdf/1706.03762. Note that our implementation adapts the attention concept for use with BiLSTM architecture rather than employing the full Transformer architecture described in the paper.

## Training Details

### Training Data

21,509 claim-evidence pairs were used to train the model

### Training Procedure
The libraries used in model training include: 
- Sklearn 
- pandas
- numpy
- PyTorch
- Optuna for hyperparameter optimization
- NLTK for tokenization

The training data is loaded from the .CSV file. We use an 80/20 split for training and validation, using the train_test_split functionality of Sklearn.  The purpose of the validation set is to keep track of the model's performance on unseen data. 

We implemented a custom PyTorch Dataset class to handle the claim-evidence pairs. The text was tokenized using NLTK's word_tokenize function and converted to numerical representations using a vocabulary built from the training data with a frequency threshold of 3.

The model architecture consists of:
1. An embedding layer initialized with pre-trained GloVe embeddings
2. Separate BiLSTM encoders for claims and evidences
3. Attention mechanisms to focus on important parts of both claim and evidence
4. A classification head that combines the attended representations

For optimization, we used the Adam optimizer with cross-entropy loss. We implemented early stopping based on validation F1 score with a patience of 3 epochs. The model was trained for a maximum of 15 epochs.


#### Training Hyperparameters

- Dropout Rate: 0.4277148867646843
- Learning Rate: 0.00033550020021625327
- Batch Size: 16
- Hidden Dimension: 256
- Weight Decay: 1.8390870324768507e-05

#### Speeds, Sizes, Times

- Hyper parameter tuning time: 33 minutes
- Training Time: 3 minutes
- Inference Time: 3 seconds
- Model Size: 18.6 MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

The 5926 pairs provided in the dev set were used for evaluating the model 

#### Metrics
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve (AUC)

### Results
The results of the model were as follows
- Accuracy: 0.8176
- Weighted Precision: 0.8141
- Weighted Recall: 0.8176
- Weighted F1 Score: 0.8155
- AUC: 0.76

We use a weighted precision, recall and F1 score as there is a significant class imbalance in the dev set. 72.33% of the examples are labelled 0, whilst 27.67% are labelled 1. Given this class imbalance, using a weighted precision, recall and F1 score are appropriate.

In general we found that this model performed slightly better than the baseline LSTM model on Codabench. This is likely due to the dropout regularisation applied which helps in reducing some of the overfitting.


## Technical Specifications

### Hardware
We used the A100 GPU available on Google Colab for training

### Software

- Sklearn 
- pandas
- PyTorch
- NLTK
- numpy
- tqdm (for progress tracking)
- Optuna (for hyperparameter optimization)



## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
This model has several technical limitations:

- **Sequence Length Limitation**: The BiLSTM architecture may struggle with very long sequences due to the vanishing gradient problem. We set a maximum sequence length of 100 tokens.

- **Fixed Vocabulary**: Words not seen during training or below the frequency threshold are treated as unknown tokens, which may limit performance on texts with specialized vocabulary.

- **Context Understanding**: The attention mechanism helps capture important parts of the text, but may not model complex hierarchical relationships as effectively as transformer architectures.

- **Domain Sensitivity**: The model is trained on specific claim-evidence pairs and may not generalize well to significantly different domains or writing styles.
