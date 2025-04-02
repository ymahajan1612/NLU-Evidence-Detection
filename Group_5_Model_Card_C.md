
---

# Model Card for r07539ym-w7572sb-ED: Approach C (GROUP 5)

<!-- Provide a quick summary of what the model is/does. -->

## Model Details
This is a transformer-based classification model. Given an evidence and claim, it detects if the evidence supports or refutes the claim

### Model Description

This is a model based on the Bidirectional Encoder Representations from Transformers (BERT). The model has been fine-tuned on 21K claim-evidence pairs as part of the Evidence Detection (ED) task and is intended for the task of pairwise sequence classification. 

To improve the performance, we added dropout regularisation and a fully connected layer on top of BERT. This helped in reducing overfitting to the training data.
<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Yash Mahajan and Shannon Barretto 
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** bert-base-uncased

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

21,509 claim-evidence pairs were used to fine-tune the model

### Training Procedure
The libraries used in model training include: 
- Sklearn 
- pandas
- PyTorch

The training data is loaded from the .CSV file. We use an 80/20 split for training and validation, using the train_test_split functionality of Sklearn.  The purpose of the validation set is to keep track of the model's performance on unseen data. 

We use a custom Pytorch Dataset class, which stores the claim-evidence pairs and their corresponding labels in a format compatible with BERT.

Inside the training loop, the training data is loaded in batches and fed to the model. The model makes predictions the parameters are updated. Immediately after this, the model.eval() command is used to put the model in evaluation model in order to test its performance on the validation split of the data. We keep track of the validation loss as a metric of how well the model generalises to unseen data. 

#### Training Hyperparameters

- Dropout Rate: 0.2
- Learning Rate: 2e-05
- Batch Size: 8
- Epochs: 5
- Optimizer: Adam
- Criterion: Cross Entropy Loss

#### Speeds, Sizes, Times

- Training Time: 10 minutes
- Inference Time: 15 seconds
- Model Size: 417 MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

The 5926 pairs provided in the dev set were used for evaluating the model 

#### Metrics
  - Accuracy
  - Precision (Weighted)
  - Recall (Weighted)
  - F1 Score
  - ROC Curve (AUC)

### Results
The results of the model were as follows
- Accuracy: 0.8766452919338509
- Precision (Weighted): 0.8764849699451085
- Recall (Weighted): 0.8766452919338509
- F1 Score: 0.8765637073041516
- AUC: 0.85

We use a weighted precision and recall as there is a significant class imbalance in the dev set.  72.33% of the examples are labelled 0, whilst 27.67% are labelled 1. Given this class imbalance, using a weighted precision and recall are appropriate.


In general we found that this model performed slightly better than the baseline BERT model on Codabench. This is likely due to the dropout regularisation applied which helps in reducing some of the overfitting. 

## Technical Specifications

### Hardware
We used the A100 GPU available on Google Colab for training

### Software

- Sklearn 
- pandas
- PyTorch



## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
The BERT language model is computationally expensive due to its size. This makes it less accessible to organisations with a limited budget.

BERT supports a maximum sequence length of 512 tokens, meaning that longer texts must be truncated or split. This may lead to a loss of contextual information.

In addition, the model has been trained on a relatively small dataset of claim-evidence pairs. This may limit the model's abiility to generalise across varied contexts. 

