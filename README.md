# NLU-Evidence-Detection

## Overview

The pytorch model files for both approaches can be found at this link: https://drive.google.com/drive/folders/1TWv5UKNsNeQGxafx3GQf87Dc8vcp5V8t?usp=drive_link

Please ensure that the relevant data for training, evaluation, and testing is stored in the `data` directory with the filenames `train.csv`, `dev.csv`, and `test.csv`, respectively.



## Transformer Based Approach

The prediction output from this task can be found in the ``predictions`` directory with filename ``Group_5_C.csv``

### ED_transformer_train.ipynb
This python notebook is responsible for training the transformer based model for evidence detection. Firstly it imports the required modules loads  and splits the training data into ``train_split.csv`` and ``val_split.csv``. 


The `Trainer` class is designed to handle the training and validation process for an evidence detection task. It integrates data loading, model training, validation, and performance tracking. Below is a detailed description of its components and functionality:

#### Constructor: `__init__(self, train_csv, validation_csv, epochs=5)`
Initializes the `Trainer` class with datasets, model, optimizer, and configurations. Key attributes include `train_loader`, `val_loader`, `device`, `model`, `criterion`, `optimizer`, and performance trackers like `best_accuracy`.

#### Method: `plot_loss(self)`
Plots training and validation loss over epochs using `matplotlib`.

#### Method: `trainModel(self)`
Handles training and validation over epochs:
- **Training**: Updates model parameters using backpropagation and tracks training loss.
- **Validation**: Evaluates model performance, calculates accuracy, and tracks validation loss.
- Saves the best model as `ED_model_C.pth` when validation accuracy improves.


### ED_transformer_evaluate.ipynb

This notebook is responsible for evaluating the trained model on the dev set. It generates predictions to the file ``Group_5_C.csv`` and produces the following evaluation metrics:
 - Accuracy
 - Weighted Precision
 - Weighted Recall
 - F1 Score
 - Reciever Operator Curve (ROC) curve


#### Method: `predict(test_csv, model_path)`

The `predict` method is used to generate predictions for a given test dataset using a pre-trained evidence detection model. It loads the model, processes the test data, and outputs predictions along with their associated probabilities. The predictions are appended to the input dataset and returned as a DataFrame for further analysis.

### ED_transformer_demo.ipynb

This python notebook is responsible for evaluating the trained model on the test dataset and it also contains functionality to run the model on user input claim-evidence pairs. 

Similar to ``ED_transformer_evaluate.ipynb``, this notebooks loads the trained model, generates predictions ``Group_5_C.csv`` found in the ``predictions`` directory and produces the same evaluation metrics as the evaluation notebook. 

It uses the same ``predict`` method to generate these predictions.

The notebook contains extra functionality for generating a prediction on  custom user input claim-evidence pairs.



## Deep Learning (w/o Transformers) Approach

The prediction output from this task can be found in the ``predictions`` directory with filename ``Group_5_B.csv``

### train_bilstm_attention.ipynb
This notebook implements and trains a Bidirectional LSTM model with attention mechanism for evidence detection. The model architecture processes both claims and evidence separately, then combines their representations for classification.

**Key components:**
- **Data Preparation**: Loads training data from CSV files and splits it into training and validation sets. The data is preprocessed using a custom vocabulary.
- **Word Embeddings**: Utilizes pre-trained `GloVe embeddings (300d)` to represent tokens semantically.
- **Model Architecture**: 
  - Dual BiLSTM networks to process claims and evidence separately
  - Attention mechanism to focus on the most relevant parts of sequences
  - Concatenation of claim and evidence representations with their element-wise product
  - Dense classification layers
- **Hyperparameter Optimization**: Uses Optuna for hyperparameter tuning to find optimal:
  - Hidden dimension size
  - Dropout rate
  - Batch size
  - Learning rate
  - Weight decay
- **Training Process**:
  - Cross-entropy loss with Adam optimizer
  - Early stopping based on validation F1 score
  - Learning rate scheduling
  - Comprehensive metrics tracking (loss, accuracy, precision, recall, F1)
- **Visualization**: Plots training and validation metrics to track performance over epochs

The notebook leverages several custom modules:
- `AttentionLayer`: Implements the attention mechanism
- `BiLSTMAttention`: The core model architecture
- `EarlyStopping`: For training optimization
- `EvidenceDetectionDataset`: Custom dataset class for data handling
- `Vocabulary`: For tokenization and vocabulary management
- `Trainer`: Orchestrates the training process


### evaluate_bilstm_attention.ipynb

This notebook evaluates the trained BiLSTM with Attention model on the development set. It provides a comprehensive assessment of the model's performance on evidence detection tasks.

**Key components:**
- **Model Loading**: Loads the pretrained BiLSTM with Attention model with optimized hyperparameters.
- **Data Preparation**: Processes the development dataset using the same vocabulary and preprocessing steps used during training.
- **Evaluation Metrics**: Calculates:
  - Accuracy
  - Precision
  - Recall
  - F1 scores
- **Visualization**:
  - Generates ROC curve with AUC score
  - Displays confusion matrix to analyze false positives and false negatives


### demo_bilstm_attention.ipynb

This notebook demonstrates how to use the trained BiLSTM with Attention model for inference on new test data. It serves as a practical demonstration of the model's application for evidence detection tasks.

**Key components:**
- **Environment Setup**: Configures dependencies, sets random seeds, and prepares necessary utilities for reproducible inference.
- **Test Data Loading**: Loads and prepares test data for evaluation.
- **Model Initialization**: Rebuilds the BiLSTM with Attention model architecture using the same hyperparameters from training.
- **Model Loading**: Loads the pretrained model weights from "ED_model_B.pt".
- **Inference Pipeline**: 
  - Processes test data in batches
  - Generates predictions for each claim-evidence pair
  - Exports predictions to "Group_5_B.csv"
- **Production Readiness**: Demonstrates how the model can be deployed for inference on unseen data.

This notebook provides a complete end-to-end workflow for using the trained model in a production environment, enabling users to generate evidence detection predictions on new data.
