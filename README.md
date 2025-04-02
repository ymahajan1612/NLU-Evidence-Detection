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