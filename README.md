# Breast Cancer Detection using Logistic Regression

This repository contains a Jupyter Notebook for detecting breast cancer using a logistic regression model. The dataset used is from the UC Irvine Machine Learning Repository.

## Overview

Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for effective treatment and better prognosis. This project demonstrates the use of logistic regression, a popular machine learning algorithm, to classify breast cancer cases based on various features.

## Data

The dataset used in this project is `breast_cancer.csv`, which contains features extracted from digitized images of breast mass and their corresponding labels (benign or malignant).

## Features

- The dataset includes several features extracted from breast mass images.
- The features are scaled values representing various characteristics of the cell nuclei present in the image.

## Model

- **Logistic Regression**: A linear model used for binary classification tasks. It predicts the probability that a given input belongs to a certain class.

## Steps

1. **Importing Libraries**: Importing necessary libraries for data manipulation, model training, and evaluation.
2. **Loading Data**: Reading the dataset and extracting features (X) and target (y) values.
3. **Data Splitting**: Splitting the dataset into training and testing sets using an 80-20 split.
4. **Model Training**: Training the logistic regression model on the training set.
5. **Prediction**: Predicting the labels of the test set.
6. **Evaluation**: Evaluating the model using a confusion matrix and k-fold cross-validation.

## Results

### Confusion Matrix

```
[[84  3]
 [ 3 47]]
```

- **True Negatives (TN)**: 84
- **True Positives (TP)**: 47
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 3

### Accuracy Metrics

- **Accuracy**: 95.6%
- **Precision**: 94%
- **Recall (Sensitivity)**: 94%
- **F1 Score**: 94%

### Cross-Validation

- **Mean Accuracy**: 96.70%
- **Standard Deviation**: 1.97%

## Conclusion

The logistic regression model for breast cancer detection demonstrates high accuracy and stability, making it a reliable tool for predicting breast cancer based on the given dataset. The model's performance is evaluated using a confusion matrix and k-fold cross-validation, both of which show promising results.

## Repository Contents

- `breast_cancer.csv`: The dataset used for the project.
- `Breast_Cancer_Detection_Logistic_Regression.ipynb`: The Jupyter Notebook containing the code and explanations.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/sumansatyal/breast-cancer-detection-logistic-regression.git
   cd breast-cancer-detection-logistic-regression
   ```
2. Ensure you have the necessary Python libraries installed:
   ```bash
   pip install pandas scikit-learn
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Breast_Cancer_Detection_Logistic_Regression.ipynb
   ```

## Acknowledgments

- The dataset is taken from the UC Irvine Machine Learning Repository.
- Special thanks to the contributors and maintainers of the dataset.
