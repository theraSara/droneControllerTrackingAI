# Drone Controller Tracking AI

This repository contains code for a **machine-learning project** focusing on predicting the _location of drone controllers using antenna data_. The project encompasses several key components, including data preparation, model training, evaluation, visualization, and a GUI for real-time predictions.

---

## Project Features

### 1. Data Preparation and Processing
- The dataset, loaded from an Excel file, consists of features obtained from **four antennas**.
- Input (X) and target (y) variables are extracted, with label encoding applied to the target variable for modelling.
- The data is preprocessed and split into training and testing sets for evaluation.

### 2. Model Selection and Training
- Multiple classifiers are utilized:
  - **Decision Trees**
  - **KNN**
  - **Neural Networks**
  - **AdaBoost**
  - **SVM**
  - **Random Forest**
  - **Naive Bayes**
- **GridSearchCV** is applied to optimize hyperparameters.
- Classifier performance is evaluated using:
  - Cross-validation scores
  - Accuracy
  - Precision
  - Recall
  - F1-score

### 3. Visualization and Analysis
- Key visualizations include:
  - **ROC and Precision-Recall Curves:** Displaying models' performance for each class.
  - **Decision Boundaries:** Illustrating model behaviour in two-dimensional feature spaces.
- Visualizations provide insights into classifier performance and decision-making capabilities.

#### Cross-Validation
![Cross-Validation](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/b2b5fddc-f3f9-46ab-a159-c2d5a3f32e34)

#### Performance Metrics
![Performance Metrics 1](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/28e6e238-7bda-43c1-a4cc-95fd4ff020ac)
![Performance Metrics 2](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/37893859-c17a-4da9-8109-8879a2f7282c)

#### ROC and Precision-Recall Curves
![Curves](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/2d6cc7bf-c0d2-4eba-b9a2-7a99aeceba93)

#### Decision Boundaries
![Decision Boundaries](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/d4f32114-c231-416f-9df6-fc04a552c1ba)

### 4. Model Persistence
- Trained models are saved using **Joblib** for future use.
- This enables easy deployment and integration into other applications.

### 5. Graphical User Interface (GUI)
- A simple **GUI application** allows real-time predictions of drone controller locations.
- Users can input antenna data, and the application provides predictions from various models.
