![DB](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/98b9a197-b2fa-4934-ba2c-ece36de76357)This repository contains code for a machine-learning project focusing on predicting the location of drone controllers using antenna data. The project encompasses several components:

1 - Data Preparation and Processing: The dataset, loaded from an Excel file, consists of features obtained from four antennas. These features are then split into input (X) and target (y) variables. Label encoding is applied to the target variable to prepare it for modelling.

2 - Model Selection and Training: Various classifiers are employed, including Decision Trees, KNN, Neural Networks, AdaBoost, SVM, Random Forest, and Naive Bayes. Each classifier is trained using GridSearchCV to find the optimal hyperparameters. The classifiers' performances are evaluated using cross-validation scores, accuracy, precision, recall, and F1-score.

3 - Visualization and Analysis: The project includes visualizations such as ROC and Precision-Recall curves for each class, showcasing the models' performance. Decision boundaries are plotted for two-dimensional feature spaces, providing insights into classifier behaviour.

4 - Model Persistence: Trained models are saved using joblib for future use, enabling easy deployment and integration into other applications.

5 - Graphical User Interface (GUI): A simple GUI application is provided for real-time prediction of drone controller locations. Users can input antenna data, and the application displays predictions from various models.

This project comprehensively explores machine learning techniques for drone controller tracking, offering insights into model performance and practical applications. Explore the repository to investigate the code and experiment with different classifiers for drone controller prediction.

![DB](https://github.com/lxotsi/droneControllerTrackingAI/assets/102247398/d4f32114-c231-416f-9df6-fc04a552c1ba)
