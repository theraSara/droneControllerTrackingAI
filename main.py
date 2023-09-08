import time
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, precision_recall_curve

pd.options.mode.chained_assignment = None
data = pd.read_excel(’Data.xlsx’)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = [ 
  (’Decision Tree’, DecisionTreeClassifier()),
  (’KNN’, KNeighborsClassifier()),
  (’Neural Network’, MLPClassifier()),
  (’AdaBoost’, AdaBoostClassifier()),
  (’SVM’, SVC(probability=True)), # New SVM model
  (’Random Forest’, RandomForestClassifier()), # New Random Forest model
  (’Naive Bayes’, GaussianNB())
]

def plot_decision_boundary(classifier, X, y, title):
  h = 0.02 # Step size in the mesh
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  
  Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  plt.contourf(xx, yy, Z, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors=’k’, s=20)
  plt.title(title)
  plt.xlabel(’Feature 1’)
  plt.ylabel(’Feature 2’)
  plt.show()

tuned_classifiers = []
for name, classifier in classifiers:
  params = {}
  if name == ’Decision Tree’:
    params = {
      ’criterion’: [’gini’, ’entropy’], 
      ’max_depth’: [None, 10, 20, 30]
    }
  elif name == ’KNN’:
    params = {
      ’n_neighbors’: [3, 5, 7]
    }
  elif name == ’Neural Network’:
    params = {
      ’hidden_layer_sizes’: [(50,), (100,), (50, 50)], 
      ’alpha’: [0.0001, 0.001, 0.01]
    }
  elif name == ’AdaBoost’:
    params = {
      ’n_estimators’: [50, 100, 200], 
      ’learning_rate’: [0.01, 0.1, 1.0]
    }
  elif name == ’SVM’:
    params = {
      ’C’: [0.1, 1, 10], 
      ’kernel’: [’linear’, ’rbf’]
    }
  elif name == ’Random Forest’:
    params = {
      ’n_estimators’: [50, 100, 200], 
      ’max_depth’: [None, 10, 20]
    }
  elif name == ’Naive Bayes’:
    params = {
      ’var_smoothing’: [1e-09, 1e-08, 1e-07]
    }

  # Record the start time
  start_time = time.time()
  
  grid_search = GridSearchCV(classifier, params, cv=5, scoring=’accuracy’, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  tuned_classifiers.append(grid_search.best_estimator_)
  
  # Record the end time
  end_time = time.time()
  
  print(f"Best parameters for {name}: {grid_search.best_params_}")
  print(f"Training time for {name}: {end_time - start_time:.2f} seconds")


for name, classifier in zip(classifiers, tuned_classifiers):
  cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
  
  start_time = time.time()
  
  print(f"{name} Cross-Validation Accuracy: {np.mean(cv_scores)}")
  
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)

  end_time = time.time()

  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average=’weighted’)
  recall = recall_score(y_test, y_pred, average=’weighted’)
  f1 = f1_score(y_test, y_pred, average=’weighted’)
  
  print(f"{name} Test Accuracy: {accuracy}")
  print(f"{name} Test Precision: {precision}")
  print(f"{name} Test Recall: {recall}")
  print(f"{name} Test F1 Score: {f1}")
  print(f"{name} Test Time: {end_time - start_time:.4f} seconds")
  
  conf_matrix = confusion_matrix(y_test, y_pred)
  print(f"{name} Confusion Matrix:")
  print(conf_matrix)
  y_score = classifier.predict_proba(X_test)
  n_classes = len(np.unique(y))
  # False-Positive Rate
  fpr = dict()
  #True-Positive Rate
  tpr = dict()
  # Area Under (ROC) Curve
  roc_auc = dict()
  # Precision-Recall Curve
  prc = dict()
  recalls = dict()
  precisions = dict()
  
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[I])
    precision, recall, _ = precision_recall_curve(y_test, y_score[:, i], pos_label=i)
    prc[i] = auc(recall, precision)
    recalls[i] = recall
    precisions[i] = precision
    
  lw = 2
  n_rows = (n_classes + 1) // 2
  n_cols = 2
  fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
  axs = axs.flatten()

  # Plot Outputs
  for i in range(n_classes):
    row = i // n_cols
    col = i % n_cols
    ax = axs[I]
    ax2 = axs[I]
    ax.plot(fpr[i], tpr[i], lw=2, label=f’ROC curve (area = {roc_auc[i]:.2f}) for class {i}’)
    ax2.plot(recalls[i], precisions[i], lw=2, label=f’PRC curve (area = {prc[i]:.2f}) for class {i}’)
    ax.plot([0, 1], [0, 1], color=’gray’, lw=2, linestyle=’--’)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(’False Positive Rate’)
    ax.set_ylabel(’True Positive Rate’)
    ax.set_title(f’ROC Curve - Class {i}’)
    ax.legend(loc="lower right")
    ax2.plot([0, 1], [0, 1], color=’gray’, lw=2, linestyle=’--’)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel(’False Positive Rate’)
    ax2.set_ylabel(’True Positive Rate’)
    ax2.set_title(f’Curve - Class {i}’)
    ax2.legend(loc="lower right")
  
  if n_classes % 2 != 0:
    fig.delaxes(axs[n_classes])

  plt.tight_layout()
  plt.show()

X, y = make_classification(n_samples=300,n_features=4, n_informative=2, n_redundant=0, random_state=42)
X_visualize = X[:, :2]

for name, classifier in classifiers:
  classifier.fit(X_visualize, y)
  plot_decision_boundary(classifier,X_visualize, y, "Decision Boundary - "+ name)

for name, classifier in zip(classifiers, tuned_classifiers):
  model_filename = f’models/{name[0]}_model.pkl’joblib.dump(classifier, model_filename)
  print(f"Saved {name[0]} model to {model_filename}")


class DroneControllerGUI:
  def __init__(self, root):
    self.root = root
    self.root.title("Drone Controller Location Prediction")
    self.root.geometry("400x400")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = (screen_width - 400) // 2
    y_position = (screen_height - 400) // 2
    self.root.geometry(f"+{x_position}+{y_position}")
    self.model_names = [
      ’Decision Tree’, 
      ’KNN’, 
      ’Neural Network’, 
      ’AdaBoost’,
      ’SVM’, 
      ’Random Forest’,
      ’Naive Bayes’
    ]
    
    self.models = self.load_models()
    self.style = ttk.Style()
    self.style.configure(’TLabel’, font=(’Helvetica’, 12))
    self.style.configure(’TEntry’, font=(’Helvetica’, 12))
    self.style.configure(’TButton’, font=(’Helvetica’, 12), fg=’white’, bg=’brown’)
    self.label_antenna1 = ttk.Label(root,text="Antenna 1:")
    self.label_antenna1.grid(row=0, column=0, padx=10, pady=5, sticky=’e’)
    self.entry_antenna1 = ttk.Entry(root)
    self.entry_antenna1.grid(row=0, column=1, padx=10, pady=5)
    self.label_antenna2 = ttk.Label(root,text="Antenna 2:")
    self.label_antenna2.grid(row=1, column=0, padx=10, pady=5, sticky=’e’)
    self.entry_antenna2 = ttk.Entry(root)
    self.entry_antenna2.grid(row=1, column=1, padx=10, pady=5)
    self.label_antenna3 = ttk.Label(root,text="Antenna 3:")
    self.label_antenna3.grid(row=2, column=0, padx=10, pady=5, sticky=’e’)
    self.entry_antenna3 = ttk.Entry(root)
    self.entry_antenna3.grid(row=2, column=1, padx=10, pady=5)
    self.label_antenna4 = ttk.Label(root,text="Antenna 4:")
    self.label_antenna4.grid(row=3, column=0, padx=10, pady=5, sticky=’e’)
    self.entry_antenna4 = ttk.Entry(root)
    self.entry_antenna4.grid(row=3, column=1, padx=10, pady=5)
    self.predict_button = ttk.Button(root,text="Predict Location", command=
    self.predict_location)
    self.predict_button.grid(row=4, column=1, padx=10, pady=10)
    self.result_label = ttk.Label(root, text="", width=25)
    self.result_label.grid(row=5, column=1,padx=10, pady=5)
    
def plot_decision_boundary(self, X, y, model, name, ax):
  # Step size in the mesh
  h = .02
  x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min,x_max, h), np.arange(y_min, y_max, h))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
  ax.scatter(X[:, 0], X[:, 1], c=y,edgecolors=’k’, cmap=plt.cm.coolwarm)
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xlabel(’Feature 1’)
  ax.set_ylabel(’Feature 2’)
  ax.set_title(f’Decision Boundaries - {name}’)

def load_models(self):
  models = {}
  for model_name in self.model_names:
    model_filename = f’models/{model_name}_model.pkl’
    model = joblib.load(model_filename)
    models[model_name] = model
    return models

def predict_location(self):
  antenna1 = float(self.entry_antenna1.get())
  antenna2 = float(self.entry_antenna2.get())
  antenna3 = float(self.entry_antenna3.get())
  antenna4 = float(self.entry_antenna4.get())
  input_data = np.array([antenna1,antenna2, antenna3, antenna4]).reshape(1, -1)

  prediction_results = []
  for model_name, model in self.models.items():
    predicted_label = encoder.inverse_transform(model.predict(input_data))
    prediction_results.append(f"{model_name}: {predicted_label}")
    self.result_label.config(text="\n".join(prediction_results))

if __name__ == "__main__":
  root = tk.Tk()
  app = DroneControllerGUI(root)
  root.mainloop()
