import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def Output(confusion_matrix_train, confusion_matrix_val, accuracy_train, accuracy_val, precision_train, precision_val, recall_train, recall_val, f1_score_train, f1_score_val, roc_auc_score_train, roc_auc_score_val, fpr_train, tpr_train, fpr_val, tpr_val):
    print('confusion_matrix_train: \n', confusion_matrix_train)
    print('confusion_matrix_val: \n', confusion_matrix_val)
    print('accuracy_train: ', accuracy_train)
    print('accuracy_val: ', accuracy_val)
    print('precision_train: ', precision_train)
    print('precision_val: ', precision_val)
    print('recall_train: ', recall_train)
    print('recall_val: ', recall_val)
    print('f1_score_train: ', f1_score_train)
    print('f1_score_val: ', f1_score_val)
    print('roc_auc_score_train: ', roc_auc_score_train)
    print('roc_auc_score_val: ', roc_auc_score_val)
    plt.plot(fpr_train, tpr_train, marker = 'o')
    plt.show()



X, y = load_breast_cancer(return_X_y=True)

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC()

# running model now
model = model3

k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"{k}-Fold")


for fold, (train_index, val_index) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
    confusion_matrix_val = confusion_matrix(y_val, y_val_pred)
    
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    
    precision_train = precision_score(y_train, y_train_pred)
    precision_val = precision_score(y_val, y_val_pred)
    
    recall_train = recall_score(y_train, y_train_pred)
    recall_val = recall_score(y_val, y_val_pred)
    
    f1_score_train = f1_score(y_train, y_train_pred)
    f1_score_val = f1_score(y_val, y_val_pred)
    
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred, pos_label=1)
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_pred, pos_label=1)
    
    roc_auc_score_train = roc_auc_score(y_train, y_train_pred)
    roc_auc_score_val = roc_auc_score(y_val, y_val_pred)
    
    Output(confusion_matrix_train, confusion_matrix_val, accuracy_train, accuracy_val, precision_train, precision_val, recall_train, recall_val, f1_score_train, f1_score_val, roc_auc_score_train, roc_auc_score_val, fpr_train, tpr_train, fpr_val, tpr_val)


