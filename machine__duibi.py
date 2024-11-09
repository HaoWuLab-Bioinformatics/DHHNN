from sklearn import svm
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

def load_alzheimers_data(dataset, use_feats, data_path):
    # Construct file path
    #data_file_path = os.path.join(data_path, f'{dataset}.xlsx')
    data_file_path = os.path.join(data_path, f'{dataset}.csv')

    # Read the data
    #data = pd.read_excel(data_file_path)
    data = pd.read_csv(data_file_path)

    # Determine the index of the label column ('Group') and use it as the label
    label_col = 'Group'
    label_index = data.columns.get_loc(label_col)

    # Features are all columns except 'Group' column
    if use_feats:
        feature_values = data.drop(label_col, axis=1).values
        features = sp.csr_matrix(feature_values)
    else:
        # If not using features, create a unit feature matrix
        features = np.ones((data.shape[0], 1))

    # Labels are the 'Group' column
    labels = data[label_col].values

    # Build adjacency matrix using KNN, fixing the use of 5 neighbors
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    # Initialize adjacency matrix
    adj = sp.lil_matrix((data.shape[0], data.shape[0]))

    # Fill the adjacency matrix
    for i in range(data.shape[0]):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # Ensure the adjacency matrix is symmetrical

    # Convert to CSR format
    adj = adj.tocsr()

    return adj, features, labels

# Load Alzheimer's data
data_path = 'data/alzheimers' # Assuming the extracted data is in this directory
dataset = 'alzheimers' # Change to your actual file name without extension
adjacency_matrix, features, labels = load_alzheimers_data(dataset, True, data_path)

# Since SVM doesn't use adjacency matrix, we will only use features and labels
X_train, X_test, y_train, y_test = train_test_split(features.toarray(), labels, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
clf = svm.SVC()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 打印准确率
print(f'Accuracy of the SVM model: {accuracy:.2f}')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每个折叠的结果
accuracies = []
confusion_matrices = []

# 五折交叉验证
for train_index, test_index in skf.split(features.toarray(), labels):
    # 分割数据
    X_train, X_test = features.toarray()[train_index], features.toarray()[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练SVM模型
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # 进行预测
    y_pred = clf.predict(X_test)

    # 计算并存储准确率
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # 生成并存储混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

# 打印交叉验证的准确率结果
print("Cross-Validation Accuracy Scores:", accuracies)

# 平均准确率
print("Average Accuracy:", np.mean(accuracies))

# 展示所有混淆矩阵
for cm in confusion_matrices:
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.show()
