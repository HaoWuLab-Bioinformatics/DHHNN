import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# 读取CSV文件
data = pd.read_csv('data/diabet/diabet.csv')

# 划分特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 重新划分验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# 初始化模型
models = {
    "Logistic Regression": LogisticRegression(),
    "SVC": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# 训练和评估模型
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    results[name] = report

# 输出结果
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"\t{metric}:")
            for label, score in value.items():
                print(f"\t\t{label}: {score}")
        else:
            print(f"\t{metric}: {value}")

# 在测试集上评估最佳模型
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]
y_pred_test = best_model.predict(X_test)
test_report = classification_report(y_test, y_pred_test, output_dict=True)
print(f"\nBest Model ({best_model_name}) Test Report:")
for metric, value in test_report.items():
    if isinstance(value, dict):
        print(f"\t{metric}:")
        for label, score in value.items():
            print(f"\t\t{label}: {score}")
    else:
        print(f"\t{metric}: {value}")
