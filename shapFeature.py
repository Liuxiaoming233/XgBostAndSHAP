import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# SHAP解释器
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 总体特征重要性
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# 单个预测解释
sample_index = 0
shap.plots.waterfall(shap_values[sample_index])

# 依赖图
shap.dependence_plot("mean radius", shap_values, X_test, feature_names=feature_names)
