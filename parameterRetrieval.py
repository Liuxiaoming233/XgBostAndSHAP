import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from  sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import KFold, cross_val_score

Physicochemical_properties = pd.read_csv('90_sdandard.txt', delimiter='\t')#读取物化性质

cumsum=Physicochemical_properties.columns[1:]#获取物化性质名称
pos_train_file = 'pos_train.txt'
neg_train_file = 'neg_train.txt'

pos_independent_file = 'pos_independent.txt'
neg_independent_file = 'neg_independent.txt'


def creat_feature(file_name):
    data = []
    for line in open(file_name):
        line = line.strip()
        if line.startswith('>'):
            continue
        else:
            length = len(line)
            K_mer_feature = []
            for i in range(length - 1):
                properties = Physicochemical_properties[str(line[i] + line[i + 1])].values
                K_mer_feature.extend([properties.min(), properties.mean(), properties.var(), properties.max(), properties.sum()])
            data.append(K_mer_feature)
    return np.array(data)


pos_train_feature = creat_feature(pos_train_file)
neg_train_feature = creat_feature(neg_train_file)

X = np.concatenate((pos_train_feature, neg_train_feature), axis=0)
Y = y_ture = np.array([1]*len(pos_train_feature) + [0]*len(pos_train_feature))

pos_independent_feature = creat_feature(pos_independent_file)
neg_independent_feature = creat_feature(neg_independent_file)

X_independent = np.concatenate((pos_independent_feature, neg_independent_feature), axis=0)
Y_independent = np.array([1]*len(pos_independent_feature) + [0]*len(pos_independent_feature))

# 数据缩放
scaler = MinMaxScaler().fit(X)
X_scale = scaler.transform(X)

X_independent = scaler.transform(X_independent)

# 方差分析求F值
F, pval = f_classif(X_scale, Y)
index = np.argsort(F)[::-1]

#聚类

cluster = AgglomerativeClustering().fit(X_scale.T)


cluster_result = cluster.children_

cluster_data = cluster_result[(cluster_result[:, 0] < 400) & (cluster_result[:, 1] < 400)]
cluster_only_feature = set(i for i in range(400)) - set(cluster_data.flatten())
cluster_select_feature = []
cluster_no_select_feature = []
for i in cluster_data:
    if F[i[0]] > F[i[1]]:
        cluster_select_feature.append(i[0])
        cluster_no_select_feature.append(i[1])
    else:
        cluster_select_feature.append(i[1])
        cluster_no_select_feature.append(i[0])

selected_feature = list(cluster_only_feature) + cluster_select_feature


loop = KFold(random_state=2, n_splits=5, shuffle=True)
# svmModel = SVC()
# param_grid = {'kernel':['rbf', 'poly', 'sigmoid', 'linear'],'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]}
# param_grid = {'kernel':['rbf', 'poly', 'sigmoid', 'linear']}

GBC = GradientBoostingClassifier(random_state=42)
param_grid = {"n_estimators": range(10,1000,20), "learning_rate":[0.1,1e-2,1e-3,1e-4,1e-5],"max_depth":range(1,11)}

model = GridSearchCV(GBC, param_grid, verbose=1, return_train_score=True, cv=loop,n_jobs=14)
# model = RandomForestClassifier(n_estimators=100)
model.fit(X_scale[:, selected_feature], Y)
print("model.best_score_:", model.best_score_,
               'model.best_params_:', model.best_params_)

means = model.cv_results_['mean_test_score']
params = model.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))