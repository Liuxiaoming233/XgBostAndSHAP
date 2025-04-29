import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib import rcParams
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

config = {
    "font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
plt.rc('font',family='Times New Roman')


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import KFold, cross_val_score

Physicochemical_properties = pd.read_csv('90_sdandard.txt', delimiter='\t')

cumsum=Physicochemical_properties.columns[1:]
# fig, ax = plt.subplots(figsize =(22, 7))
# sns.violinplot(data=Physicochemical_properties[cumsum], scale='count',width=0.8)
#
# plt.xticks(rotation=40, size=14)
#plt.show()
#plt.savefig("小提琴图.tif", dpi=300, format='tif')
pos_train_file = 'pos_train.txt'
neg_train_file = 'neg_train.txt'

pos_independent_file = 'pos_test.txt'
neg_independent_file = 'neg_test.txt'


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

# 特征选择可视化
# ln_F =np.log(F)
# f_cluster_only_feature = ln_F[list(cluster_only_feature)]
# f_cluster_select_feature =ln_F[cluster_select_feature]
# f_cluster_no_select_feature =ln_F[cluster_no_select_feature]
# fig, ax = plt.subplots(figsize =(22, 7))
# plt.scatter(cluster_no_select_feature, f_cluster_no_select_feature, c='r', linewidths=1, alpha=0.5, marker='v')
# plt.scatter(cluster_select_feature, f_cluster_select_feature, c='b', linewidths=1, alpha=0.5, marker='o')
# plt.scatter(list(cluster_only_feature), f_cluster_only_feature, c='k', linewidths=1, alpha=0.5, marker='d')
#
# for i in range(184):
#     plt.plot([cluster_no_select_feature[i], cluster_select_feature[i]],[f_cluster_no_select_feature[i],f_cluster_select_feature[i]], 'k-.', alpha = 0.4)
#
#
# plt.xlim(-0.5, 405)
# plt.ylim(ln_F.min()-1,ln_F.max()+1)
# plt.xticks(np.arange(-10,401,10))
# plt.yticks(np.arange(np.min(ln_F)-0.8, np.max(ln_F)+0.8, step=1))
# plt.xticks(rotation=40, size=14)
# plt.legend(['Low F-value in clustering (185)', 'High F-value in clustering (185)', 'clustering into one cluster (30)'], loc=2, fontsize=10)
# plt.ylabel('ln(F)')
# plt.xlabel('Index of features')
# plt.savefig('特征选择结果1.tif', dpi=300, format='tif')
# plt.show()

# selected_F = F[selected_feature]
# selected_index = np.argsort(selected_F)[::-1]

# linkage_matrix = ward(X_scale.T)
# dendrogram(linkage_matrix)
#
# plt.xticks(rotation=90, size=4)
#
#plt.savefig("层次聚类.tif", dpi=600, format='tif')
#plt.show()




rf = RandomForestClassifier(random_state=42)
# cross_s = cross_val_score(rf, X_scale[:, selected_feature], Y, cv=5)
# #cross_s = cross_val_score(svm, X_scale[:, selected_index], Y, cv=5)
# print(cross_s.mean())
#
# rf.fit(X_scale[:, selected_feature], Y)
# predict_Y = rf.predict(X_independent[:, selected_feature])
#
# report = classification_report(Y_independent, predict_Y, digits=5)
# print(report)
#
# print('over')

# param_grid = {
#   'n_estimators':np.arange(80, 150, 5),
#   'max_depth':np.arange(15, 18),
#   'min_samples_leaf': np.arange(1, 8),
#   'min_samples_split':np.arange(2, 5),
#    'max_features':np.arange(0.1, 1)
# }
#
#
loop = KFold(random_state=2, n_splits=5, shuffle=True)
# model = GridSearchCV(rf, param_grid, n_jobs=12, verbose=1, return_train_score=True, cv=loop)
# model.fit(X_scale[:, selected_feature], Y)
# print("model.best_score_:", model.best_score_,
#               'model.best_params_:', model.best_params_)

#loop = KFold(random_state=2, n_splits=5, shuffle=True)

def cross_valide(name):

    #rf = RandomForestClassifier(random_state=42, max_depth=15, max_features=0.1, min_samples_leaf=1, min_samples_split=3, n_estimators=145, n_jobs=5)
    rf = MLPClassifier()
    x_best = X_scale[:, selected_feature]
    result = []
    #f = open('roc_阈值_五折.txt', 'a')
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.figure(dpi=350)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    col = ['darkorange', 'yellow', 'red', 'green', 'blue']
    num = 0
    for train, test in loop.split(x_best, y_ture):
        X_train, X_test = x_best[train], x_best[test]
        y_train, y_test = y_ture[train], y_ture[test]
        rf.fit(X_train, y_train)
        predict = rf.predict(X_test)
        predict_proba = rf.predict_proba(X_test)
        report = classification_report(y_test, predict, output_dict=True, digits=5)
        sn = report["0"]["recall"]
        sp = report["1"]["recall"]
        acc = report['accuracy']
        MCC = matthews_corrcoef(y_test, predict)
        ROC  = roc_auc_score(y_test, predict_proba[:, 1])
        fpr_rf, tpr_rf, _ = roc_curve(y_test, predict_proba[:, 1], drop_intermediate=False)
        # f.write("thread:\n"+str(_)+"\n")
        # f.write("fpr_rf:\n"+str(fpr_rf) + "\n")
        # f.write("tpr_rf:\n"+str(tpr_rf) + "\n")

        #plt.plot(fpr_rf, tpr_rf, linewidth=2, color=col[num], label=str(num+1)+'-cross validation (area = %0.2f)' % ROC)

        scores = [sn, sp, acc, MCC, ROC]
        #print(scores)
        result.append(scores)
        num += 1
    result = np.array(result)
    #np.savetxt(name+".txt", result, delimiter=',', fmt='%0.5f')
   # f.close()
   #  plt.xlabel('False positive rate')
   #  plt.ylabel('True positive rate')
   #  plt.title('ROC curve')
   #  plt.legend(loc='best')
   #  plt.savefig('ROC.tif', dpi=300, format='tif')
    print(result.mean(0))

cross_valide('result')

# rf = RandomForestClassifier(random_state=42, max_depth=15, max_features=0.1, min_samples_leaf=1, min_samples_split=3,
#                             n_estimators=145, n_jobs=5)
rf  = MLPClassifier()
rf.fit(X_scale[:, selected_feature], Y)

predict = rf.predict(X_independent[:, selected_feature])
predict_proba = rf.predict_proba(X_independent[:, selected_feature])
report = classification_report(Y_independent, predict, output_dict=True, digits=5)

sn = report["0"]["recall"]
sp = report["1"]["recall"]
acc = report['accuracy']
MCC = matthews_corrcoef(Y_independent, predict)
ROC = roc_auc_score(Y_independent, predict_proba[:, 1])
scores = [sn, sp, acc, MCC, ROC]
print(scores)