import numpy as np
import pandas as pd
from openpyxl.styles.builtins import output
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb


Physicochemical_properties = pd.read_csv('90_sdandard.txt', delimiter='\t')  # 读取物化性质

cumsum = Physicochemical_properties.columns[1:]  # 获取物化性质名称
pos_train_file = 'pos_train.txt'
neg_train_file = 'neg_train.txt'

pos_independent_file = 'pos_independent.txt'
neg_independent_file = 'neg_independent.txt'

OUTPUTFILE = pd.DataFrame(columns=['trials', "sn", "sp", "acc", "MCC", "ROC"])


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
                K_mer_feature.extend(
                    [properties.min(), properties.mean(), properties.var(), properties.max(), properties.sum()])
            data.append(K_mer_feature)
    return np.array(data)


pos_train_feature = creat_feature(pos_train_file)
neg_train_feature = creat_feature(neg_train_file)

X = np.concatenate((pos_train_feature, neg_train_feature), axis=0)
Y = y_ture = np.array([1] * len(pos_train_feature) + [0] * len(pos_train_feature))

pos_independent_feature = creat_feature(pos_independent_file)
neg_independent_feature = creat_feature(neg_independent_file)

X_independent = np.concatenate((pos_independent_feature, neg_independent_feature), axis=0)
Y_independent = np.array([1] * len(pos_independent_feature) + [0] * len(pos_independent_feature))

# 数据缩放
scaler = MinMaxScaler().fit(X)
X_scale = scaler.transform(X)

X_independent = scaler.transform(X_independent)

# 方差分析求F值
F, pval = f_classif(X_scale, Y)
index = np.argsort(F)[::-1]

# 聚类

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


"""
给XBoost分类器进行寻参
"""
# rf = xgb.XGBClassifier()
# param_grid = {'random_state':[42],\
#               'learning_rate':[0.01, 0.015, 0.025, 0.05, 0.1],\
#               'gamma':[0]+list(np.arange(0.05,0.1,0.01))+[ 0.3, 0.5, 0.7, 0.9, 1],\
#               'alpha':[0]+list(np.arange(0.01,0.1,0.01))+[1],\
#               'lambda':[0, 0.1, 0.5, 1],\
#               'max_depth':[3, 5, 6, 7, 9, 12, 15, 17, 25],\
#               'min_child_weight':[1, 3, 5, 7]}
#
# grid_search  = GridSearchCV(rf, param_grid, n_jobs=12, verbose=1, return_train_score=True, cv=loop)
# grid_result = grid_search.fit(X_scale[:, selected_feature], Y)
# print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# outputParams = open('output_params.txt','w')
# for mean,param in zip(means,params):
#     outputParams.write("%f  with:   %r\n" % (mean,param))
#
# print('-------------------------------------------------------------------------------')


def cross_valide(name):
    rf = RandomForestClassifier(random_state=42, max_depth=15, max_features=0.1, min_samples_leaf=1, \
                                min_samples_split=3, n_estimators=145, n_jobs=5)
    rf = xgb.XGBClassifier(reg_lambda = 0.1,alpha= 0.08, gamma= 0.5,  learning_rate= 0.1, max_depth= 5, min_child_weight= 7, random_state= 42)

    x_best = X_scale[:, selected_feature]
    result = []
    num = 0
    print("sn", "sp", "acc", "MCC", "ROC")
    num = 1
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
        ROC = roc_auc_score(y_test, predict_proba[:, 1])
        fpr_rf, tpr_rf, _ = roc_curve(y_test, predict_proba[:, 1], drop_intermediate=False)
        scores = [sn, sp, acc, MCC, ROC]
        print(scores)
        result.append(scores)
        OUTPUTFILE.loc[len(OUTPUTFILE.index)] = ['loop-%d' % num] + scores
        num += 1
    result = np.array(result)
    print(result.mean(0))
    OUTPUTFILE.loc[len(OUTPUTFILE.index)] = ['loop-mean'] + list(result.mean(0))


if __name__ == '__main__':
    cross_valide('result')

    # rf = RandomForestClassifier(random_state=42, max_depth=15, max_features=0.1, min_samples_leaf=1,
    #                             min_samples_split=3,
    #                             n_estimators=145, n_jobs=5)
    rf = xgb.XGBClassifier(reg_lambda = 0.1,alpha= 0.08, gamma= 0.5,  learning_rate= 0.1, max_depth= 5, min_child_weight= 7, random_state= 42)
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
    print([i*100 for i in scores])
    print('-------------------------------------------------------------------------------')
    OUTPUTFILE.loc[len(OUTPUTFILE.index)] = ['All'] + scores
    from culture import calculation
    print(calculation('./output_scores.csv',OUTPUTFILE)*100)
    OUTPUTFILE.to_csv('output_scores_XGBoost_42.csv')