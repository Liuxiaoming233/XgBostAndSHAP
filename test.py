import numpy as np
import pandas as pd
from matplotlib.pyplot import xlabel
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
import matplotlib.pyplot as plt
from sympy import false

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

#聚类热图
import seaborn as sns
sns.set()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# sns.set_context({'figure.figsize':[1, 3]})
g = sns.clustermap(X_scale,row_cluster=False,yticklabels=False,figsize=[5,5])
# ax = g.ax_heatmap
# label_y = ax.get_yticklabels()
# plt.setp(label_y,rotation=3600, horizontalalignment='left')

plt.savefig('./output/figue/heatmap.jpg', dpi=300)
plt.show()

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
    XGBoost = xgb.XGBClassifier(reg_lambda = 0.1, alpha= 0.08, gamma= 0.5, learning_rate= 0.1, max_depth= 5, min_child_weight= 7, random_state= 42)
    XGBoost.fit(X_scale[:, selected_feature], Y)

    sns.heatmap(data=X_scale[:, selected_feature],yticklabels=False,cmap="RdBu_r")
    plt.savefig('./output/figue/heatmap_feature.jpg',dpi=300)
    plt.show()

    predict = XGBoost.predict(X_independent[:, selected_feature])
    predict_proba = XGBoost.predict_proba(X_independent[:, selected_feature])
    report = classification_report(Y_independent, predict, output_dict=True, digits=5)
    sn = report["0"]["recall"]
    sp = report["1"]["recall"]
    acc = report['accuracy']
    MCC = matthews_corrcoef(Y_independent, predict)
    ROC = roc_auc_score(Y_independent, predict_proba[:, 1])
    scores = [sn, sp, acc, MCC, ROC]
    print([i*100 for i in scores])
    print('-------------------------------------------------------------------------------')

    plt.rcParams.update({"font.size": 16})
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    #特征名称
    featureNames = []
    for i in range(80):
        featureNames+=['%s_min'%i ,  '%s_mean'%i ,  '%s_var'%i ,  '%s_max'%i ,  '%s_sum'%i]
    # 获取特征重要性
    importance_dict = XGBoost.get_booster().get_score(importance_type='weight')

    # 将重要性转换为DataFrame
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    })

    # 按重要性排序
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    np.savetxt( "./output/table/importance.csv", importance_df, delimiter="," ,fmt = '%s')

    ###################################################
    import shap

    # 创建一个SHAP解释器
    explainer = shap.TreeExplainer(XGBoost)
    data = pd.DataFrame(X_scale[:, selected_feature],columns= [featureNames[x] for x in selected_feature] )
    # 计算SHAP值
    shap_values = explainer(data)
    # feature_importance = np.abs(shap_values).mean(axis=0)

    # 绘制SHAP总结图
    shap.summary_plot(shap_values, data, plot_type="bar",max_display=22,show=False)
    plt.xlabel('SHAP Value Mean')
    plt.axvline(x=0.3, color='r', linestyle='--')
    plt.axvline(x=0.2, color='r', linestyle='--')
    plt.axvline(x=0.1, color='r', linestyle='--')
    plt.xticks([x*0.1 for x in range(11)])
    plt.savefig('./output/figue/summary_plot_bar.jpg',dpi=300)
    plt.show()
    # shap.plots.bar(shap_values)
    # plt.title("XGBoost 模型特征重要性 - SHAP 总结图")
    # plt.show()

    # 你也可以绘制更详细的总结图
    shap.summary_plot(shap_values, data,max_display=16,show=False)
    plt.xlabel('SHAP Value')
    plt.savefig('./output/figue/summary_plot.jpg', dpi=300)
    plt.show()
    # plt.title("XGBoost 模型特征重要性 - SHAP 详细总结图")
    # plt.show()
    #特征的饼状图
    # plt.figure(figsize=[3,3],dpi=300)
    plt.pie([9,3,4],labels=['Sum','Max',"Min"], autopct='%1.1f%%')
    # plt.legend(loc="center left",bbox_to_anchor=(1, 0, 1, 1))
    plt.savefig('./output/figue/pie.jpg', dpi=300)
    plt.show()



    #特征微点的柱状图
    plt.bar([30,45,46,47,48,49,52,53,56,59,61,75],height=[1,1,1,2,1,1,2,1,1,3,1,1])
    plt.yticks([1,2,3])
    plt.ylabel("Times")
    plt.xticks(range(30,76,3))
    plt.xlabel("Location of Dinucleotide")
    plt.savefig('./output/figue/bar.jpg', dpi=300)
    plt.show()


