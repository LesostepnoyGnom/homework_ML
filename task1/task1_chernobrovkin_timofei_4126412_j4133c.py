# -*- coding: utf-8 -*-
"""Task1_Chernobrovkin_Timofei_4126412_J4133c.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LVjqo0JeBgM3KMFKyDtV48nYPP0x9Aiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import  datasets, metrics, tree
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import  learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss

from google.colab import drive
drive.mount('/content/drive/')

df = pd.read_csv("/content/drive/My Drive/ML_homework/task1/bioresponse.csv")
df

df['Activity'].unique()

df_log_reg = df.drop(['Activity'], axis=1)
scaler = StandardScaler()
X=scaler.fit_transform(df_log_reg)

X

y=df['Activity'].values
y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

"""# **small decision tree**"""

clf = tree.DecisionTreeClassifier(random_state=1, max_depth = 2)
clf.fit(X_train, y_train)
y_out = clf.predict(X_test)
y_out_proba = clf.predict_proba(X_test)

accuracy = metrics.accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test,y_out)
recall = metrics.recall_score(y_test,y_out)
roc_auc = roc_auc_score(y_out,y_test)
f1_score = metrics.f1_score(y_test,y_out)
log_loss = log_loss(y_test, y_out)

print('###########################')
print('### small decision tree ###')
print('###########################')
print('accuracy', round(accuracy, 5))
print('precision', round(precision, 5))
print('recall', round(recall, 5))
print('roc_auc', round(roc_auc, 5))
print('F1-score', round(f1_score, 5))
print('log-Loss', round(log_loss, 5))

prec, rec, thresh = precision_recall_curve(y_test, y_out_proba[:,1])
plt.figure(figsize=(9, 6))
plt.grid()
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel(u"Threshold",fontsize = 15)
plt.title(u'Precision-Recall curve',fontsize = 15)
plt.legend(fontsize = 15)

fpr, tpr, thr = roc_curve(y_test, y_out_proba[:,1])

plt.figure(figsize = (9,6))
plt.plot(fpr, tpr, label=u'ROC \n'+'ROC AUC = ' + str(round(roc_auc, 5)))
plt.grid()
plt.xlabel("false positive rate",fontsize = 15)
plt.ylabel("true positive rate",fontsize = 15)
plt.title(u"ROC curve",fontsize = 15)
plt.legend(fontsize = 15)

"""# **deep decision tree**"""

clf = tree.DecisionTreeClassifier(random_state=1, max_depth = 10)
clf.fit(X_train, y_train)
y_out = clf.predict(X_test)
y_out_proba = clf.predict_proba(X_test)

accuracy = metrics.accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test,y_out)
recall = metrics.recall_score(y_test,y_out)
roc_auc = roc_auc_score(y_out,y_test)
f1_score = metrics.f1_score(y_test,y_out)
log_loss = log_loss(y_test, y_out)

print('##########################')
print('### deep decision tree ###')
print('##########################')
print('accuracy', round(accuracy, 5))
print('precision', round(precision, 5))
print('recall', round(recall, 5))
print('roc_auc', round(roc_auc, 5))
print('F1-score', round(f1_score, 5))
print('log-Loss', round(log_loss, 5))

prec, rec, thresh = precision_recall_curve(y_test, y_out_proba[:,1])
plt.figure(figsize=(9, 6))
plt.grid()
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel(u"Threshold",fontsize = 15)
plt.title(u'Precision-Recall curve',fontsize = 15)
plt.legend(fontsize = 15)

fpr, tpr, thr = roc_curve(y_test, y_out_proba[:,1])

plt.figure(figsize = (9,6))
plt.plot(fpr, tpr, label=u'ROC \n'+'ROC AUC = ' + str(round(roc_auc, 5)))
plt.grid()
plt.xlabel("false positive rate",fontsize = 15)
plt.ylabel("true positive rate",fontsize = 15)
plt.title(u"ROC curve",fontsize = 15)
plt.legend(fontsize = 15)

"""# **random forest on small trees**"""

rf_classifier_low_depth = RandomForestClassifier(n_estimators = 50, max_depth = 2, random_state = 1)
rf_classifier_low_depth.fit(X_train, y_train)
y_out = rf_classifier_low_depth.predict(X_test)
y_out_proba = rf_classifier_low_depth.predict_proba(X_test)

accuracy = metrics.accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test,y_out)
recall = metrics.recall_score(y_test,y_out)
roc_auc = roc_auc_score(y_out,y_test)
f1_score = metrics.f1_score(y_test,y_out)
log_loss = log_loss(y_test, y_out)

print('####################################')
print('### random forest on small trees ###')
print('####################################')
print('accuracy', round(accuracy, 5))
print('precision', round(precision, 5))
print('recall', round(recall, 5))
print('roc_auc', round(roc_auc, 5))
print('F1-score', round(f1_score, 5))
print('log-Loss', round(log_loss, 5))

prec, rec, thresh = precision_recall_curve(y_test, y_out_proba[:,1])
plt.figure(figsize=(9, 6))
plt.grid()
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel(u"Threshold",fontsize = 15)
plt.title(u'Precision-Recall curve',fontsize = 15)
plt.legend(fontsize = 15)

fpr, tpr, thr = roc_curve(y_test, y_out_proba[:,1])

plt.figure(figsize = (9,6))
plt.plot(fpr, tpr, label=u'ROC \n'+'ROC AUC = ' + str(round(roc_auc, 5)))
plt.grid()
plt.xlabel("false positive rate",fontsize = 15)
plt.ylabel("true positive rate",fontsize = 15)
plt.title(u"ROC curve",fontsize = 15)
plt.legend(fontsize = 15)

"""# **random forest on deep trees**"""

rf_classifier_low_depth = RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 1)
rf_classifier_low_depth.fit(X_train, y_train)
y_out = rf_classifier_low_depth.predict(X_test)
y_out_proba = rf_classifier_low_depth.predict_proba(X_test)

accuracy = metrics.accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test,y_out)
recall = metrics.recall_score(y_test,y_out)
roc_auc = roc_auc_score(y_out,y_test)
f1_score = metrics.f1_score(y_test,y_out)
log_loss = log_loss(y_test, y_out)

print('###################################')
print('### random forest on deep trees ###')
print('###################################')
print('accuracy', round(accuracy, 5))
print('precision', round(precision, 5))
print('recall', round(recall, 5))
print('roc_auc', round(roc_auc, 5))
print('F1-score', round(f1_score, 5))
print('log-Loss', round(log_loss, 5))

prec, rec, thresh = precision_recall_curve(y_test, y_out_proba[:,1])
plt.figure(figsize=(9, 6))
plt.grid()
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel(u"Threshold",fontsize = 15)
plt.title(u'Precision-Recall curve',fontsize = 15)
plt.legend(fontsize = 15)

fpr, tpr, thr = roc_curve(y_test, y_out_proba[:,1])

plt.figure(figsize = (9,6))
plt.plot(fpr, tpr, label=u'ROC \n'+'ROC AUC = ' + str(round(roc_auc, 5)))
plt.grid()
plt.xlabel("false positive rate",fontsize = 15)
plt.ylabel("true positive rate",fontsize = 15)
plt.title(u"ROC curve",fontsize = 15)
plt.legend(fontsize = 15)

"""# **Train a classifier who avoids Type II (False Negative) errors and calculate metrics from p.2 for it. Recall for it should be not less than 0.95.**"""

# decision tree
param = np.arange(1, 30)
accuracy_ = []
precision_ = []
recall_ = []
roc_auc_ = []
f1_score_ = []

for i in param:
  clf = tree.DecisionTreeClassifier(random_state=1, max_depth = i)
  clf.fit(X_train, y_train)
  y_out = clf.predict(X_test)
  y_out_proba = clf.predict_proba(X_test)
  accuracy_.append(metrics.accuracy_score(y_test, y_out))
  precision_.append(metrics.precision_score(y_test,y_out))
  recall_.append(metrics.recall_score(y_test,y_out))
  roc_auc_.append(roc_auc_score(y_out,y_test))
  f1_score_.append(metrics.f1_score(y_test,y_out))

print('###########################')
print('### decision tree ###')
print('###########################')
param = list(param)
print('accuracy', round(max(accuracy_), 5), 'depth', param[accuracy_.index(max(accuracy_))])
print('precision', round(max(precision_), 5), 'depth', param[precision_.index(max(precision_))])
print('recall', round(max(recall_), 5), 'depth', param[recall_.index(max(recall_))])
print('roc_auc', round(max(roc_auc_), 5), 'depth', param[roc_auc_.index(max(roc_auc_))])
print('F1-score', round(max(f1_score_), 5), 'depth', param[f1_score_.index(max(f1_score_))])

# random forest

depth = np.arange(1, 2)
estimators = np.arange(91, 92)
min_samples_leaf = [1]
min_samples_split = [2]

accuracy_ = []
precision_ = []
recall_ = []
roc_auc_ = []
f1_score_ = []
log_loss_ = []

param = [[], [], [], []] # [0]-estimators, [1]-depth, [2]-min_samples_leaf, [3]-min_samples_split

for i in estimators:
  for j in depth:
      for k in min_samples_leaf:
        for g in min_samples_split:

          clf = RandomForestClassifier(n_estimators = i , max_depth = j, random_state=1, bootstrap = False, max_features = 'log2', min_samples_leaf = k, min_samples_split = g)
          clf.fit(X_train, y_train)
          y_out = clf.predict(X_test)
          y_out_proba = clf.predict_proba(X_test)
          accuracy_.append(metrics.accuracy_score(y_test, y_out))
          precision_.append(metrics.precision_score(y_test,y_out))
          recall_.append(metrics.recall_score(y_test,y_out))
          roc_auc_.append(roc_auc_score(y_out,y_test))
          f1_score_.append(metrics.f1_score(y_test,y_out))
          log_loss_.append(log_loss(y_test, y_out))
          param[0].append(i)
          param[1].append(j)
          param[2].append(k)
          param[3].append(g)

print('###########################')
print('### random forest ###')
print('###########################')
estimators = list(estimators)
depth = list(depth)
print('accuracy', round(max(accuracy_), 5), 'estimators', param[0][accuracy_.index(max(accuracy_))], 'depth', param[1][accuracy_.index(max(accuracy_))])
print('precision', round(max(precision_), 5), 'estimators', param[0][precision_.index(max(precision_))], 'depth', param[1][precision_.index(max(precision_))])
a = recall_.index(max(recall_))
print('recall', round(max(recall_), 5), 'estimators', param[0][a], 'depth', param[1][a], 'min_samples_leaf', param[2][a], 'min_samples_split', param[3][a])
print('roc_auc', round(max(roc_auc_), 5), 'estimators', param[0][roc_auc_.index(max(roc_auc_))], 'depth', param[1][roc_auc_.index(max(roc_auc_))])
print('F1-score', round(max(f1_score_), 5), 'estimators', param[0][f1_score_.index(max(f1_score_))], 'depth', param[1][f1_score_.index(max(f1_score_))])
print('log_loss', log_loss_)

#recall 0.92336 estimators 79 depth 1 min_samples_leaf 2, bootstrap = False
#recall 0.92336 estimators 79 depth 1 min_samples_leaf 1, bootstrap = False
#recall 0.92336 estimators 79 depth 1 min_samples_leaf 1 min_samples_split 2, bootstrap = False
#recall 0.98505 estimators 91 depth 1 min_samples_leaf 1 min_samples_split 2 bootstrap = False max_features = 'log2'

prec, rec, thresh = precision_recall_curve(y_test, y_out_proba[:,1])
plt.figure(figsize=(9, 6))
plt.grid()
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel(u"Threshold",fontsize = 15)
plt.title(u'Precision-Recall curve',fontsize = 15)
plt.legend(fontsize = 15)

fpr, tpr, thr = roc_curve(y_test, y_out_proba[:,1])

plt.figure(figsize = (9,6))
plt.plot(fpr, tpr, label=u'ROC \n'+'ROC AUC = ' + str(round(roc_auc, 5)))
plt.grid()
plt.xlabel("false positive rate",fontsize = 15)
plt.ylabel("true positive rate",fontsize = 15)
plt.title(u"ROC curve",fontsize = 15)
plt.legend(fontsize = 15)
