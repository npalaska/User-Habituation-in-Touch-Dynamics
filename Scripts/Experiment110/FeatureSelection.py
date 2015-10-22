import numpy as np
from numpy import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

traindata = genfromtxt(
    '../../Output Files/Experiment60/GIUser1/1-1-1-Test-1-1.csv', dtype=float,
    delimiter=',')
sample_train = traindata[:, 3:]

# scaler = preprocessing.StandardScaler().fit(sample_train)
X_train = preprocessing.normalize(sample_train, norm='l2')
# X_train = scaler.transform(sample_train)
target_train = np.ones(len(traindata))
np.savetxt("../../Output Files/Experiment110/Random Forest/ScaledFeatures.csv",
           X_train, delimiter=",")
row = 0
while row < len(traindata):
    if np.any(traindata[row, 0:3] != [1, 2, 1]):
        target_train[row] = 0
    row += 1
    # print((X_train.size)/len(X_train))
    #i = 0
    #for row in X_train:
    #   while i < X_train.size/len(X_train):
    #      if row[i]*(-1)>0:
    #          row[i] = row[i]*(-1)
    #    i += 1
    #i = 0

# estimator = SVR(kernel='linear')
#selector = RFECV(estimator, step=1, cv=2)
X_new = SelectKBest(f_classif, k=10).fit_transform(X_train, target_train)
np.savetxt(
    "../../Output Files/Experiment110/Random Forest/ImportantFeaturesBySelectKBest.csv",
    X_new, delimiter=",")
clf_scaled = ExtraTreesClassifier(n_estimators=100)
x_new1 = clf_scaled.fit(X_new, target_train).transform(X_new)
np.savetxt(
    "../../Output Files/Experiment110/Random Forest/ImportantFeaturesBySelectKBest-Tree.csv",
    x_new1,
    delimiter=",")
print(x_new1.shape)
#selector = selector.fit(X_train, target_train)
#print(selector.ranking_)

clf_scaled = ExtraTreesClassifier(n_estimators=100)
X_new = clf_scaled.fit(X_train, target_train).transform(X_train)
print(X_new.shape)
np.savetxt(
    "../../Output Files/Experiment110/Random Forest/ImportantFeatures.csv",
    X_new, delimiter=",")

"""
clf_scaled = ExtraTreesClassifier(n_estimators=100)
X_new = clf_scaled.fit(sample_train, target_train).transform(sample_train)
print(X_new.shape)
np.savetxt("../../Output Files/Experiment10/Random Forest/ImportantFeatures.csv", X_new, delimiter=",")

X_new = SelectKBest(f_classif, k=5).fit_transform(sample_train, target_train)
print(X_new.shape)
X_new = SelectKBest(chi2, k=5).fit_transform(sample_train, target_train)
print(X_new)
X_new = SelectPercentile(f_classif, percentile=10)
X_new.fit(sample_train, target_train)
importance = -np.log10(X_new.pvalues_)
print(max(importance))
print(importance/max(importance))
print(X_new)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
temp = sel.fit_transform(sample_train)
print(temp.shape)

X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(sample_train, target_train)
print(X_new.shape)
"""