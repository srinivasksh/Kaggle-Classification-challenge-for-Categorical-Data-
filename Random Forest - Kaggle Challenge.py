import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mtl 
import seaborn as sns

## Set higher data limits for plotting
mtl.rcParams['agg.path.chunksize'] = 10000

## Import the training file into pandas
traindf = pd.read_csv("train.csv")

## Import the test file into pandas
testdf = pd.read_csv("test.csv")

## Explorator Data Analysis

traindf_plot = traindf.astype(int)

## Batplot of age vs class
g = sns.factorplot(x="age", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of resting_electrocardiographic_results vs class
g = sns.factorplot(x="resting_electrocardiographic_results", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of fasting_blood_sugar vs class
g = sns.factorplot(x="fasting_blood_sugar", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of serum_cholestoral vs class
g = sns.factorplot(x="serum_cholestoral", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of resting_blood_pressure vs class
g = sns.factorplot(x="resting_blood_pressure", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of maximum_heart_rate_achieved vs class
g = sns.factorplot(x="maximum_heart_rate_achieved", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of chest vs class
g = sns.factorplot(x="chest", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of exercise_induced_angina vs class
g = sns.factorplot(x="exercise_induced_angina", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of oldpeak vs class
g = sns.factorplot(x="oldpeak", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of slope vs class
g = sns.factorplot(x="slope", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of number_of_major_vessels vs class
g = sns.factorplot(x="number_of_major_vessels", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Batplot of thal vs class
g = sns.factorplot(x="thal", y="class", data=traindf_plot,size=6, kind="bar", palette="muted")

## Correlation of training set
corr = traindf.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


## Create X and Y dataframes
X = traindf.drop(['ID','class'],axis=1)
y = traindf.loc[:,['class']]

## Split training data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=15)

## XG Boost classifier
clf = XGBClassifier(silent=True,
			  #booster = 'gbtree',
			  booster = 'gbtree',
			  scale_pos_weight=1,
			  learning_rate=0.1,
			  colsample_bytree = 0.8,
			  min_child_weight=6,
			  subsample = 0.8,
			  objective='binary:logistic', 
			  n_estimators=200, 
			  reg_alpha = 0.4,
			  max_depth=4, 
			  gamma=0.4,
			  seed=27)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy with XGBoost: %f" % accuracy_score(y_test,y_pred))

## Decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy with Decision Tree: %f" % accuracy_score(y_test,y_pred))

## Decision Random Forest classifier
clf = RandomForestClassifier(n_estimators=200, max_depth=4,random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy with Random Forest: %f" % accuracy_score(y_test,y_pred))

## Naive Bayes classifier (Gaussian)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy with Naive Bayes: %f" % accuracy_score(y_test,y_pred))

## Predict test set using xgboost (Which gave the highest accuracy)
test_X = testdf.drop(['ID'],axis=1)
y_output = clf.predict(test_X)

## Write output to file (to be uploaded to Kaggle)
predict_output = pd.DataFrame({'ID': testdf['ID'], 'class': y_output}, columns=['ID', 'class'])
predict_output.to_csv("CS17EMDS11028_Heart_prediction_02.csv",index=False)
print("Output file written!")
