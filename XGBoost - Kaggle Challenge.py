import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib as mtl 
import seaborn as sns
from collections import Counter


## Function to identify Outliers
def check_outliers(df,IQR_thr):
	index_val = []
	features = ["age","resting_electrocardiographic_results","fasting_blood_sugar"]    
    # Identify IQR range (first minues second quantile) and identify outliers which are 
	# outside range of threshold multiplied by IQRrange
	for col in features:
		outlier_range = (np.percentile(df[col], 25) - np.percentile(df[col],75))*IQR_thr
		index_val.extend(df[(df[col] < np.percentile(df[col], 25) - outlier_range) | (df[col] > np.percentile(df[col],75) + outlier_range )].index)
        
    # select observations containing more than 2 outliers
	index_val = Counter(index_val)        
	
	consol_index = []
	for k, v in index_val.items():
		if v > 2 :
			consol_index.append(k)
	
	return consol_index   

mtl.rcParams['agg.path.chunksize'] = 10000

## Import the training file into pandas
traindf = pd.read_csv("train.csv")

## Import the test file into pandas
testdf = pd.read_csv("test.csv")

## Explorator Data Analysis

# Explore resting_electrocardiographic_results distribution 
g = sns.distplot(traindf["resting_electrocardiographic_results"], color="m", label="Skewness : %.2f"%(traindf["resting_electrocardiographic_results"].skew()))
g = g.legend(loc="best")

# Explore fasting_blood_sugar distribution 
g = sns.distplot(traindf["fasting_blood_sugar"], color="m", label="Skewness : %.2f"%(traindf["fasting_blood_sugar"].skew()))
g = g.legend(loc="best")

# Explore serum_cholestoral distribution 
g = sns.distplot(traindf["serum_cholestoral"], color="m", label="Skewness : %.2f"%(traindf["serum_cholestoral"].skew()))
g = g.legend(loc="best")

# Explore resting_blood_pressure distribution 
g = sns.distplot(traindf["resting_blood_pressure"], color="m", label="Skewness : %.2f"%(traindf["resting_blood_pressure"].skew()))
g = g.legend(loc="best")

# Explore maximum_heart_rate_achieved distribution 
g = sns.distplot(traindf["maximum_heart_rate_achieved"], color="m", label="Skewness : %.2f"%(traindf["maximum_heart_rate_achieved"].skew()))
g = g.legend(loc="best")

corr = traindf.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

## Pre-processing of data

# detect outliers from Age, SibSp , Parch and Fare
outlier_index = check_outliers(traindf,1.5)
traindf = traindf.drop(outlier_index, axis = 0).reset_index(drop=True)

## Create X and Y dataframes
X = traindf.drop(['ID','class'],axis=1)
y = traindf.loc[:,['class']]

## Split training data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=15)

## Hyper parameter tuning of xgboost
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic',scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='roc_auc',iid=False, cv=5)
gsearch1.fit(X_train,y_train)

## XG Boost with original data
clf = XGBClassifier(silent=True,
			  #booster = 'gbtree',
			  booster = 'gbtree',
			  scale_pos_weight=1,
			  learning_rate=0.1,
			  colsample_bytree = 0.8,
			  min_child_weight=5,
			  subsample = 0.8,
			  objective='binary:logistic', 
			  n_estimators=200, 
			  reg_alpha = 0.4,
			  max_depth=7, 
			  gamma=0.4,
			  seed=27)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy with XGBoost: %f" % accuracy_score(y_test,y_pred))

## Predict using original xgboost
#y_output = clf.predict(testdf.loc[:,'age':'thal'])
test_X = testdf.drop(['ID'],axis=1)
y_output = clf.predict(test_X)

predict_output = pd.DataFrame({'ID': testdf['ID'], 'class': y_output}, columns=['ID', 'class'])
predict_output.to_csv("CS17EMDS11028_Heart_prediction_01.csv",index=False)
print("Output file written!")
