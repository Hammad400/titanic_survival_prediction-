import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

df=pd.read_csv(r'E:\step to ML\DataSets\titanic\train2.csv')
titanic_test=pd.read_csv(r'E:\step to ML\DataSets\titanic\test2.csv')

df=df.drop(labels="Cabin",axis=1)
df['Age']=df['Age'].fillna(df['Age'].median())
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
catagarical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
df['Sex']=(df['Sex']=='male').astype(int)
numerical_features2 = [feature for feature in df.columns if df[feature].dtypes != 'O']
c_matrix=df.corr()

strong_relation_features=[]
for feature in c_matrix['Survived'].sort_values(ascending=False).keys():
    if c_matrix['Survived'][feature]>=0.04 or c_matrix['Survived'][feature]<=-0.04:
        strong_relation_features.append(feature)
strong_relation_features.remove('Survived')
numerical_features2.remove('PassengerId')
X=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
# X=df[strong_relation_features]
# X=df[['Age','Sex','Fare','Pclass']]
y=df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_predicted))
print("Precision:",metrics.precision_score(y_test, y_predicted))
print("Recall:",metrics.recall_score(y_test, y_predicted))

def calculate_F1_score(x,y):
    from sklearn import metrics as mt
    return 2*mt.precision_score(x,y)*mt.recall_score(x,y)/(mt.precision_score(x,y)+mt.recall_score(x,y))
print("F1_score is:",calculate_F1_score(y_test,y_predicted))


import pickle
pickle.dump(model, open('titanic_servival_prediction.pkl','wb'))