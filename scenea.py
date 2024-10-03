import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier,XGBRFClassifier
from lightgbm import LGBMClassifier
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import  LimeTabularExplainer
import lime



data=pd.read_csv('Scenario-A-merged_5s.csv')
data.columns=data.columns.str.strip()
print(data.columns)
print(data.info())
print(data.isna().sum())
data=data.replace([-np.inf,np.inf],np.NAN)
print(data.info())
def ip_convert(ip):
    ip=str(ip).split('.')
    if ip[0]<'127' and ip[0]>'0':
        return 'A'
    elif ip[0]>'128' and ip[0]<'192':
        return 'B'
    elif ip[0]>'192' and ip[0]<'223':
        return 'C'
    elif ip[0]>'223' and ip[0]<'240':
        return 'D'
    elif ip[0]>'240':
        return 'E'

data['Source IP']=data['Source IP'].apply(ip_convert)
data['Destination IP']=data['Destination IP'].apply(ip_convert)

lab=LabelEncoder()

for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])
print(data.label.value_counts())


x=[]
for i in data.columns.values:
    data['z-scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['z-scores']>3).sum()
    if outliers>0:
        x.append(i)

print(len(data))
thresh=3
for i in x[:7]:
    upper=data[i].mean()+thresh*data[i].std()
    lower=data[i].mean()-thresh*data[i].std()
    data=data[(data[i]>lower)&(data[i]<upper)]
print(len(data))

x=data.drop(['label','z-scores'],axis=1)
y=data.label

def backward_ele(x,y,val=0.05):
    for i in range(0,x.shape[1]):
        linear=sm.OLS(y,x).fit()
        if max(linear.pvalues) > val:
            index=np.argmax(linear.pvalues)
            column= x.columns[index]
            x = x.drop(columns=[column])
        else:
            break
    return x

back=backward_ele(x,y)
back=back.drop(['Source IP','Destination IP'],axis=1)
print('--------------------------------')
print(len(back.columns.values))
print('-------------------------------')
print(back.columns)

smote=SMOTE()
back,y=smote.fit_resample(back,y)

x_train,x_test,y_train,y_test=train_test_split(back,y,train_size=0.7,test_size=0.3,random_state=20)

print(y_test.values[0])

rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
print(rfc.score(x_test,y_test))
print(rfc.predict([x_test.values[0]]))
joblib.dump(rfc,'rf_tor.pkl')

lime_explainer = LimeTabularExplainer(
    training_data=np.array(x_train),  # Use the training data as numpy array
    mode='classification',
    feature_names=back.columns.tolist(),  # List of feature names
    class_names=[str(label) for label in np.unique(y)],  # Unique class names
    discretize_continuous=True  # Optionally discretize continuous features
)

# Explain the prediction for the first sample in the test set
exp = lime_explainer.explain_instance(x_test.values[0], rfc.predict_proba, num_features=len(back.columns))

# Save the explanation as an HTML file
exp.save_to_file('lime_explanation_scene_a.html')