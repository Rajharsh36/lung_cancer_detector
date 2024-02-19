import pandas as pd
from sklearn.ensemble import RandomForestClassifier


import numpy as np

data=pd.read_csv("survey lung cancer.csv")

show_d=pd.read_csv("survey lung cancer.csv")
age_bytes=pd.get_dummies(data.AGE , dtype=int)

data=pd.concat([data.drop('AGE', axis=1),age_bytes], axis=1)

data['GENDER']=data["GENDER"].apply(lambda x: 1 if x=='M' else 0)
data['SMOKING']=data["SMOKING"].apply(lambda x: 1 if x==2 else 0)
data['YELLOW_FINGERS']=data["YELLOW_FINGERS"].apply(lambda x: 1 if x==2 else 0)
data['ANXIETY']=data["ANXIETY"].apply(lambda x: 1 if x==2 else 0)
data['PEER_PRESSURE']=data["PEER_PRESSURE"].apply(lambda x: 1 if x==2 else 0)
data['CHRONIC DISEASE']=data["CHRONIC DISEASE"].apply(lambda x: 1 if x==2 else 0)
data['FATIGUE ']=data["FATIGUE "].apply(lambda x: 1 if x==2 else 0)
data['ALLERGY ']=data["ALLERGY "].apply(lambda x: 1 if x==2 else 0)
data['WHEEZING']=data["WHEEZING"].apply(lambda x: 1 if x==2 else 0)
data['ALCOHOL CONSUMING']=data["ALCOHOL CONSUMING"].apply(lambda x: 1 if x==2 else 0)
data['COUGHING']=data["COUGHING"].apply(lambda x: 1 if x==2 else 0)
data['SHORTNESS OF BREATH']=data["SHORTNESS OF BREATH"].apply(lambda x: 1 if x==2 else 0)
data['SWALLOWING DIFFICULTY']=data["SWALLOWING DIFFICULTY"].apply(lambda x: 1 if x==2 else 0)
data['CHEST PAIN']=data["CHEST PAIN"].apply(lambda x: 1 if x==2 else 0)
data['LUNG_CANCER']=data["LUNG_CANCER"].apply(lambda x: 1 if x=='YES' else 0)
label=data['LUNG_CANCER'].values
model=RandomForestClassifier(n_estimators=10)
data=data.drop('LUNG_CANCER' ,axis=1)
all_row_data=[]

for index,row in data.iterrows():
    row_data=row.to_list()
    all_row_data.append(row_data)
model.fit(all_row_data,label)

test_x=np.array([1,0,0,0,1,0,1,1,1,1,0,0,1,0,      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]).reshape(1,-1)
input_age=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
input_feature=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
age_needed=[21,38,39,44,46,47,48,49,51,52,53,54,55,56,57,58,59, 60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81,87]
feature_needed=[ 'GENDER','SMOKING',
                'YELLOW_FINGERS','ANXIETY',
                'PEER_PRESSURE','CHRONIC DISEASE',
                'FATIGUE ', 'ALLERGY ',
                'WHEEZING','ALCOHOL CONSUMING',
                'COUGHING', 'SHORTNESS OF BREATH',
                'SWALLOWING DIFFICULTY', 'CHEST PAIN',]

a=int(input("age  (in digit)\n:  "))
features_name=data.columns
i=0
f=0

for age_ in age_needed:
    if a==age_:
    
        input_age[i]=1
        break
    i+=1
for feature_ in feature_needed:
    
    if feature_=='GENDER':
        entered=input(f"{feature_} (m for male / f for female)\n:   ")
        if entered=='m':
            input_feature[f]=1
        elif entered=='f':
            input_feature[f]=0

    else:
        entered=input(f"{feature_} (y for yes / n for no)\n:   ")

        if entered=='y':
            input_feature[f]=1
        elif entered=='n':
            input_feature[f]=0
    f+=1
entered_data=np.concatenate([input_feature,input_age])

prediction=model.predict(entered_data.reshape(1,-1))
if prediction:
    print("you may have lung cancer")

else:
    print("you do not have lung cancer")
