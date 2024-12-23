from prefect import task,flow
import pandas as pd
from joblib import load

@task
def load_data(data):
    data = pd.DataFrame(data,columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn'])
    return data

mode_features = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
mean_features = ["tenure","MonthlyCharges","TotalCharges"]
mode_imputer = load("../pipelines/mode_imputer.joblib")
mean_imputer = load("../pipelines/mean_imputer.joblib")
@task
def clean_data(org_data):
    data = org_data.copy()
    mode_features = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
    mean_features = ["tenure","MonthlyCharges","TotalCharges"]

    mode_cleaned_data = pd.DataFrame(mode_imputer.transform(data[mode_features]),columns=mode_features)
    mean_cleaned_data = pd.DataFrame(mean_imputer.transform(data[mean_features]),columns=mean_features)

    return pd.concat([mode_cleaned_data,mean_cleaned_data],axis=1)

@task
def feature_engineering(org_data):
    if isinstance(org_data,pd.DataFrame):
        data = org_data.copy()
    else:
        data = pd.DataFrame(org_data,columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn'])
        
    data['TenureCategory'] = pd.cut(data['tenure'], bins=[-1,3,12,np.inf],labels=["Recent", "Established", "Loyal"]) 
    data['Tenure_X_MonthlyCharges'] = data['tenure'] * data['MonthlyCharges']
    data['HighSpender'] = data['TotalCharges'].apply(lambda x: True if x > 5000 else False)
    data['MonthlyChargesRatio'] = np.where(data['TotalCharges']!=0,data['MonthlyCharges'] / data['TotalCharges'],0)
    data['SqrtTotalCharges'] = data['TotalCharges'].apply(np.sqrt)
    data['SqrtMonthlyCharges'] = data['MonthlyCharges'].apply(np.sqrt)
    data['AutomaticPayment'] = data['PaymentMethod'].apply(lambda x: True if "automatic" in x else False)
    data['AllServicesActivated'] =  (data['OnlineSecurity'] == "Yes") & (data['OnlineBackup'] == "Yes") & \
                                   (data['DeviceProtection'] == "Yes") & (data['TechSupport'] == "Yes")
    data['TeleServicesActivated'] = (data['PhoneService'] == "Yes") & (data['MultipleLines'] == "Yes") & \
                                    (data['InternetService'] == "Yes")
    data['Connected'] = (data['Partner'] == "Yes") & (data['Dependents'] == "Yes")
    
    return data

ordinal_features = ["SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","TenureCategory","HighSpender","AutomaticPayment","AllServicesActivated","TeleServicesActivated","Connected"]
nominal_features = ["gender","PaymentMethod"]
ord_enc = load("../pipelines/ord_enc.joblib")
one_hot_enc = load("../pipelines/one_hot_enc.joblib")
@task
def encoding_data(org_data):
    data = org_data.copy()
    ordinal_features = ["SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","TenureCategory","HighSpender","AutomaticPayment","AllServicesActivated","TeleServicesActivated","Connected"]
    nominal_features = ["gender","PaymentMethod"]

    data[ordinal_features] = ord_enc.transform(data[ordinal_features])
    
    one_hot_encoded_data = one_hot_enc.transform(data[nominal_features])
    one_hot_encoded_data_columns = one_hot_enc.get_feature_names_out(input_features=data[nominal_features].columns)
    one_hot_data = pd.DataFrame(one_hot_encoded_data,columns=one_hot_encoded_data_columns)
    
    return pd.concat([data,one_hot_data],axis=1).drop(nominal_features,axis=1)

num_features = ["tenure","MonthlyCharges","TotalCharges","MonthlyChargesRatio","Tenure_X_MonthlyCharges","SqrtTotalCharges","SqrtMonthlyCharges"]
scaler = load("../pipelines/scaler.joblib")
@task
def scaling_data(data):
    data[num_features] = scaler.transform(data[num_features])

    return data

@flow(name="Pipeline")
def pipeline(data):
    processed_data = load_data(data)
    processed_data = clean_data(processed_data)
    processed_data = feature_engineering(processed_data)
    processed_data = scaling_data(processed_data)
    processed_data = encoding_data(processed_data)
    return processed_data
