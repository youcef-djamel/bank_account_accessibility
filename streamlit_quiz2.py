import pandas as pd
bank = pd.read_csv("C:\\Users\\User\\Downloads\\Datasets\\Financial_inclusion_dataset.csv")
#%%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
cols = list(bank.select_dtypes('object'))

bank[cols] = bank[cols].apply(encoder.fit_transform)
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

stan = StandardScaler()
bank_norm = stan.fit_transform(bank.drop(['bank_account', 'uniqueid'], axis=1))
bank_df = pd.DataFrame(bank_norm, columns=['country', 'year', 'location_type', 'cellphone_access', 'household_size', 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type'])
bank_df['bank_account'] = bank['bank_account']

x = bank_df.drop('bank_account', axis=1)
y = bank_df['bank_account']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy:', logreg.score(x_test, y_test))
#%%
import streamlit as st
import joblib
import numpy as np

#bank2 = pd.read_csv("C:\\Users\\User\\Downloads\\Datasets\\Financial_inclusion_dataset.csv")

joblib.dump(logreg, "Regression for bank account.pkl")
joblib.dump(stan, "standardScaler.pkl")

model = joblib.load("Regression for bank account.pkl")
scaler = joblib.load("standardScaler.pkl")

st.title('Machine learning classifier for Bank account accessibility')

country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.selectbox("Year", [2016, 2017, 2018])
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)
age = st.number_input("Age", min_value=16, max_value=100, value=30)
gender = st.selectbox("Gender", ["Female", "Male"])
relationship = st.selectbox("Relationship with Head", [
    "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
])
marital_status = st.selectbox("Marital Status", [
    "Married", "Single", "Widowed", "Divorced/Seperated", "Dont know"
])
education_level = st.selectbox("Education Level", [
    "No formal education", "Primary education", "Secondary education",
    "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"
])
job_type = st.selectbox("Job Type", [
    "Self employed", "Government Dependent", "Formally employed Private",
    "Informally employed", "Farming and Fishing", "Remittance Dependent",
    "Formally employed Government", "Other Income", "No Income"
])

# 2. Manual encoding (you should replace this with saved LabelEncoders if available)
country_map = {"Kenya": 0, "Rwanda": 1, "Tanzania": 2, "Uganda": 3}
location_map = {"Urban": 1, "Rural": 0}
cellphone_map = {"Yes": 1, "No": 0}
gender_map = {"Female": 0, "Male": 1}
relationship_map = {
    "Head of Household": 0, "Spouse": 1, "Child": 2, "Parent": 3,
    "Other relative": 4, "Other non-relatives": 5
}
marital_map = {
    "Married": 0, "Single": 1, "Widowed": 2, "Divorced/Seperated": 3, "Dont know": 4
}
education_map = {
    "No formal education": 0, "Primary education": 1, "Secondary education": 2,
    "Tertiary education": 3, "Vocational/Specialised training": 4, "Other/Dont know/RTA": 5
}
job_map = {
    "Self employed": 0, "Government Dependent": 1, "Formally employed Private": 2,
    "Informally employed": 3, "Farming and Fishing": 4, "Remittance Dependent": 5,
    "Formally employed Government": 6, "Other Income": 7, "No Income": 8
}

# 3. Encode values
input_raw = np.array([[
    country_map[country],
    year,
    location_map[location_type],
    cellphone_map[cellphone_access],
    household_size,
    age,
    gender_map[gender],
    relationship_map[relationship],
    marital_map[marital_status],
    education_map[education_level],
    job_map[job_type]
]])
input_scaled = scaler.transform(input_raw)



if st.button('Predict'):
    predictions = model.predict(input_scaled)
    if (predictions[0]==0):
        st.markdown('they have a bank account')
    elif (predictions[0]==1):
        st.markdown('they do not have a bank account')
















#country = st.selectbox('country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
#year = st.selectbox('year', [2018, 2016, 2017])
#location_type = st.selectbox('Location type', ['Rural', 'Urban'])
#cellphone_access = st.selectbox('Cellphone access',['Yes', 'No'])
#household_size = st.selectbox('household_size', options=bank2['household_size'].unique())
#age_of_respondent = st.selectbox('age of respondent', options=bank2['age_of_respondent'].unique())
#gender_of_respondent = st.selectbox('gender of respondent', ['Female', 'Male'])
#relationship_with_head = st.selectbox('relationship with head', options=bank2['relationship_with_head'].unique())
#marital_status = st.selectbox('marital status', options=bank2['marital_status'].unique())
#education_level = st.selectbox('education level', options=bank2['education_level'].unique())
#job_type = st.selectbox('job type', options=bank2['job_type'].unique())

