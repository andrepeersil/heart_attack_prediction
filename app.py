import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import kagglehub


path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")
df = pd.read_csv(f'{path}/heart_failure_clinical_records_dataset.csv')

df_features = df.drop(columns=['DEATH_EVENT'])
X = df[pd.Series(df_features.columns)]
y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_gini = tree.DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3)
clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)

model = clf_gini.fit(X_train, y_train)

########

st.set_page_config(
    page_title="heart failure analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("## heart failure analysis")

options_sex = {"Male" : 1,
               "Female" : 0}
with st.sidebar:
    sex = st.pills("Sex", options_sex.keys(), selection_mode="single", default="Male" )
    sex = options_sex[sex]
    col1, col2 = st.columns(2)


    with col1:
        anaemia = st.checkbox("Anaemia")

    with col2:
        diabetes = st.checkbox("Diabetes")

    with col1:
        high_blood_pressure = st.checkbox("high blood pressure")

    with col2:
        smoking = st.checkbox("smoking")


    age = st.number_input("age", 
                          min_value=1, 
                          max_value=100, 
                          value=30)
    creatinine_phosphokinase = st.number_input("creatinine phosphokinase",
                                               min_value=df["creatinine_phosphokinase"].min(), 
                                               max_value=df["creatinine_phosphokinase"].max())
    ejection_fraction = st.number_input("ejection fraction",
                                        min_value=df["ejection_fraction"].min(), 
                                        max_value=df["ejection_fraction"].max())
    platelets = st.number_input("platelets",
                                min_value=df["platelets"].min(), 
                                max_value=df["platelets"].max())
    serum_creatinine = st.number_input("serum creatinine",
                                min_value=df["serum_creatinine"].min(), 
                                max_value=df["serum_creatinine"].max())
    serum_sodium = st.number_input("serum sodium",
                                min_value=df["serum_sodium"].min(), 
                                max_value=df["serum_sodium"].max())
    time = st.number_input("time",
                                min_value=df["time"].min(), 
                                max_value=df["time"].max())


response = [age,
        int(anaemia),
        creatinine_phosphokinase,
        int(diabetes),
        ejection_fraction,
        int(high_blood_pressure),
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        int(smoking),
        time]



# st.write("age: ", age)
# st.write("anaemia: ", int(anaemia))
# st.write("creatinine_phosphokinase :", creatinine_phosphokinase)
# st.write("diabetes :", int(diabetes))
# st.write("ejection_fraction :", ejection_fraction)
# st.write("high_blood_pressure :", int(high_blood_pressure))
# st.write("platelets :", platelets)
# st.write("serum_creatinine :", serum_creatinine)
# st.write("serum_sodium :", serum_sodium)
# st.write("sex :", sex)
# st.write("smoking :", int(smoking))
# st.write("time :", time)

# response

response = np.array(response).reshape(1, -1)


death_prob = clf_gini.predict_proba(response)

st.write(f"Heart failure probability: {round(death_prob[0][1],2)*100}%")