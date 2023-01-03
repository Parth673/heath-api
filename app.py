from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

model = pickle.load(open('model.pkl', 'rb'))
df1 = pd.read_csv('Symptom-severity.csv')
df2 = pd.read_csv('symptom_description_precaution.csv')

app = Flask(__name__)

@app.route("/")
def home():
    return "The following API is used to predict diseases from given symptoms using ML." \
           "Takes 5 symptoms and returns an array that contains info about disease matching to it." \
           "-by Parth Kakadia."


@app.route('/predict', methods=['POST'])
def predict():
    # stomach_pain, acidity,  ulcers_on_tongue, vomiting, cough,  chest_pain
    s1 = request.form.get('s1')
    s2 = request.form.get('s2')
    s3 = request.form.get('s3')
    s4 = request.form.get('s4')
    s5 = request.form.get('s5')

    psymptoms = [s1, s2, s3, s4, s5]
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    c = np.array(df2["Disease"])
    d = np.array(df2["Description"])
    e = np.array(df2["Precaution"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    nulls = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    psy = [psymptoms + nulls]
    disease = model.predict(psy)[0]

    # description, precaution = ""
    for i in range(len(c)):
        if disease == c[i]:
            description = d[i]
            precaution = e[i]

    result = {"disease":disease, "description":description, "precaution":precaution}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
