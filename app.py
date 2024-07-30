from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data
data = pd.read_csv('alzheimers_disease_data.csv')
data.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

# Split data into features and target
X = data.drop('Diagnosis', axis=1)
Y = data['Diagnosis']

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

# Define and train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {key: [float(value)] for key, value in request.form.items()}
        input_data_df = pd.DataFrame(input_data)
        prediction = model.predict(input_data_df)[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return "Error: Missing form field {}".format(str(e))

if __name__ == '__main__':
    app.run()
