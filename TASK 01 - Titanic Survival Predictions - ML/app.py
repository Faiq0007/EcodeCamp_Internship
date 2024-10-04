from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model
random_forest = joblib.load('random_forest_model.pkl')  # Ensure this path is correct

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Initialize prediction as None
    if request.method == 'POST':
        # Get form data
        pclass = request.form['Pclass']
        sex = request.form['Sex']
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])
        embarked = request.form['Embarked']

        # Encode categorical variables
        sex_encoded = 1 if sex == 'female' else 0
        embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                                   columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

        # Make prediction
        prediction = random_forest.predict(input_data)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
