from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

df = pd.read_csv('yield_df.csv')  # Replace 'path_to_your_dataframe.csv' with the actual path to your DataFrame

# Load the trained model and preprocessor
with open('dtr.pkl', 'rb') as model_file:
    dtr = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocesser = pickle.load(preprocessor_file)

# Set handle_unknown='ignore' for OneHotEncoder
preprocesser.named_transformers_['OHE'].handle_unknown = 'ignore'

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/index2', methods=['POST'])
def index2():
    if request.method == 'POST':
        # Get input values from the form
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Make the prediction
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

        # Dynamically update the one-hot encoding for the new data
        transformed_features = preprocesser.transform(features)

        # Add the new categories to the preprocessor if needed
        if not set(df['Item'].unique()).issuperset([Item]):
            preprocesser.named_transformers_['OHE'].handle_unknown = 'ignore'
            transformed_features = preprocesser.transform(features)
            preprocesser.named_transformers_['OHE'].handle_unknown = 'error'

        # Make the prediction
        predicted_yield = dtr.predict(transformed_features).reshape(1, -1)

        # Display the result
        result = predicted_yield[0][0]
        return render_template('index2.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
