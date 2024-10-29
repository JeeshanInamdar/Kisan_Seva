import requests,json
import vonage
from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
import pickle
import requests
from geopy.geocoders import Nominatim

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#loading api keys
apikey="71ba422da362fb4025626bfd67adea37"
baseURL="https://api.openweathermap.org/data/2.5/weather?q="

#api for alert messages
client = vonage.Client(key="80466434", secret="sLBr4GvI7H3ZQcsV")
sms = vonage.Sms(client)

# Load the trained model and preprocessor
with open('dtr.pkl', 'rb') as model_file:
    dtr = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocesser = pickle.load(preprocessor_file)



# Set handle_unknown='ignore' for OneHotEncoder
preprocesser.named_transformers_['OHE'].handle_unknown = 'ignore'

# creating flask app
# Sample model and scaler objects for the first part of the code

fmodel = pickle.load(open('classifier1.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__, static_url_path='/static')
app = Flask(__name__, template_folder='/templates')

# Render front.html
app = Flask(__name__)
@app.route('/')
def index_amc():
    return render_template("index_amc.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/i')
def i():
    return render_template("i.html")

@app.route('/index3')
def index3():
    return render_template("index3.html")

@app.route('/weather')
def weather():
    # Your weather view logic goes here
    return render_template("weather.html")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')



#===============================================================================================================
# Render index.html with the first set of features
@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('i2.html', result=result)


@app.route('/predict3', methods=['POST'])
def predict3():
    Nitrogen = request.form.get('Nitrogen')
    Potassium = request.form.get('Potassium')
    Phosphorous = request.form.get('Phosphorous')

    # prediction
    result = fmodel.predict(np.array([[Nitrogen, Potassium, Phosphorous]]))
    # ... (rest of your code)
    if result[0] == 0:
        result = "TEN-TWENTY SIX-TWENTY SIX"
    elif result[0] == 1:
        result = "Fourteen-Thirty Five-Fourteen"
    elif result[0] == 2:
        result = "Seventeen-Seventeen-Seventeen"
    elif result[0] == 3:
        result = "TWENTY-TWENTY"
    elif result[0] == 4:
        result = "TWENTY EIGHT-TWENTY EIGHT"
    elif result[0] == 5:
        result = "DAP"
    else:
        result = "UREA"
    return render_template('result.html', result=str(result))

#api alert system
def get_current_location():
    try:
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode("me")
        return location.latitude, location.longitude
    except Exception as e:
        print(f"Error getting current location: {e}")
        return None

def get_city_name(longitude, latitude):
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.reverse((longitude, latitude), language='en')

    if location and location.address:
        city_name = location.address.split(",")[-3]
        return city_name.strip()
    else:
        return "City not found"

@app.route('/loc',methods=['POST'])
def loc():
    apikey = "71ba422da362fb4025626bfd67adea37"
    baseURL = "https://api.openweathermap.org/data/2.5/weather?q="
    cityName=request.form('user_input')
    completeURL = baseURL + cityName + "&appid=" + apikey
    response = requests.get(completeURL)
    data = response.json()

    T = round(data["main"]["temp"])
    Tmin = round(data["main"]["temp_min"])
    Tmax = round(data["main"]["temp_max"])
    hum = data["main"]["humidity"]
    sea = data["main"]["sea_level"]
    wind = data["wind"]["speed"]

   # l = f"Temperature = {T} degree celcius\nMinimum Temperature = {Tmin}\nMaximum Temperature = {Tmax}\nHumidity = {hum}\nCurrent sea level = {sea}\nCurrent wind speed is {wind} m/s"
   # return render_template('i2.html',result=l)

def main():
    location = get_current_location()
    cityName=get_city_name(location[0],location[1])
    apikey = "71ba422da362fb4025626bfd67adea37"
    baseURL = "https://api.openweathermap.org/data/2.5/weather?q="
    completeURL=baseURL+cityName+"&appid="+apikey
    response = requests.get(completeURL)
    data = response.json()
    print(data)
    Temp=round((data["main"]["temp"]-273.15))
    wind = data["wind"]["speed"]
    print(data)
    if True:
        responseData = sms.send_message(
            {
                "from": "Kisan SEVA",
                "to": "+918197034809",
                "text": f"WARNING!!!  Wind speed above threshold level\nWind speed = {wind}m/s\nTemperature = {Temp*3.6} degree celcius\n\n",
            }
        )

        if responseData["messages"][0]["status"] == "0":
            print("Message sent successfully.")
        else:
            print(f"Message failed with error: {responseData['messages'][0]['error-text']}")



# python main
if __name__ == "__main__":
    #main()
    app.run(debug=True)


