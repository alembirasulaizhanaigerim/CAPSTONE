from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('C:\Users\Айгерим\OneDrive\Рабочий стол\carrental\voting_model.pkl')
scaler = joblib.load('C:\Users\Айгерим\OneDrive\Рабочий стол\carrental\scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    date = int(data['date'].replace("-", ""))
    day_of_week = data['day_of_week']
    car_count = data['car_count']
    bike_count = data['bike_count']
    bus_count = data['bus_count']
    truck_count = data['truck_count']
    total = car_count + bike_count + bus_count + truck_count
    hour = data['hour']
    minute = data['minute']
    am_pm = data['am_pm']

    features = np.array([[date, day_of_week, car_count, bike_count, bus_count, truck_count, total, hour, minute, am_pm]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
