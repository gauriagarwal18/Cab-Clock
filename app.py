from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import math
from datetime import datetime
from datetime import date
import calendar
import pandas as pd
from xgboost import XGBRegressor

app = Flask(__name__)
# model = pickle.load(open('xgboost_simple_regression_model.pkl', 'rb'))

try:
    with open('xgboost_simple_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
        print("Model loaded successfully!")
except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading the model: {e}")

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()



def calculate_distance(lat1_deg, lat2_deg, long1_deg, long2_deg):
    '''
    Calculate the great-circle distance between two points on the Earth using the spherical law of cosines.

    Parameters:
    lat1_deg : float - pickup latitude (in degrees)
    lat2_deg : float - dropoff latitude (in degrees)
    long1_deg : float - pickup longitude (in degrees)
    long2_deg : float - dropoff longitude (in degrees)

    Returns:
    float - distance in kilometers

    '''
    # Convert degrees to radians
    lat1_rad = np.array([math.radians(lat1_deg)])
    long1_rad = np.array([math.radians(long1_deg)])
    lat2_rad = np.array([math.radians(lat2_deg)])
    long2_rad = np.array([math.radians(long1_deg)])

    # Calculate distance using the spherical law of cosines
    a = (np.sin(lat1_rad)*np.sin(lat2_rad))+(np.cos(lat1_rad)*np.cos(lat2_rad)*np.cos(long2_rad-long1_rad))
  
    a = np.clip(a, -1, 1)
    b = np.arccos(a)
    distance_miles = 3963.0 * b  # Earth's radius in miles
    distance_km = distance_miles * 1.609344  # Convert miles to kilometers

    return distance_km[0]

def check_weekend(d):
  #this will return the array of ["pickup_year","pickup_month","pickuphour","pickup_weekday","is_weekend","pickup_shift"]
  x=calendar.day_name[d.date().weekday()]
  if x in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
    return 0
  else:
    return 1


def get_shift(x):
  if x in [0,1,2,3,4,5,6]:
    x = 2
    return x
  elif x in [7,8,9,10,11,12]:
    x = 3
    return x
  elif x in [13,14,15,16,17,18]:
    x = 1
    return x
  else:
    x = 0
    return x

    
@app.route("/predict", methods=['POST'])
def predict():

    """
    All the columns i want for training are:
    ['total_distance', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'is_weekend',
       'passenger_count', 'vendor_id', 'pickup_shift']

    All the columns which i want as input from the user
    ['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime',
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
       'trip_duration']

    #so i will take these columns from the user 
    vendor_id, pickup_datetime, 'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'

    total_distance: from longitudes and latitudes
    pickup_shift, is_weekend: pickup_datetime
    """
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        vendor_id = int(request.form['vendor_id'])
        datetime=(request.form['pickup_datetime'])
        pickup_datetime = pd.to_datetime(datetime, format='%Y-%m-%d %H:%M:%S') 
        passenger_count=int(request.form['passenger_count'])
        pickup_longitude=float(request.form['pickup_longitude'])
        pickup_latitude=float(request.form['pickup_latitude'])
        dropoff_longitude=float(request.form['dropoff_longitude'])
        dropoff_latitude=float(request.form['dropoff_latitude'])
        cols=[]

        #calculate total_distance, is_weekend, pickup_shift
        total_distance= calculate_distance(pickup_latitude,dropoff_latitude,pickup_longitude,dropoff_longitude)
        is_weekend= check_weekend(pickup_datetime)
        pickup_shift= get_shift(pickup_datetime.hour)
        
        cols=[total_distance, pickup_longitude, pickup_latitude,
       dropoff_longitude, dropoff_latitude, is_weekend,
       passenger_count, vendor_id, pickup_shift]

        
        prediction=model.predict([cols])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry this service is unavailable")
        else:
            return render_template('index.html',prediction_text="your trip duration will be(in sec): {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
