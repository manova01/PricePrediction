from flask import Flask, request, render_template
import logging
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

# Configure logging
logging.basicConfig(level=logging.DEBUG)

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            data = CustomData(
                CarName=request.form.get('CarName'),
                fueltype=request.form.get('fueltype'),
                aspiration=request.form.get('aspiration'),
                doornumber=request.form.get('doornumber'),
                carbody=request.form.get('carbody'),
                drivewheel=request.form.get('drivewheel'),
                enginelocation=request.form.get('enginelocation'),
                enginetype=request.form.get('enginetype'),
                cylindernumber=request.form.get('cylindernumber'),
                fuelsystem=request.form.get('fuelsystem'),
                car_ID=int(request.form.get('car_ID')),
                symboling=int(request.form.get('symboling')),
                wheelbase=float(request.form.get('wheelbase')),
                carlength=float(request.form.get('carlength')),
                carwidth=float(request.form.get('carwidth')),
                carheight=float(request.form.get('carheight')),
                curbweight=float(request.form.get('curbweight')),
                enginesize=float(request.form.get('enginesize')),
                boreratio=float(request.form.get('boreratio')),
                stroke=float(request.form.get('stroke')),
                compressionratio=float(request.form.get('compressionratio')),
                horsepower=float(request.form.get('horsepower')),
                peakrpm=float(request.form.get('peakrpm')),
                citympg=float(request.form.get('citympg')),
                highwaympg=float(request.form.get('highwaympg'))
            )

            pred_df = data.get_data_as_data_frame()
            logging.debug(f"Data for prediction: {pred_df}")

            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except CustomException as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('home.html', error=str(e))
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return render_template('home.html', error="An unexpected error occurred.")
    return render_template('home.html')
