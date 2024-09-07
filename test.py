from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd

# Provide all required arguments, including 'price'
data = CustomData(
    car_ID=1,
    symboling=1,
    wheelbase=88.6,
    carlength=168.0,
    carwidth=64.0,
    carheight=48.0,
    curbweight=2548,
    enginesize=130,
    boreratio=3.47,
    stroke=2.68,
    compressionratio=9.0,
    horsepower=111,
    peakrpm=5000,
    citympg=21,
    highwaympg=27,
    #price=13495.0,  # Make sure to include 'price'
    CarName='toyota',
    fueltype='gas',
    aspiration='std',
    doornumber='two',
    carbody='convertible',
    drivewheel='fwd',
    enginelocation='front',
    enginetype='dohc',
    cylindernumber='four',
    fuelsystem='mpfi'
)

# Convert to DataFrame
df = data.get_data_as_data_frame()

# Initialize the prediction pipeline
pipeline = PredictPipeline()

# Make prediction
result = pipeline.predict(df)

print(result)
