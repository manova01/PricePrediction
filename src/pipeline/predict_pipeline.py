import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model = None
        self.preprocessor = None

    def _load_model_and_preprocessor(self):
        try:
            self.model = load_object(file_path=self.model_path)
            self.preprocessor = load_object(file_path=self.preprocessor_path)
        except Exception as e:
            raise CustomException(f"Error loading model or preprocessor: {str(e)}", sys)

    def predict(self, features):
        try:
            if self.model is None or self.preprocessor is None:
                self._load_model_and_preprocessor()
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 car_ID: int,
                 symboling: int,
                 wheelbase: float,
                 carlength: float,
                 carwidth: float,
                 carheight: float,
                 curbweight: float,
                 enginesize: float,
                 boreratio: float,
                 stroke: float,
                 compressionratio: float,
                 horsepower: float,
                 peakrpm: float,
                 citympg: float,
                 highwaympg: float,
                 price: float,
                 CarName: str,
                 fueltype: str,
                 aspiration: str,
                 doornumber: str,
                 carbody: str,
                 drivewheel: str,
                 enginelocation: str,
                 enginetype: str,
                 cylindernumber: str,
                 fuelsystem: str):
        
        # Numerical features
        self.car_ID = car_ID
        self.symboling = symboling
        self.wheelbase = wheelbase
        self.carlength = carlength
        self.carwidth = carwidth
        self.carheight = carheight
        self.curbweight = curbweight
        self.enginesize = enginesize
        self.boreratio = boreratio
        self.stroke = stroke
        self.compressionratio = compressionratio
        self.horsepower = horsepower
        self.peakrpm = peakrpm
        self.citympg = citympg
        self.highwaympg = highwaympg
        self.price = price
        
        # Categorical features
        self.CarName = CarName
        self.fueltype = fueltype
        self.aspiration = aspiration
        self.doornumber = doornumber
        self.carbody = carbody
        self.drivewheel = drivewheel
        self.enginelocation = enginelocation
        self.enginetype = enginetype
        self.cylindernumber = cylindernumber
        self.fuelsystem = fuelsystem

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "car_ID": [self.car_ID],
                "symboling": [self.symboling],
                "wheelbase": [self.wheelbase],
                "carlength": [self.carlength],
                "carwidth": [self.carwidth],
                "carheight": [self.carheight],
                "curbweight": [self.curbweight],
                "enginesize": [self.enginesize],
                "boreratio": [self.boreratio],
                "stroke": [self.stroke],
                "compressionratio": [self.compressionratio],
                "horsepower": [self.horsepower],
                "peakrpm": [self.peakrpm],
                "citympg": [self.citympg],
                "highwaympg": [self.highwaympg],
                "price": [self.price],
                "CarName": [self.CarName],
                "fueltype": [self.fueltype],
                "aspiration": [self.aspiration],
                "doornumber": [self.doornumber],
                "carbody": [self.carbody],
                "drivewheel": [self.drivewheel],
                "enginelocation": [self.enginelocation],
                "enginetype": [self.enginetype],
                "cylindernumber": [self.cylindernumber],
                "fuelsystem": [self.fuelsystem]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
