import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    numerical_columns = [
        'car_id', 'symboling', 'wheelbase', 'carlength', 'carwidth',
        'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke',
        'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'
    ]
    
    categorical_columns = [
        'carname', 'fueltype', 'aspiration', 'doornumber', 'carbody',
        'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'
    ]
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocess_column_names(self, df):
        '''
        Replace spaces in column names with underscores and convert to lowercase.
        '''
        df.columns = df.columns.str.replace(' ', '_', regex=False).str.lower()
        return df

    def preprocess_string_columns(self, df, string_columns):
        '''
        Convert string columns to lowercase and replace spaces with underscores in the column values.
        '''
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].str.lower().str.replace(' ', '_')
        return df

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            # DictVectorizer will handle categorical features
            logging.info(f"Numerical columns: {self.numerical_columns}")
            logging.info(f"Categorical columns: {self.categorical_columns}")

            return {
                'numerical': Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]
                ),
                'categorical': DictVectorizer(sparse=False)  # No handle_unknown parameter
            }
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Preprocess column names to replace spaces with underscores and convert to lowercase
            train_df = self.preprocess_column_names(train_df)
            test_df = self.preprocess_column_names(test_df)

            # List of string columns to process
            string_columns = self.categorical_columns
            
            # Preprocess string columns to convert to lowercase and replace spaces with underscores
            train_df = self.preprocess_string_columns(train_df, string_columns)
            test_df = self.preprocess_string_columns(test_df, string_columns)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Split data into features and target
            target_column_name = "price"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing
            preprocessing_objs = self.get_data_transformer_object()
            numerical_pipeline = preprocessing_objs['numerical']
            categorical_vectorizer = preprocessing_objs['categorical']

            # Numerical features transformation
            numerical_features_train = input_feature_train_df[self.numerical_columns]
            numerical_features_test = input_feature_test_df[self.numerical_columns]

            numerical_features_train_transformed = numerical_pipeline.fit_transform(numerical_features_train)
            numerical_features_test_transformed = numerical_pipeline.transform(numerical_features_test)

            # Convert categorical features to dictionaries for DictVectorizer
            categorical_features_train = input_feature_train_df[self.categorical_columns].to_dict(orient='records')
            categorical_features_test = input_feature_test_df[self.categorical_columns].to_dict(orient='records')

            categorical_features_train_transformed = categorical_vectorizer.fit_transform(categorical_features_train)
            categorical_features_test_transformed = categorical_vectorizer.transform(categorical_features_test)

            # Combine numerical and categorical features
            train_arr = np.hstack([
                numerical_features_train_transformed,
                categorical_features_train_transformed,
                np.array(target_feature_train_df).reshape(-1, 1)
            ])

            test_arr = np.hstack([
                numerical_features_test_transformed,
                categorical_features_test_transformed,
                np.array(target_feature_test_df).reshape(-1, 1)
            ])

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={
                    'numerical': numerical_pipeline,
                    'categorical': categorical_vectorizer
                }
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
