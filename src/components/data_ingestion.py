# import necessary libraries
import os
import sys
import pandas as pd 

from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import data_transformation
from src.components.data_transformation import data_transformation_config


from src.components.model_trainer import model_trainer
from src.components.model_trainer import model_trainer_config



#Data ingestion Config class
@dataclass
class data_ingestion_config:
    #creating path string such as artifacts/train.csv
    train_data_path:str= os.path.join('artifacts', 'train.csv')
    test_data_path:str= os.path.join('artifacts', 'test.csv')
    raw_data_path:str= os.path.join('artifacts', 'data.csv')


# Data ingestion class
class data_ingestion:
    def __init__(self):
        # brings in the path string into this class 
        self.data_ingestion_config=data_ingestion_config()
    # initiate data ingestion
    def initiate_data_ingestion(self):
        try:
            filepath = 'notebook\data\stud.csv'
            df = pd.read_csv(filepath)
            logging.info('Read the dataset as dataframe')
            # creates only the artifact directory from artifacts/data.csv
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            #saves the read dataframe to artifacts/data.csv
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            
            #train test split
            logging.info("Train test split initiated")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)
            
            #saves the split dataframe into artifacts/train, test.csv respectively
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)


# if name is __main__ then run
if __name__ == '__main__':
    obj = data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    dataTransformation = data_transformation()
    train_arr, test_arr, _ = dataTransformation.initiate_data_transformation(train_data, test_data)
    
    modelTrainer = model_trainer()
    print(modelTrainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))
    
    