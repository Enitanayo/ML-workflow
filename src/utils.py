import os
import sys
import pickle

from src.exception import CustomException
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train ,X_test,  y_test, models, params):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            
            pred_train_val = model.predict(X_train)
            pred_test_val = model.predict(X_test)
            
            r2_pred_train_val = r2_score(y_train, pred_train_val)
            r2_pred_test_val = r2_score(y_test, pred_test_val)
            
            report[list(models.keys())[i]] = r2_pred_test_val
            
        return report
    except Exception as e:
        raise CustomException(e, sys)