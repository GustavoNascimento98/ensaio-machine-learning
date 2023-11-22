import pandas as pd
import numpy as np
import itertools

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def get_scores_classification(model_name, y_real, y_hat, verbose=False):
    
    """
    Returns a 2D list with the main performance metrics of a classification algorithm.
    
    Parameters
    ----------
    model_name: Algorithm's Name
    y_real: real labels
    y_hat: predicted labels
    """
    
    model_accuracy  = accuracy_score(y_real, y_hat)
    model_precision = precision_score(y_real, y_hat)
    model_recall    = recall_score(y_real, y_hat)
    model_f1score   = f1_score(y_real, y_hat)
    
    if verbose == True:
        print(f'{model_name} performance metrics')
        print(f'Accuracy: \t{model_accuracy}')
        print(f'Precision: \t{model_precision}')
        print(f'Recall: \t{model_recall}')
        print(f'F1-score: \t{model_f1score}')

    data = [model_name, model_accuracy, model_precision, model_recall, model_f1score]
    
    return data


def get_scores_regression(model_name, y_real, y_hat, verbose=False):
    
    """
    Returns a 2D list with the main performance metrics of a regression algorithm.
    
    Parameters
    ----------
    model_name: Algorithm's Name
    y_real: real labels
    y_hat: predicted labels
    """
    
    model_r2  = r2_score(y_real, y_hat)
    model_mse = mean_squared_error(y_real, y_hat)
    model_mae    = mean_absolute_error(y_real, y_hat)
    model_mape   = mean_absolute_percentage_error(y_real, y_hat)
    

    if verbose == True:
        print(f'{model_name} performance metrics')
        print(f'R2 score: \t{model_r2}')
        print(f'MSE score: \t{model_mse}')
        print(f'RMSE score: \t{np.sqrt(model_mse)}')
        print(f'MAE score: \t{model_mae}')
        print(f'MAPE score: \t{model_mape}')

    data = [model_name, model_r2, model_mse, np.sqrt(model_mse), model_mae, model_mape]
    
    return data




def update_dataframe(df_metrics, data):
    
    df_update = pd.DataFrame([data], columns=df_metrics.columns)
    df_metrics = pd.concat([df_metrics, df_update], ignore_index=True)
    
    return df_metrics






def ml_training(algorithm, category='classification', *args, **kwargs):
    """Train an machine learning algorithm for a given set of parameters.

    Args:
        algorithm (sklearn class): The class of a machine learning estimator.
        *args: Sequence of datasets\n
                - Training data\n
                - Training label\n
                - Evaluation data\n
                - Evaluation label\n
        **kwargs: Parameters of the estimator passed as list.

    Returns:
        df_model: dataframe containing the metrics for all combination of parameters.
    """
    
    # creates a list of dictionaries containing the combination of all parameters
    keys = kwargs.keys()
    grid_params = [ dict(zip(keys, values)) for values in itertools.product(*kwargs.values()) ]


    if category == 'classification':
        df_model = pd.DataFrame(columns=['name', *keys, 'accuracy', 'precision', 'recall', 'f1score'])
        get_scores_function = get_scores_classification

    else:
        df_model = pd.DataFrame(columns=['name', *keys, 'R2', 'MSE', 'RMSE', 'MAE', 'MAPE'])
        get_scores_function = get_scores_regression
    
    
    for parameters in grid_params:

        X_train, y_train, X_val, y_val = args
        original_parameters = parameters.copy()

        # check if it is a polynomial regression
        if 'degree' in keys:
            degree = parameters.pop('degree')
            poly = PolynomialFeatures(degree=degree)

            X_train = poly.fit_transform(X_train)
            X_val = poly.transform(X_val)


        # model definition
        model = algorithm(**parameters)

        # model training
        model.fit(X_train, y_train)

        # model prediction
        yhat_val = model.predict(X_val)

        
        data = get_scores_function(type(model).__name__, y_val, yhat_val, verbose=False)

        data = data[:1] + [*original_parameters.values()] + data[1:]
        
        df_model = update_dataframe(df_model, data)
        
    return df_model