import numpy as np
from numpy import random
import pandas as pd
import sklearn.metrics as skm


def predict_random(x: pd.DataFrame) -> pd.Series:
    """
    Labels each data point as a printed T-shirt or not.

    :param x: dataframe of input item data to be labelled

    :return y: series of whether or not the data is a printed T-shirt
    """

    N, _ = x.shape
    yhat = pd.Series(random.randint(0, 2, N))

    return yhat


def evaluate(predictor, x: pd.Series, y: pd.Series):
    """
    Displays evaluation metrics of how well the predicted output matches the 
    true labels.

    :param x: the input data for a predict function
    :param y: the true labels for the data
    """

    yhat = predictor(x)

    # Compute confusion matrix
    conf_matrix = skm.confusion_matrix(y, yhat)
    print("Confusion Matrix:")
    print(conf_matrix)

    print(skm.classification_report(y, yhat))


if __name__ == "__main__":
    from src import scrape
    from src import preprocessor
    
    df = scrape.load_data()
    x_pre = preprocessor.preprocessor(df)

    N, _ = x_pre.shape

    evaluate(predict_random, x_pre, pd.Series(np.ones(N, dtype=int)))
