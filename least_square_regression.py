import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def estimate(data):
   """
   This function is responsible for:
   1. Converting the data frame in two matrices of predictors and response.
   2. Split the data into training and testing data in 5 folds.
   3. For each fold, compute regression coefficients.
   4. Add the list of coefficients in a result array.
   5. Obtain the final values of coefficients by taking the mean.
   6. Call the target_prediction function to obtain the predicted target values.
   :param data: data set in frame
   :return: returns data frames of actual and predicted values of prices.
   """
   # Data frame of predictors
   df_x = (pd.DataFrame(data, columns=(['id', 'wheel_base','length','width','curb_weight','num_of_cylinders','engine_size','horsepower']))).as_matrix()

   # Data frame of target
   df_y = (pd.DataFrame(data, columns=(['price']))).as_matrix()

   # Split the data into training and testing dataset using K-fold
   x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=10)

   kf = KFold(n_splits=5)
   result=[]
   for k,(train, test) in enumerate (kf.split(df_x,df_y)):
       # calculate the regression coefficients
       x = df_x[train]
       y = df_y[train]

       df_x_transpose = x.transpose()
       b = np.dot(inv(np.dot(df_x_transpose, x)), np.dot(df_x_transpose, y))

       # Add the list of coefficients in result arrays
       result.append(b)

   # Compute mean of coefficients from k folds
   coefficients_mean = np.mean(result,axis=0)

   # Use the coefficients mean to calculte the measure closeness of data with regression line.
   y_pred = target_prediction(coefficients_mean, x_test)
   return (y_pred,y_test)

def target_prediction(coefficients_mean, x_test):
    """
    This function takes the final coefficient values and testing data set and predict the target values.
    :param coefficients_mean: Mean of coefficients' value after k fold.
    :param x_test: the test data split.
    :return: Predicted target values data.
    """
    coefficients_mean_transpose = np.transpose(coefficients_mean)
    x_test_transpose = np.transpose(x_test)

    # Predict the values based on calculated coefficients and linear equation
    y_pred = np.dot(coefficients_mean_transpose, x_test_transpose)
    y_pred = np.transpose(y_pred)
    return (y_pred)


def performance_parameters(y_pred, y_test):
   """
   This function is used to calculate the measure of accuracy and prints the parameters of performance accuracy.
   :param y_pred: Predicted values of prices for testing data set
   :param y_test: Actual prices from testing data set.
   :return: None
   """
   print "Mean square of errors",np.mean(((y_test) - (y_pred)) ** 2)
   ss_res = np.sum(((y_test) - (y_pred)) ** 2)
   ss_tot = np.sum(((y_test) - (np.mean(y_test))) ** 2)
   r2 = 1-(ss_res / ss_tot)
   print('Variance score: %.2f' % r2_score(y_test, y_pred))
   print "R^2: ", r2


def plot_regression_line(y_pred, y_test):
   """
   This function prints the scatted plot of actual versus predicted values of prices.
   :param y_pred: Predicted values of prices after training of model.
   :param y_test: Actual prices from testing data.
   :return: None
   """
   plt.plot(y_pred, y_test, 'ro')
   plt.xlabel('Predicted')
   plt.ylabel('Actual')
   plt.show()


def main():
   """
   1. Loads the data from final_auto which is preprocessed based on p and r values.
   2. Calls the estimate function which in turn implements K- fold and returns final results of model.
   3. Calls performance_parameters function to compute performance values.
   4. Calls the plot_regression_line the regression line based on performance values.
   """
   data = pd.read_csv('auto.csv')
   (y_pred,y_actual)=estimate(data)
   performance_parameters(y_pred,y_actual)
   plot_regression_line(y_pred,y_actual)

if __name__ == '__main__':
   main()