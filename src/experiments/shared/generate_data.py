import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

warnings.simplefilter(action='ignore', category=FutureWarning)

def underlying_function(*args):
    """Generates the output with linear and quadratic elements.

    Args:
        *args: Variable length argument list for input features.

    Returns:
        float: The output value generated from the linear and quadratic components.
    """
    linear_component = sum(args)
    quadratic_component = sum(np.power(x, np.random.randint(2, 3)) for x in args)
    epsilon = np.random.normal(100, 10)
    output = linear_component + quadratic_component + epsilon
    return output

def generate_data(n, num_variables):
    """Generates random input data with different distributions.

    Args:
        n (int): Number of observations.
        num_variables (int): Number of input variables.

    Returns:
        pd.DataFrame: DataFrame containing the generated input data and output.
    """
    np.random.seed(0)
    data = {}

    distributions = [
        ('Uniform', lambda: np.random.uniform(-5, 5, n)),
        ('Exponential', lambda: np.random.exponential(1, n)),
        ('Poisson', lambda: np.random.poisson(2, n)),
        ('Normal', lambda: np.random.normal(0, 3, n))
    ]

    num_distributions = len(distributions)
    for i in range(num_variables):
        dist_name, dist_func = distributions[i % num_distributions]
        data[f'x{i+1}'] = dist_func()
        print(f"x{i+1} is generated using {dist_name} distribution.")

    output = underlying_function(*[data[f'x{i+1}'] for i in range(num_variables)])
    data['Output'] = output
    data_frame = pd.DataFrame(data)

    return data_frame

if __name__ == "__main__":
    np.random.seed(0)

    n = 5000
    num_variables = 4
    data = generate_data(n, num_variables)
    print("Shape of the data is: ", data.shape)
    print(data.head())

    X = data.drop('Output', axis=1)
    y = data['Output']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(n_estimators=10)
    xgb_model.fit(X_train, y_train)

    linear_preds = linear_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    linear_mse = mean_squared_error(y_test, linear_preds)
    linear_r2 = r2_score(y_test, linear_preds)
    xgb_mse = mean_squared_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    print(f"Linear Model MSE: {linear_mse}")
    print(f"Linear Model R2: {linear_r2}")
    print(f"XGBoost Model MSE: {xgb_mse}")
    print(f"XGBoost Model R2: {xgb_r2}")

    alpha = 0.05
    alphas = np.array([(alpha / 2), (1 - (alpha / 2)), 0.5])

    Xy = xgb.QuantileDMatrix(X, y)

    xgb_q_model = xgb.train(
        {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": alphas,
        },
        Xy,
        num_boost_round=10)

    xgb_q_preds = xgb_q_model.inplace_predict(X_test)

    print("Original XGB shape: ", xgb_preds.shape)
    print("Quantile XGB shape: ", xgb_q_preds.shape)

    xgb_mse = mean_squared_error(y_test, xgb_q_preds[:, 2])
    xgb
