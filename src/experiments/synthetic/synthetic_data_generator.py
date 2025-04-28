import numpy as np
import pandas as pd

def generate_synthetic_data(n, num_variables, seed, noise_distribution="normal", method="cosine"):
    """
    Generates synthetic data using the specified method.
    
    Options for 'method':
      - "cosine": Uses a cosine-based process 
          y = sum(cos(x_i)) + noise
      - "quadratic": Uses a quadratic-plus-periodic process 
          y = sum(x_i**2 + 0.5 * sin(3*x_i)) + noise
    
    Noise is drawn based on noise_distribution which can be "normal", "uniform", "lognormal", or "gumbel".
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, num_variables))
    
    if method == "cosine":
        deterministic = np.sum(np.cos(X), axis=1)
    elif method == "quadratic":
        deterministic = np.sum(X**2 + 0.5 * np.sin(3 * X), axis=1)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'cosine' or 'quadratic'.")
    
    if noise_distribution == "normal":
        noise = rng.normal(0, 0.5, size=n)
    elif noise_distribution == "uniform":
        noise = rng.uniform(-0.5, 0.5, size=n)
    elif noise_distribution == "lognormal":
        temp = rng.lognormal(mean=0, sigma=0.5, size=n)
        noise = temp - np.exp((0.5**2) / 2)
    elif noise_distribution == "gumbel":
        noise = rng.gumbel(loc=0, scale=0.5, size=n)
    else:
        raise ValueError(f"Unsupported noise distribution: {noise_distribution}")
    
    y = deterministic + noise
    data = {f'x{i+1}': X[:, i] for i in range(num_variables)}
    data['Output'] = y
    return pd.DataFrame(data)

def compute_true_values(x, method="quadratic"):
    """
    Computes the true underlying deterministic values for the given x based on the chosen method.
    
    Parameters:
        x (array-like): Predictor values.
        method (str): Either "cosine" or "quadratic".
    
    Returns:
        array-like: The true (deterministic) function values.
        
    For method "quadratic", it returns: x**2 + 0.5 * sin(3*x).
    For method "cosine", it returns: cos(x).
    """
    if method == "quadratic":
        return x**2 + 0.5 * np.sin(3 * x)
    elif method == "cosine":
        return np.cos(x)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'cosine' or 'quadratic'.") 