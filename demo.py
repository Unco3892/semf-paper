#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
import xgboost as xgb # Import XGBoost
from sklearn.metrics import mean_pinball_loss # Import pinball loss
# Import the data generator function
import sys
sys.path.append("src")
from experiments.synthetic.synthetic_data_generator import generate_synthetic_data

# ----------------------------
# Helper functions for metrics
# ----------------------------
def calculate_coverage_width(y_true, y_lower, y_upper):
    """Calculates empirical coverage and average width of prediction intervals."""
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    width = np.mean(y_upper - y_lower)
    return coverage, width

def calculate_crps(y_true, lower, upper):
    """Calculates the Continuous Ranked Probability Score assuming a uniform forecast distribution U(lower, upper)."""
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    d = upper - lower
    # Handle cases where lower == upper (point prediction or zero width)
    # In this case, CRPS simplifies to Absolute Error
    crps_zero_width = np.abs(y_true - lower)
    
    # Calculate CRPS components for non-zero width intervals
    term1_lower = ((lower + upper) / 2 - y_true) - d / 6
    term1_upper = (y_true - (lower + upper) / 2) - d / 6
    term1_within = (((y_true - lower)**2 + (upper - y_true)**2) / (2 * d)) - d / 6
    
    # Combine using np.where
    crps = np.where(np.isclose(d, 0), crps_zero_width,
                        np.where(y_true < lower, term1_lower,
                                 np.where(y_true > upper, term1_upper,
                                          term1_within)))
    return np.mean(crps)

# ----------------------------
# CQR Helper Function
# ----------------------------
def apply_and_evaluate_cqr(y_valid_true, valid_lower_initial, valid_upper_initial,
                           y_test_true, test_lower_initial, test_upper_initial,
                           alpha, model_name, epsilon=1e-8):
    """
    Applies Conformalized Quantile Regression (CQR) and evaluates the results.

    Args:
        y_valid_true (pd.Series or np.ndarray): True target values for validation set.
        valid_lower_initial (np.ndarray): Initial lower quantile preds on validation set.
        valid_upper_initial (np.ndarray): Initial upper quantile preds on validation set.
        y_test_true (pd.Series or np.ndarray): True target values for test set.
        test_lower_initial (np.ndarray): Initial lower quantile preds on test set.
        test_upper_initial (np.ndarray): Initial upper quantile preds on test set.
        alpha (float): Desired miscoverage rate.
        model_name (str): Name of the model for printing results.
        epsilon (float): Small constant for CWR calculation stability.

    Returns:
        tuple: (cqr_lower, cqr_upper, results_dict)
               - cqr_lower: Adjusted lower bounds for the test set.
               - cqr_upper: Adjusted upper bounds for the test set.
               - results_dict: Dictionary containing evaluation metrics.
    """
    print(f"\nApplying CQR and evaluating for {model_name}...") # Add newline for spacing

    # Ensure y_valid and y_test are 1D numpy arrays
    y_valid_arr = y_valid_true.values.squeeze() if isinstance(y_valid_true, pd.Series) else np.asarray(y_valid_true).squeeze()
    y_test_arr = y_test_true.values.squeeze() if isinstance(y_test_true, pd.Series) else np.asarray(y_test_true).squeeze()
    n_cal = len(y_valid_arr)

    # 1. Calculate conformity scores on the validation set
    conformity_scores = np.maximum(valid_lower_initial - y_valid_arr, y_valid_arr - valid_upper_initial)
    print(f"Calculated {len(conformity_scores)} conformity scores for {model_name} calibration.")

    # 2. Find the CQR quantile correction factor 'q_hat'
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(conformity_scores, q_level, method="higher")
    print(f"{model_name} CQR quantile correction (q_hat) at level {q_level:.4f}: {q_hat:.4f}")

    # 3. Adjust the original test intervals using q_hat
    cqr_lower = test_lower_initial - q_hat
    cqr_upper = test_upper_initial + q_hat

    # 4. Evaluate CQR intervals on the test set
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0
    cqr_coverage, cqr_width = calculate_coverage_width(y_test_arr, cqr_lower, cqr_upper)
    cqr_cwr = cqr_coverage / (cqr_width + epsilon)
    cqr_pinball_lower = mean_pinball_loss(y_test_arr, cqr_lower, alpha=lower_quantile)
    cqr_pinball_upper = mean_pinball_loss(y_test_arr, cqr_upper, alpha=upper_quantile)
    cqr_mean_pinball = (cqr_pinball_lower + cqr_pinball_upper) / 2
    cqr_crps = calculate_crps(y_test_arr, cqr_lower, cqr_upper)

    results = {
        "coverage": cqr_coverage,
        "width": cqr_width,
        "cwr": cqr_cwr,
        "pinball": cqr_mean_pinball,
        "crps": cqr_crps
    }

    print(f"--- {model_name} + CQR Results ---")
    print(f"Coverage: {results['coverage']:.3f}")
    print(f"Avg. Width: {results['width']:.3f}")
    print(f"CWR: {results['cwr']:.3f}")
    print(f"Mean Pinball Loss: {results['pinball']:.3f}")
    print(f"Mean CRPS (Uniform): {results['crps']:.3f}")
    print("--------------------------")

    return cqr_lower, cqr_upper, results

# ----------------------------
# Data Loading using Your Preprocessing Module
# ----------------------------
from semf.preprocessing import DataPreprocessor

# Use the imported generator function
seed = 42
np.random.seed(seed)
n = 1000
num_vars = 2 # Set number of features
df = generate_synthetic_data(n=n, num_variables=num_vars, seed=seed, method="cosine") # Example: using cosine method
print(df)

# Instantiate the DataPreprocessor - ** IMPORTANT: Use 'Output' as y_col **
preprocessor = DataPreprocessor(df, y_col="Output", train_size=0.7, valid_size=0.15)
preprocessor.split_data()
preprocessor.scale_data(scale_output=True)
df_train, df_valid, df_test = preprocessor.get_train_valid_test()
X_train, y_train = preprocessor.split_X_y(df_train)
X_valid, y_valid = preprocessor.split_X_y(df_valid)
X_test, y_test = preprocessor.split_X_y(df_test)

# ----------------------------
# SEMF Wrapper
# ----------------------------
# Note: Your SEMF (located in semf.py) does not currently implement fit and predict.
# We wrap it here to provide a minimal interface.
from semf.semf import SEMF  # import your SEMF implementation

class SemfWrapper:
    """
    A thin wrapper around SEMF that exposes fit() and predict() methods.
    Here, fit() simply calls train_semf() and predict() wraps infer_semf().
    """
    def __init__(self, semf_model):
        self.semf = semf_model

    def fit(self):
        # Call your internal SEMF training routine.
        self.semf.train_semf()

    def predict(self, X, return_interval=False):
        # Infer predictions.
        # When return_interval is True, we expect SEMF.infer_semf to return a 2D array
        # with shape (n_samples, n_draws). We then compute the lower and upper bounds
        # (here, 2.5th and 97.5th percentiles for a 95% interval).
        if return_interval:
            result = self.semf.infer_semf(X, return_type="interval")
            if isinstance(result, np.ndarray):
                lower = np.percentile(result, 2.5, axis=1)
                upper = np.percentile(result, 97.5, axis=1)
                return lower, upper
            elif isinstance(result, tuple):
                return result
            else:
                raise ValueError("Unexpected output type from SEMF.infer_semf")
        else:
            return self.semf.infer_semf(X, return_type="point")

# ----------------------------
# Instantiate SEMF with Modified z_norm_sd and R for debugging
# ----------------------------
# Try increasing z_norm_sd and R
# z_norm_sd_value = 1  # increase from 0.01 to 0.1 (experiment with higher values)
z_norm_sd_value = 'train_residual_models'
R_inference = 50 
# increase the number of inference samples for better percentile estimation

# Ensure nodes_per_feature matches num_vars
nodes_per_feat = np.array([20] * num_vars) # e.g., [20, 20] if num_vars is 2

semf_model = SEMF(
    data_preprocessor=preprocessor,
    R=R_inference,
    # nodes_per_feature=np.array([20, 20]), # Old way
    nodes_per_feature=nodes_per_feat, # Use dynamically created array
    model_class="MultiXGBs",
    z_norm_sd=z_norm_sd_value,
    return_mean_default=True,
    stopping_metric="RMSE",
    tree_config = {"tree_n_estimators":100, "xgb_max_depth":None, "xgb_patience":0, "et_max_depth":10},
    models_val_split=0.15,
    stopping_patience=5,
    max_it=100,           # For a fast demo; adjust as needed
    verbose=False,
    seed=seed,
    # device='cpu'
    # device='gpu'
)
semf_wrapper = SemfWrapper(semf_model)

# # ----------------------------
# # Train and Predict with SEMF
# # ----------------------------
print("Training SEMF model...")
semf_wrapper.fit()
print("SEMF training complete.")

print("Predicting intervals with SEMF...")
# Predict intervals on the test set.
# We expect infer_semf to return a tuple (lower_bounds, upper_bounds)
semf_intervals = semf_wrapper.predict(X_test, return_interval=True)
if isinstance(semf_intervals, tuple):
    semf_lower, semf_upper = semf_intervals
else:
    raise ValueError("SEMF.predict did not return a tuple of intervals.")

# Convert y_test to a 1D NumPy array for proper comparison
y_test_arr = y_test.values.squeeze()

# Calculate interval metrics for SEMF
semf_coverage, semf_width = calculate_coverage_width(y_test_arr, semf_lower, semf_upper)
epsilon = 1e-8 # For CWR calculation
semf_cwr = semf_coverage / (semf_width + epsilon)
# Define alpha and quantiles for pinball loss
alpha = 0.05 # Desired miscoverage rate (e.g., for 95% target coverage)
lower_quantile = alpha / 2.0
upper_quantile = 1.0 - alpha / 2.0
print(f"Using alpha = {alpha} ({(1-alpha)*100:.1f}% coverage target)")
print(f"Lower quantile: {lower_quantile}, Upper quantile: {upper_quantile}")
# Calculate Pinball Loss for SEMF Raw
semf_pinball_lower = mean_pinball_loss(y_test_arr, semf_lower, alpha=lower_quantile)
semf_pinball_upper = mean_pinball_loss(y_test_arr, semf_upper, alpha=upper_quantile)
semf_mean_pinball = (semf_pinball_lower + semf_pinball_upper) / 2
# Calculate CRPS for SEMF Raw
semf_crps = calculate_crps(y_test_arr, semf_lower, semf_upper)

print("--- SEMF (Raw Percentiles) Results ---")
print(f"Coverage: {semf_coverage:.3f}")
print(f"Avg. Width: {semf_width:.3f}")
print(f"CWR: {semf_cwr:.3f}")
print(f"Mean Pinball Loss: {semf_mean_pinball:.3f}")
print(f"Mean CRPS (Uniform): {semf_crps:.3f}") # Add CRPS
print("-------------------------------------")

# ----------------------------
# Apply CQR to SEMF
# ----------------------------
# Predict intervals on the validation set for CQR calibration
print("\nPredicting intervals on validation set for SEMF CQR calibration...") # Add newline
semf_valid_lower_initial, semf_valid_upper_initial = semf_wrapper.predict(X_valid, return_interval=True)

# Apply CQR using the new function
# Note: We pass y_valid and y_test directly, the function handles conversion to numpy array
semf_cqr_lower, semf_cqr_upper, semf_cqr_results = apply_and_evaluate_cqr(
    y_valid_true=y_valid,
    valid_lower_initial=semf_valid_lower_initial,
    valid_upper_initial=semf_valid_upper_initial,
    y_test_true=y_test,
    test_lower_initial=semf_lower, # From initial SEMF test prediction
    test_upper_initial=semf_upper, # From initial SEMF test prediction
    alpha=alpha,
    model_name="SEMF"
    # Epsilon defaults to 1e-8 in the function
)

# ----------------------------
# Baseline using XGBoost (Point Prediction) + MAPIE
# ----------------------------
print("Training XGBoost + MAPIE Baseline...")

# Instantiate XGBoost Regressor
xgb_point_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed, n_estimators=100, xgb_max_depth=None)

# Use MAPIE with cross-validation (CV+) for potentially better calibration
mapie_xgb = MapieRegressor(estimator=xgb_point_model, method='base', cv=5) # Use 'plus' method, cv=5 enables residual calculation on out-of-fold predictions

# Fit MAPIE on training data
mapie_xgb.fit(X_train, y_train.values.ravel()) # y_train needs to be 1D

# Predict intervals on the test set
y_pred_mapie_xgb, mapie_xgb_interval = mapie_xgb.predict(X_test, alpha=alpha)

# Extract lower and upper bounds
mapie_xgb_lower = mapie_xgb_interval[:, 0, 0]
mapie_xgb_upper = mapie_xgb_interval[:, 1, 0]

# Calculate coverage and width
mapie_xgb_coverage, mapie_xgb_width = calculate_coverage_width(y_test_arr, mapie_xgb_lower, mapie_xgb_upper)
mapie_xgb_cwr = mapie_xgb_coverage / (mapie_xgb_width + epsilon)
# Calculate Pinball Loss for XGBoost + MAPIE
mapie_xgb_pinball_lower = mean_pinball_loss(y_test_arr, mapie_xgb_lower, alpha=lower_quantile)
mapie_xgb_pinball_upper = mean_pinball_loss(y_test_arr, mapie_xgb_upper, alpha=upper_quantile)
mapie_xgb_mean_pinball = (mapie_xgb_pinball_lower + mapie_xgb_pinball_upper) / 2
# Calculate CRPS for XGBoost + MAPIE
mapie_xgb_crps = calculate_crps(y_test_arr, mapie_xgb_lower, mapie_xgb_upper)

print("--- XGBoost + MAPIE Results ---")
print(f"Coverage: {mapie_xgb_coverage:.3f}")
print(f"Avg. Width: {mapie_xgb_width:.3f}")
print(f"CWR: {mapie_xgb_cwr:.3f}")
print(f"Mean Pinball Loss: {mapie_xgb_mean_pinball:.3f}")
print(f"Mean CRPS (Uniform): {mapie_xgb_crps:.3f}") # Add CRPS
print("-------------------------------")

# ----------------------------
# Baseline using Quantile XGBoost + Manual CQR
# ----------------------------
print("Training Quantile XGBoost + CQR Baseline...")

## NOTE: With early stopping that we use in our SEMF technique, the result actually worsens, so we omitted from using it

# Train lower quantile model
xgb_lower_model = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=lower_quantile,
    random_state=seed,
    n_estimators=100
)
xgb_lower_model.fit(X_train, y_train)
print("Lower quantile XGBoost model trained.")

# Train upper quantile model
xgb_upper_model = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=upper_quantile,
    random_state=seed,
    n_estimators=100
)
xgb_upper_model.fit(X_train, y_train)
print("Upper quantile XGBoost model trained.")

# Predict initial quantiles on validation and test sets
print("\nPredicting initial quantiles for XGBoost CQR...") # Add newline
xgb_valid_lower_initial = xgb_lower_model.predict(X_valid)
xgb_valid_upper_initial = xgb_upper_model.predict(X_valid)
xgb_test_lower_initial = xgb_lower_model.predict(X_test)
xgb_test_upper_initial = xgb_upper_model.predict(X_test)

# Apply CQR using the new function
xgb_cqr_lower, xgb_cqr_upper, xgb_cqr_results = apply_and_evaluate_cqr(
    y_valid_true=y_valid,
    valid_lower_initial=xgb_valid_lower_initial,
    valid_upper_initial=xgb_valid_upper_initial,
    y_test_true=y_test,
    test_lower_initial=xgb_test_lower_initial,
    test_upper_initial=xgb_test_upper_initial,
    alpha=alpha,
    model_name="Quantile XGBoost"
    # Epsilon defaults to 1e-8 in the function
)

# ----------------------------
# Visualize Prediction Intervals (Sorted)
# ----------------------------
print("\nVisualizing prediction intervals on sorted test data...")

# Get sorting indices based on true values
sort_indices = np.argsort(y_test_arr)

# Sort true values and predictions
y_test_sorted = y_test_arr[sort_indices]
semf_cqr_lower_sorted = semf_cqr_lower[sort_indices]
semf_cqr_upper_sorted = semf_cqr_upper[sort_indices]
mapie_xgb_lower_sorted = mapie_xgb_lower[sort_indices]
mapie_xgb_upper_sorted = mapie_xgb_upper[sort_indices]
xgb_cqr_lower_sorted = xgb_cqr_lower[sort_indices]
xgb_cqr_upper_sorted = xgb_cqr_upper[sort_indices]

# Create x-axis for plotting (just sequence for sorted data)
x_plot = range(len(y_test_sorted))

plt.figure(figsize=(12, 7))

# Plot true values
plt.scatter(x_plot, y_test_sorted, color='black', label='True Values', s=10, zorder=5)

# Plot SEMF + CQR intervals
plt.fill_between(x_plot, semf_cqr_lower_sorted, semf_cqr_upper_sorted,
                 color='red', alpha=0.3, label=f'SEMF + CQR ({(1-alpha)*100:.0f}%)')

# Plot XGBoost + MAPIE intervals
plt.fill_between(x_plot, mapie_xgb_lower_sorted, mapie_xgb_upper_sorted,
                 color='green', alpha=0.3, label=f'XGBoost + MAPIE ({(1-alpha)*100:.0f}%)')

# Plot Quantile XGBoost + CQR intervals
plt.fill_between(x_plot, xgb_cqr_lower_sorted, xgb_cqr_upper_sorted,
                 color='blue', alpha=0.3, label=f'Quantile XGBoost + CQR ({(1-alpha)*100:.0f}%)')

plt.xlabel("Test Samples (Sorted by True Value)")
plt.ylabel("Target Value")
plt.title("Comparison of Conformalized Prediction Intervals on Test Data")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# # ----------------------------
# # Train and Predict with SEMF using MultiMLPs
# # ----------------------------
# print("\nTraining SEMF with MLP model...")

try:
    import torch
    device_used = "gpu" if torch.cuda.is_available() else "cpu"
except ImportError:
    device_used = "cpu"

# Define configuration for the neural network
nn_config = {"nn_batch_size": None, "nn_epochs": 1000, "nn_lr": 0.001, "nn_patience": 10} # Note that we set `nn_batch_size` to None here so we train everything in memory
val_split = 0.15

# Instantiate SEMF with MLP configuration (using MLP base)
semf_mlp_model = SEMF(
    data_preprocessor=preprocessor,
    R=R_inference,
    nodes_per_feature=nodes_per_feat,
    model_class="MultiMLPs",  # Using MLP for SEMF
    nn_config= nn_config,
    models_val_split=val_split,
    stopping_patience=5,
    max_it=100,
    verbose=False,
    seed=seed,
    z_norm_sd=z_norm_sd_value,
    return_mean_default=True,
    device=device_used
)
semf_mlp_wrapper = SemfWrapper(semf_mlp_model)

print("Training SEMF with MLP...")
semf_mlp_wrapper.fit()
print("SEMF with MLP training complete.")

print("Predicting intervals with SEMF with MLP...")
mlp_intervals = semf_mlp_wrapper.predict(X_test, return_interval=True)
if isinstance(mlp_intervals, tuple):
    mlp_lower, mlp_upper = mlp_intervals
else:
    raise ValueError("SEMF MLP.predict did not return a tuple of intervals.")

print("Computing point predictions with SEMF with MLP...")
mlp_point_preds = semf_mlp_wrapper.predict(X_test, return_interval=False)
# print("SEMF with MLP point prediction sample:", mlp_point_preds[:5])

# ----------------------------
# Apply SQR Conformal Prediction to SEMF with MLP
# ----------------------------
print("\nPredicting intervals on validation set for SEMF MLP SQR conformal calibration...")
mlp_valid_lower_initial, mlp_valid_upper_initial = semf_mlp_wrapper.predict(X_valid, return_interval=True)

mlp_cqr_lower, mlp_cqr_upper, mlp_cqr_results = apply_and_evaluate_cqr(
    y_valid_true=y_valid,
    valid_lower_initial=mlp_valid_lower_initial,
    valid_upper_initial=mlp_valid_upper_initial,
    y_test_true=y_test,
    test_lower_initial=mlp_lower,
    test_upper_initial=mlp_upper,
    alpha=alpha,
    model_name="SEMF MultiMLPs"
)

# ----------------------------
# Baseline using MLP (from semf.models) for Point Prediction
print("\nTraining baseline MLP (Point Prediction) using semf.models.MLP...")
from semf.models import MLP, QNN  # Import baseline models from semf/models
# Determine device based on availability (using semf_mlp_model.device if defined or torch.cuda)
# out_memory_batch_size = 32 if device_used == "cpu" else None


baseline_mlp_model = MLP(input_size=X_train.shape[1], output_size=1, device=device_used)
baseline_mlp_model.train_model(
    inputs=X_train,
    outputs=y_train,
    batch_size=nn_config["nn_batch_size"],
    epochs=nn_config["nn_epochs"],
    lr=nn_config["nn_lr"],
    load_into_memory=True,
    nn_patience=nn_config["nn_patience"],
    val_split=val_split,
    verbose=False
)
baseline_mlp_preds = baseline_mlp_model.predict(X_test)
# print("Baseline MLP point prediction sample:", baseline_mlp_preds[:5])

# ----------------------------
# Baseline using MLP for Simultaneous Quantile Regression (SQR) using semf.models.QNN
print("\nTraining baseline MLP for Simultaneous Quantile Regression (SQR) using semf.models.QNN...")
baseline_mlp_sqr = QNN(input_size=X_train.shape[1], output_size=1, device=device_used)
baseline_mlp_sqr.train_model(
    X_train,
    y_train,
    batch_size=nn_config["nn_batch_size"],
    epochs=nn_config["nn_epochs"],
    lr=nn_config["nn_lr"],
    load_into_memory=True,
    nn_patience=nn_config["nn_patience"],
    val_split=val_split,
    verbose=False
)
baseline_mlp_sqr_preds = baseline_mlp_sqr.predict(X_test, quantiles=[lower_quantile, upper_quantile])
baseline_mlp_lower = baseline_mlp_sqr_preds[:, 0]
baseline_mlp_upper = baseline_mlp_sqr_preds[:, 1]
# print("Baseline MLP SQR prediction sample (lower):", baseline_mlp_lower[:5])
# print("Baseline MLP SQR prediction sample (upper):", baseline_mlp_upper[:5])

print("\nPredicting intervals on validation set for baseline MLP SQR conformal calibration...")
baseline_mlp_sqr_valid_preds = baseline_mlp_sqr.predict(X_valid, quantiles=[lower_quantile, upper_quantile])
baseline_mlp_valid_lower = baseline_mlp_sqr_valid_preds[:, 0]
baseline_mlp_valid_upper = baseline_mlp_sqr_valid_preds[:, 1]
baseline_mlp_cqr_lower, baseline_mlp_cqr_upper, baseline_mlp_cqr_results = apply_and_evaluate_cqr(
    y_valid_true=y_valid,
    valid_lower_initial=baseline_mlp_valid_lower,
    valid_upper_initial=baseline_mlp_valid_upper,
    y_test_true=y_test,
    test_lower_initial=baseline_mlp_lower,
    test_upper_initial=baseline_mlp_upper,
    alpha=alpha,
    model_name="Baseline MLP SQR"
)

# ----------------------------
# Baseline using MLP with MAPIE for Quantile Prediction (with skorch wrapper)
# ----------------------------
# Baseline using PyTorch MLP with MAPIE for Quantile Prediction
print("\nTraining PyTorch MLP with MAPIE for quantile prediction...")
from semf.models import NeuralNetwork
import skorch
from skorch import NeuralNetRegressor

# # Create a custom NeuralNetRegressor that flattens predictions to 1D
class FlattenNeuralNetRegressor(NeuralNetRegressor):
    def predict(self, X):
        """Flatten predictions to 1D for compatibility with MAPIE"""
        # Call original predict method
        predictions = super().predict(X)
        # Flatten to 1D if needed (2D array with shape (n_samples, 1) -> (n_samples,))
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            return predictions.flatten()
        return predictions

# Convert DataFrame to numpy first (skorch can handle numpy better than pandas)
X_train_np = X_train.to_numpy().astype(np.float32)  # Convert to float32
y_train_np = y_train.to_numpy().ravel().astype(np.float32)  # Convert to float32
X_test_np = X_test.to_numpy().astype(np.float32)  # Convert to float32
X_valid_np = X_valid.to_numpy().astype(np.float32)  # Also for validation
y_test_arr = y_test.values.squeeze()  # Ensure this is defined

# Create skorch wrapper with float32 numpy data - use our flattening version
skorch_mlp = FlattenNeuralNetRegressor(
    module=NeuralNetwork,
    module__input_size=X_train_np.shape[1],
    module__output_size=1,
    optimizer=torch.optim.Adam,
    lr=nn_config["nn_lr"],
    max_epochs=nn_config["nn_epochs"],
    batch_size= len(X_train_np) if nn_config["nn_batch_size"] is None else nn_config["nn_batch_size"],
    train_split=skorch.dataset.ValidSplit(val_split),
    callbacks=[skorch.callbacks.EarlyStopping(patience=nn_config["nn_patience"])],
    device="cuda" if device_used == "gpu" else "cpu",
    verbose=0
)

# Use with MAPIE on numpy data
mapie_skorch_mlp = MapieRegressor(estimator=skorch_mlp, method="base", cv=5)
mapie_skorch_mlp.fit(X_train_np, y_train_np)

skorch_mlp_point, skorch_mlp_interval = mapie_skorch_mlp.predict(X_test_np, alpha=alpha)
skorch_mlp_lower = skorch_mlp_interval[:, 0, 0]
skorch_mlp_upper = skorch_mlp_interval[:, 1, 0]

# Calculate metrics
skorch_mlp_coverage, skorch_mlp_width = calculate_coverage_width(y_test_arr, skorch_mlp_lower, skorch_mlp_upper)
skorch_mlp_cwr = skorch_mlp_coverage / (skorch_mlp_width + epsilon)
skorch_mlp_pinball_lower = mean_pinball_loss(y_test_arr, skorch_mlp_lower, alpha=lower_quantile)
skorch_mlp_pinball_upper = mean_pinball_loss(y_test_arr, skorch_mlp_upper, alpha=upper_quantile)
skorch_mlp_mean_pinball = (skorch_mlp_pinball_lower + skorch_mlp_pinball_upper) / 2
skorch_mlp_crps = calculate_crps(y_test_arr, skorch_mlp_lower, skorch_mlp_upper)

print("--- MLP + skorch + MAPIE Results ---")
print(f"Coverage: {skorch_mlp_coverage:.3f}")
print(f"Avg. Width: {skorch_mlp_width:.3f}")
print(f"CWR: {skorch_mlp_cwr:.3f}")
print(f"Mean Pinball Loss: {skorch_mlp_mean_pinball:.3f}")
print(f"Mean CRPS (Uniform): {skorch_mlp_crps:.3f}")
print("-------------------------------")