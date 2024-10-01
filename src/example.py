import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from semf.semf import SEMF
from semf.preprocessing import DataPreprocessor
from semf import utils
from semf.visualize import (
    visualize_prediction_intervals,
    visualize_prediction_intervals_kde,
    plot_violin,
    get_confidence_intervals,
    plot_confidence_intervals,
    plot_tr_val_metrics,
)
from experiments.shared.benchmark import BenchmarkSEMF, display_results
from experiments.shared.generate_data import generate_data  # Import the data generation function

# Set the style for seaborn plots
sns.set(style="whitegrid")

if __name__ == "__main__":
    # Step 1: Generate synthetic data
    # --------------------------------
    # Here we generate a dataset with 10000 observations and 4 predictor variables.
    utils.set_seed(0)
    n = 10000
    num_variables = 3
    data = generate_data(n, num_variables)
    
    # Print the shape and the first few rows of the generated data to understand its structure
    print("Shape of the data is: ", data.shape)
    print(data.head())

    # Define the input features (X) and the target variable (y)
    X = data.drop('Output', axis=1)
    y = data['Output']

    # Step 2: Split the data into training, validation, and test sets
    # -----------------------------------------------------------------
    train_size = 0.7
    valid_size = 0.15
    test_size = 0.15

    X_train, X_temp, y_train, y_temp = utils.train_test_split(X, y, train_size=train_size)
    X_valid, X_test, y_valid, y_test = utils.train_test_split(X_temp, y_temp, train_size=valid_size / (valid_size + test_size))

    # Combine the splits into DataFrames
    df_train = pd.concat([X_train, y_train], axis=1)
    df_valid = pd.concat([X_valid, y_valid], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Step 3: Initialize the data preprocessor
    # ------------------------------------------
    # The DataPreprocessor will handle missing data, scaling, and data splitting.
    data_preprocessor = DataPreprocessor(
        data,
        y_col='Output',
        complete_x_col='x1',
        rate=0,  # Assuming no missing data rate for this example
        train_size=train_size,
        valid_size=valid_size
    )
    data_preprocessor.split_data()
    data_preprocessor.scale_data(scale_output=True)
    # Generate missing if interesting
    data_preprocessor.generate_missing_values(all_columns=False)

    # Retrieve the processed data splits (not necessary but optional especially when missing_rate > 0)
    df_train, df_valid, df_test = data_preprocessor.get_train_valid_test()

    # Step 4: Initialize and train the SEMF model
    # ---------------------------------------------
    # SEMF (Supervised Expectation Maximization Framework) is initialized with various hyperparameters and configuration settings.
    semf = SEMF(
        data_preprocessor,
        # The number of sampling operations $R$
        R=5,
        # The number of nodes per feature group corresponding to $m_k$ (2 per for each input $k$)
        nodes_per_feature=np.array([2] * 4),
        # The model class to be used for the simulator (`MultiETs` or `MultiMLPs`` are the other options)
        model_class='MultiXGBs',
        # The configuration setting to be used by both `MultiETs` and `MultiXGBs` models
        tree_config={"tree_n_estimators": 100, "xgb_max_depth": None, "xgb_patience": 10, "et_max_depth": 10},
        # The configuration setting to be used by the `MultiMLPs` model
        nn_config={"nn_batch_size": None, "nn_load_into_memory": True, "nn_epochs": 1000, "nn_lr": 0.001, "nn_patience": 100},
        # The proportion of the training data to be used for validating SEMF and early stopping
        models_val_split=0.15,
        # The parallelization type to be for training the $p_\phi$ models
        parallel_type="semf_joblib",
        # The device (gpu is supported `MultiMLPs` and also `MultiXGBs` but the latter is not ideal) to be used for training the $p_\phi$ and $p_\theta$ models
        device="cpu",
        # The number of jobs to be used for parallelization of the $p_\phi$ models. $\phi_theta$ is automatically parallelized
        n_jobs=2,
        # Whether to force the number of jobs which has to be set to `True` if `n_jobs` is set to `1` and device is `gpu` for faster training of `MultiMLPs`
        force_n_jobs=False,
        # The maximum number of iterations to be used for training the SEMF model, although early stopping almost always kicks in so it can be arbitrarily large
        max_it=500,
        # The patience to be used for early stopping based on the validation set
        stopping_patience=10,
        # The stopping metric to be used for early stopping based on the validation set (other option is `MAE`)
        stopping_metric="RMSE",
        # Fixing the $\sigma^{*})^2$ for the $p_\theta$ models (Eq. 15)
        custom_sigma_R=None,
        # Fixing the $\sigma_{m_k}$ that is always done in our experiments
        z_norm_sd=0.1,
        # The initial $\sigma_{m_k}$ can be different than the fixing value of `z_norm_sd`
        initial_z_norm_sd=None,
        # The point prediction is done via the mean values, but it can also be done via other inferences
        return_mean_default=True,
        # If `return_mean_default` is set to `False`, the mode type to be used for $\hat{y}` (other options are `exact`, `approximate` and `scipy.stats.mode`) or `quantile_50th` median point prediction
        mode_type="approximate",
        # Can fix the weights for debugging instead of letting them be computed via Eq. 6
        use_constant_weights=False,
        # Printing some options for debugging including
        verbose=False,
        # How many columns of the dataset to be grouped together, other options than 1 are not fully supported
        x_group_size=1,
        # The random seed to be used for reproducibility (for all the models and the data splits)
        seed=0,
        # The architecture of the simulator ($p_\xi$) of missing values
        simulator_architecture=[{"units": 100, "activation": "selu"}],
        # The number of epochs to be used for training the simulator of missing values
        simulator_epochs=100
    )

    # Train the SEMF model
    semf.train_semf()

    # Plot the training and validation metrics
    optimal_i_value = getattr(semf, "optimal_i", getattr(semf, "i", None))
    plot_optimal_i_value = None
    if optimal_i_value == getattr(semf, "i", None):
        optimal_i_value -= 1
    else:
        plot_optimal_i_value = optimal_i_value

    plot_tr_val_metrics(semf, optimal_i_value=plot_optimal_i_value)

    # Step 5: Benchmark the SEMF model
    # -----------------------------------
    # Benchmarking the model's performance using various metrics and methods.
    benchmark = BenchmarkSEMF(
        df_train,
        df_valid,
        df_test,
        y_col='Output',
        missing_rate=0,
        semf_model=semf,
        alpha=0.05,
        knn_neighbors=5,
        base_model='XGB',
        test_with_wide_intervals=True,
        seed=0,
        inference_R=50,
        tree_n_estimators=100,
        xgb_max_depth=None,
        et_max_depth=10,
        nn_batch_size=None,
        nn_epochs=1000,
        nn_lr=0.001,
        nn_load_into_memory=True,
        device="cpu",
        models_val_split=0.15,
        xgb_patience=10,
        nn_patience=100
    )

    # Run point predictions
    results_pointpred = benchmark.run_pointpred()
    print("\nResults with 0% missing data:\n")
    display_results(results_pointpred, sort_descending_by="MAE")

    # Run interval predictions for benchmarking
    results_intervals = benchmark.run_intervals()
    display_results(results_intervals, sort_descending_by="CWR")

    # Plot predicted intervals
    fig = benchmark.plot_predicted_intervals(
        semf.x_valid, semf.y_valid, sample_size=100
    )
    plt.show()

    # Step 6: Visualize the predictions and intervals
    # -------------------------------------------------
    instance_n = 0
    preds = semf.infer_semf(semf.x_test.iloc[[instance_n]], return_type="interval")
    preds = preds.flatten()
    visualize_prediction_intervals_kde(
        y_preds=preds,
        y_true=semf.y_test.loc[instance_n].values[0],
        central_tendency="mean",
    )

    plt_n_instances = 10
    preds = semf.infer_semf(semf.x_test.iloc[0:plt_n_instances], return_type="interval")
    actuals = semf.y_test.iloc[0:plt_n_instances].values
    visualize_prediction_intervals(preds, actuals, central_tendency="mean")

    plot_violin(y_preds=preds, y_true=semf.y_test.iloc[0:plt_n_instances].values if semf.y_test is not None else None, n_instances=plt_n_instances)

