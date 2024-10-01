import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
import pandas as pd
import time
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))
from shared.cmd_configs import parse_args, default_config, openml_datasets
from shared.benchmark import BenchmarkSEMF, display_results
from semf.semf import SEMF
from semf.preprocessing import DataPreprocessor
from semf import utils
from semf.visualize import (
    visualize_prediction_intervals,
    visualize_prediction_intervals_kde,
    visualize_prediction_intervals_kde_multiple,
    plot_violin,
    get_confidence_intervals,
    plot_confidence_intervals,
    plot_tr_val_metrics,
)

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, "..", "..", "..", "data", "tabular_benchmark")

def train_on_dataset(ds_name, config, SEED=100):
    """
    Train and evaluate the model on a single dataset using the hyperparameters from the command line (locally).

    Parameters:
    - ds_name (str): The name of the dataset.
    - config (SimpleNamespace or dict): Configuration including hyperparameters.
    - SEED (int): The random seed for reproducibility.
    """
    utils.set_seed(SEED)

    if ds_name == "simulate_linear_quadratic":
        from src.experiments.shared.generate_data import generate_data
        data = generate_data(10000, 4)
        column_names = list(data.columns.values)
        outcome_variable = column_names[-1]
    else:
        sub_dir_path = os.path.join(data_dir, ds_name)
        os.makedirs(sub_dir_path, exist_ok=True)
        print(f"Dataset: {ds_name}")
        data, outcome_variable = utils.load_openml_dataset(
            ds_name, dataset_names=openml_datasets, cache_dir=sub_dir_path
        )
        column_names = data.columns.tolist()

    print("Columns in the dataset:", data.columns.tolist())

    if outcome_variable not in data.columns:
        raise ValueError(f"Outcome variable '{outcome_variable}' not found in dataset columns.")

    first_predictor = column_names[0]

    print(f"Dataset shape: {data.shape}")
    print(f"First predictor variable: {first_predictor}")
    print(f"Outcome variable: {outcome_variable}")

    visualize_outcome = False
    if visualize_outcome:
        sns.set_theme(style="whitegrid")
        print(f"Min and Max values of Global Output: {data[outcome_variable].min()}, {data[outcome_variable].max()}")
        plt.figure(figsize=(12, 6))
        sns.histplot(data[outcome_variable], edgecolor=".2", palette="Set3")
        plt.xlabel(outcome_variable, size=16, alpha=0.7)
        plt.ylabel("Frequency", size=16, alpha=0.7)
        plt.savefig('y_dist.png')
        plt.show()

    data_preprocessor = DataPreprocessor(
        data,
        y_col=outcome_variable,
        complete_x_col=first_predictor,
        rate=config.missing_rate,
        train_size=config.training_set_size,
        valid_size=config.valid_set_size,
    )
    data_preprocessor.split_data()
    if config.missing_rate > 0:
        data_preprocessor.generate_missing_values(all_columns=config.missing_all_columns)
    data_preprocessor.scale_data(scale_output=True)

    df_train, df_valid, df_test = data_preprocessor.get_train_valid_test()

    if visualize_outcome:
        df_scaled = pd.concat([df_train, df_valid, df_test])
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.histplot(df_scaled[outcome_variable], edgecolor=".2", palette="Set3")
        plt.xlabel(f"Standardized {outcome_variable}", size=16, alpha=0.7)
        plt.ylabel("Frequency", size=16, alpha=0.7)
        plt.savefig('standardized_y_dist.png')
        plt.show()
    
    utils.print_data_completeness(df_train, df_valid, df_test)

    y_std = data_preprocessor.df_train[outcome_variable].std()
    print(f"- Std observed `y`: {y_std}")
    print(f"- Custom sigma_R: {config.custom_sigma_R}")
    print(f"- Input z_norm_sd: {config.z_norm_sd}")
    print(f"- Initial z_norm_sd: {config.initial_z_norm_sd}")

    num_columns = df_train.drop(columns=outcome_variable).shape[1]
    if config.x_group_size_prop == "all":
        x_group_size = int(np.ceil(1 * num_columns))
    elif config.x_group_size_prop == "half":
        x_group_size = int(np.ceil(0.5 * num_columns))
    else:
        x_group_size = 1
    num_groups = int(np.ceil(num_columns / x_group_size))
    n_nodes = np.array([config.nodes_per_feature] * (num_groups))

    semf = SEMF(
        data_preprocessor,
        R=config.R,
        nodes_per_feature=n_nodes,
        model_class=config.model_class,
        tree_config=config.tree_config,
        nn_config=config.nn_config,
        models_val_split=config.models_val_split,
        parallel_type=config.parallel_type,
        device=config.device,
        n_jobs=config.n_jobs,
        force_n_jobs=config.force_n_jobs,
        max_it=config.max_it,
        stopping_patience=config.stopping_patience,
        stopping_metric=config.stopping_metric,
        custom_sigma_R=config.custom_sigma_R,
        z_norm_sd=config.z_norm_sd,
        initial_z_norm_sd=config.initial_z_norm_sd,
        fixed_z_norm_sd=config.fixed_z_norm_sd,
        return_mean_default=config.return_mean_default,
        mode_type=config.mode_type,
        use_constant_weights=config.use_constant_weights,
        verbose=config.verbose,
        x_group_size=x_group_size,
        seed=config.seed,
        simulator_architecture=config.simulator_architecture,
        simulator_epochs=config.simulator_epochs,
    )

    st = time.time()
    semf.train_semf()
    et = time.time()
    elapsed_time = et - st
    print("Training execution time:", elapsed_time, "seconds")

    optimal_i_value = getattr(semf, "optimal_i", getattr(semf, "i", None))
    plot_optimal_i_value = None
    if optimal_i_value == getattr(semf, "i", None):
        optimal_i_value -= 1
    else:
        plot_optimal_i_value = optimal_i_value

    # plot_tr_val_metrics(semf, optimal_i_value=plot_optimal_i_value)

    print(f"Running benchmark for {ds_name}...")
    if config.return_point_benchmark:
        if config.model_class == "MultiXGBs":
            base_model = "XGB"
        elif config.model_class == "MultiETs":
            base_model = "ET"
        elif config.model_class == "MultiMLPs":
            base_model = "MLP"
        else:
            base_model = "all"

        benchmark = BenchmarkSEMF(
            df_train,
            df_valid,
            df_test,
            y_col=data_preprocessor.y_col,
            missing_rate=config.missing_rate,
            semf_model=semf,
            alpha=config.alpha_certainty,
            knn_neighbors=config.benchmark_knn_neighbors,
            base_model=base_model,
            test_with_wide_intervals=config.test_with_wide_intervals,
            seed=SEED,
            inference_R=config.R_inference,
            tree_n_estimators=config.tree_config["tree_n_estimators"],
            xgb_max_depth=config.tree_config["xgb_max_depth"],
            et_max_depth=config.tree_config["et_max_depth"],
            nn_batch_size=config.nn_config["nn_batch_size"],
            nn_epochs=config.nn_config["nn_epochs"],
            nn_lr=config.nn_config["nn_lr"],
            nn_load_into_memory=config.nn_config["nn_load_into_memory"],
            device=config.device,
            models_val_split=config.models_val_split,
            xgb_patience=config.tree_config["xgb_patience"],
            nn_patience=config.nn_config["nn_patience"]
        )

        results_pointpred = benchmark.run_pointpred()
        print(f"\nResults with {config.missing_rate * 100}% missing data:\n")
        display_results(results_pointpred, sort_descending_by="MAE")

        if config.return_interval_benchmark:
            results_intervals = benchmark.run_intervals()
            display_results(results_intervals, sort_descending_by="CWR")
            # fig = benchmark.plot_predicted_intervals(
            #     semf.x_valid, semf.y_valid, sample_size=100
            # )

    # instance_n = 0
    # preds = semf.infer_semf(semf.x_test.iloc[[instance_n]], return_type="interval")
    # preds = preds.flatten()
    # visualize_prediction_intervals_kde(
    #     y_preds=preds,
    #     y_true=semf.y_test.loc[instance_n].values[0],
    #     central_tendency="mean",
    # )

    # plt_n_instances = 10
    # preds = semf.infer_semf(semf.x_test.iloc[0:plt_n_instances], return_type="interval")
    # actuals = semf.y_test.iloc[0:plt_n_instances].values
    # visualize_prediction_intervals(preds, actuals, central_tendency="mean")

    # plot_violin(y_preds=preds, y_true=semf.y_test.iloc[0:plt_n_instances].values if semf.y_test is not None else None, n_instances=plt_n_instances)

    # plot_confidence_intervals(
    #     get_confidence_intervals(preds, actuals),
    #     n_instances=plt_n_instances,
    #     central_tendency="mean",
    # )

    print("-" * 40)

    try:
        if config.save_models:
            semf.save_semf(data_preprocessor, ds_name, base_dir)
    except NotImplementedError as e:
        print(e)

if __name__ == "__main__":
    args = parse_args()
    datasets_to_run = args.dataset

    for data in datasets_to_run:
        print(data)
        print("Configuration:", default_config)
        train_on_dataset(ds_name=data, config=default_config, SEED=args.seed)
