import wandb
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
import time
import shutil
import sys
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
    get_confidence_intervals,
    plot_confidence_intervals,
    plot_tr_val_metrics,
)
from dotenv import load_dotenv
load_dotenv()

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, "..", "..", "..", "data", "tabular_benchmark")

def train_on_dataset_wandb(config):
    """
    Train and evaluate the model on a single OpenML dataset using the hyperparameters from the wandb configuration.

    Parameters:
    - config (SimpleNamespace or dict): Configuration including hyperparameters.
    """
    utils.set_seed(config.seed)

    try:
        if wandb.run is None:
            wandb.init(config=config)
        else:
            wandb.config.update(config)
    except Exception as e:
        raise Exception(f"Error initializing wandb for dataset {config.dataset}: {e}")

    config = wandb.config
    print(f"Model configuration: {config}")
    ds_name = wandb.config["dataset"]
    print(f"Processing dataset: {ds_name}")

    try:
        sub_dir_path = os.path.join(data_dir, ds_name)
        os.makedirs(sub_dir_path, exist_ok=True)
        data, outcome_variable = utils.load_openml_dataset(
            ds_name, dataset_names=openml_datasets, cache_dir=sub_dir_path
        )
        column_names = data.columns.tolist()
        first_predictor = column_names[0]
        outcome_variable = column_names[-1]
    except Exception as e:
        raise Exception(f"Error downloading dataset {ds_name}: {e}")

    data_preprocessor = DataPreprocessor(
        data,
        y_col=outcome_variable,
        train_size=config.training_set_size,
        valid_size=config.valid_set_size,
        seed=config.seed
    )
    data_preprocessor.split_data()
    data_preprocessor.scale_data(scale_output=True)
    df_train, df_valid, df_test = data_preprocessor.get_train_valid_test()

    y_std = data_preprocessor.df_train[outcome_variable].std()

    num_columns = df_train.drop(columns=outcome_variable).shape[1]
    if config.x_group_size_prop == "all":
        x_group_size = int(np.ceil(1 * num_columns))
    elif config.x_group_size_prop == "half":
        x_group_size = int(np.ceil(0.5 * num_columns))
    else:
        x_group_size = 1
    num_groups = int(np.ceil(num_columns / x_group_size))
    n_nodes = np.array([config.nodes_per_feature] * (num_groups))

    config.nn_config["nn_epochs"] = config.nn_n_epochs

    try:
        st = time.time()
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
            seed=config.seed
        )

        semf.train_semf()
        et = time.time()
        elapsed_time = et - st
        print("Training execution time:", elapsed_time, "seconds")
    except Exception as e:
        raise Exception(f"Error running SEMF for dataset {ds_name}: {e}")

    optimal_i_value = getattr(semf, "optimal_i", getattr(semf, "i", None))
    plot_optimal_i_value = None
    if optimal_i_value == getattr(semf, "i", None):
        optimal_i_value -= 1
    else:
        plot_optimal_i_value = optimal_i_value

    os.makedirs("plots", exist_ok=True)

    fig_path = f"plots/tr_val_metrics_{ds_name}.png"
    fig = plot_tr_val_metrics(
        semf,
        optimal_i_value=plot_optimal_i_value,
        return_fig=True,
        save_fig=True,
        fig_path=fig_path,
    )
    wandb.log({"tr_val_metrics": wandb.Image(fig_path)}, commit=False)

    results_pointpred = None
    results_intervals = None

    print(f"Running benchmark for {ds_name}...")
    try:
        if isinstance(config.tree_config, str):
            import json
            tree_config = json.loads(config.tree_config)
        else:
            tree_config = config.tree_config

        if isinstance(config.nn_config, str):
            import json
            nn_config = json.loads(config.nn_config)
        else:
            nn_config = config.nn_config

        if config.return_point_benchmark:
            if config.model_class == "MultiXGBs":
                base_model = "XGB"
            elif config.model_class == "MultiETs":
                base_model = "ET"
            elif config.model_class == "MultiMLPs":
                base_model = "MLP"
            if config.benchmark_all:
                base_model = "all"

            benchmark = BenchmarkSEMF(
                df_train,
                df_valid,
                df_test,
                y_col=data_preprocessor.y_col,
                semf_model=semf,
                alpha=config.alpha_certainty,
                base_model=base_model,
                test_with_wide_intervals=config.test_with_wide_intervals,
                seed=config.seed,
                inference_R=config.R_inference,
                tree_n_estimators=tree_config["tree_n_estimators"],
                xgb_max_depth=tree_config["xgb_max_depth"],
                xgb_patience=tree_config["xgb_patience"],
                et_max_depth=tree_config["et_max_depth"],
                nn_batch_size=config.nn_config["nn_batch_size"],
                nn_epochs=config.nn_config["nn_epochs"],
                nn_lr=config.nn_config["nn_lr"],
                nn_load_into_memory=config.nn_config["nn_load_into_memory"],
                device=config.device,
                models_val_split=config.models_val_split,
                nn_patience=config.nn_config["nn_patience"],
                # baseline_interval_method=config.baseline_interval_method,
            )
            
            results_pointpred = benchmark.run_pointpred()
            print("\nResults:\n")
            display_results(results_pointpred, sort_descending_by="MAE", include_imputation=False)

            wandb.log(utils.format_model_metrics(results_pointpred), commit=False)

            if config.return_interval_benchmark:
                results_intervals = benchmark.run_intervals()
                display_results(results_intervals, sort_descending_by="CWR", include_imputation=False)
                wandb.log(utils.format_model_metrics(results_intervals), commit=False)
    except Exception as e:
        raise Exception(f"Error during benchmarking for dataset {ds_name}: {e}")
    print("Run completed")

    # instance_n = 0
    # preds = semf.infer_semf(semf.x_test.iloc[[instance_n]], return_type="interval")
    # preds = preds.flatten()
    # fig_path = f"plots/kde_intervals_{ds_name}.png"
    # fig = visualize_prediction_intervals_kde(
    #     y_preds=preds,
    #     y_true=semf.y_test.loc[instance_n].values[0],
    #     central_tendency="mean",
    #     return_fig=True,
    #     save_fig=True,
    #     fig_path=fig_path,
    # )
    # wandb.log({"kde_intervals": wandb.Image(fig_path)}, commit=False)

    # plt_n_instances = 100
    # preds = semf.infer_semf(semf.x_test.iloc[0:plt_n_instances], return_type="interval")
    # actuals = semf.y_test.iloc[0:plt_n_instances].values
    # fig_path = f"plots/boxplot_intervals_{ds_name}.png"
    # fig = visualize_prediction_intervals(
    #     preds,
    #     actuals,
    #     central_tendency="mean",
    #     return_fig=True,
    #     save_fig=True,
    #     fig_path=fig_path,
    # )
    # wandb.log({"boxplot_intervals": wandb.Image(fig_path)}, commit=False)

    # fig_path = f"plots/confidence_intervals_{ds_name}.png"
    # plot_confidence_intervals(
    #     get_confidence_intervals(preds, actuals),
    #     n_instances=plt_n_instances,
    #     central_tendency="mean",
    #     return_fig=True,
    #     save_fig=True,
    #     fig_path=fig_path,
    # )
    # wandb.log({"confidence_intervals": wandb.Image(fig_path)}, commit=False)

    print("-" * 40)

    wandb.log(
        {
            "val_RMSE": semf.valid_perf[optimal_i_value]["RMSE"],
            "val_MAE": semf.valid_perf[optimal_i_value]["MAE"],
            "train_perf": semf.train_perf,
            "valid_perf": semf.valid_perf,
            "num_iterations": optimal_i_value + 1,
            "sigma_Rp": semf.sigmaR_p,
            "sigma_R_perf": semf.sigmaR_perf,
            "elapsed_time": elapsed_time,
            "dataset": ds_name,
            "y_true_std": y_std,
            "dataset": f"{ds_name}",
        }
    )

if __name__ == "__main__":
    # Should be envoked from `evaluate_results.py` or the sweep `yaml` files
    args = parse_args()
    train_on_dataset_wandb(config=default_config)
    shutil.rmtree("plots")
