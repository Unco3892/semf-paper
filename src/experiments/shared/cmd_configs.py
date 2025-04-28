import json
from types import SimpleNamespace
from argparse import ArgumentParser, ArgumentTypeError
import argparse
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))
from semf.utils import load_json_config, add_boolean_argument

# List of datasets for regression tasks with less than 50,000 instances
openml_datasets = {
    "space_ga": 45402,
    "cpu_activity": 44978,
    "naval_propulsion_plant": 44969,
    "miami_housing": 44983,
    "kin8nm": 44980,
    "concrete_compressive_strength": 44959,
    "cars": 44994,
    "energy_efficiency": 44960,
    "california_housing": 44977,
    "airfoil_self_noise": 44957,
    "QSAR_fish_toxicity": 44970,
    "simulate_linear_quadratic": None
}

regression_datasets_lt_30k = list(openml_datasets.keys())

default_config = SimpleNamespace(
    seed=0,
    # SEMF settings (in addition to this, there are the the hyperparameter of the models themselves)
    training_set_size=0.7,
    valid_set_size=0.15,
    R=5,
    nodes_per_feature=10,
    x_group_size_prop="one",
    stopping_metric="RMSE",
    model_class = 'MultiXGBs',
    tree_config = {"tree_n_estimators":100, "xgb_max_depth":None, "xgb_patience":10, "et_max_depth":10},
    nn_config = {"nn_batch_size": None, "nn_load_into_memory": True, "nn_epochs": 1000, "nn_lr": 0.001, "nn_patience": 100},
    nn_n_epochs= None, # only relevant when using wandb sweeps since it cannot parse the nn_config
    models_val_split=0.15,
    parallel_type="semf_joblib",
    device="cpu",
    n_jobs=2,
    force_n_jobs=False,
    return_mean_default=True,
    mode_type="approximate", # ignored (not relevant) if return_mean_default is True
    custom_sigma_R=None,
    z_norm_sd=0.1,
    initial_z_norm_sd=None,
    fixed_z_norm_sd=None,
    stopping_patience=10,
    max_it=500,
    # Uncertainty method
    use_uncertainty_wrapper=False,
    use_empirical_wrapper=False,
    # For debugging SEMF class
    use_constant_weights=False,
    verbose=False,
    # Benchmark and Experiment settings
    save_models=False,
    return_point_benchmark=True,
    return_interval_benchmark=True,
    alpha_certainty=0.05,
    R_inference=50,
    test_with_wide_intervals=True,
    benchmark_knn_neighbors=5,
    benchmark_all = False
)


# Add the other arguments here
def parse_args():
    """Parse and handle command-line arguments."""
    parser = ArgumentParser(
        description="Run experiments with the SEMF framework either locally or integrated with Weights & Biases.",
        epilog="""
    Example of use:

    For Linux/MacOS:
        python run_experiments_local.py --nn_config '{"nn_batch_size":32,"nn_epochs":100}' --tree_config '{"tree_n_estimators":100}' --force_n_jobs --no-save_models --verbose --test_with_wide_intervals --no-return_interval_benchmark --no-use_constant_weights

    For Windows (option 1):
        python run_experiments_local.py --dataset="space_ga" --nn_config="{""nn_batch_size"":32,""nn_epochs"":100}" --tree_config="{""tree_n_estimators"":100}" --force_n_jobs --no-save_models --verbose --test_with_wide_intervals --no-return_interval_benchmark --no-use_constant_weights

    For Windows (option 2):
        python run_experiments_local.py --nn_config "{\\"nn_batch_size\\":32,\\"nn_epochs\\":100}" --tree_config "{\\"tree_n_estimators\\":100}" --force_n_jobs --no-save_models --verbose --test_with_wide_intervals --no-return_interval_benchmark --no-use_constant_weights

    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--nn_n_epochs",
        type=int,
        default=default_config.nn_n_epochs,
        help="Number of epochs for neural network training, which is a duplicate of nn_config.nn_epochs for wandb sweeps (parsing issue in yaml https://github.com/wandb/wandb/issues/982) and does not apply to local runs.",
    )
    parser.add_argument(
        "--models_val_split",
        type=float,
        default=default_config.models_val_split,
        help="Validation split for early stopping of MLP, QNN and XGBooost withthe exception of ET that has no early stopping",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_config.seed,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=regression_datasets_lt_30k,
        help="List of datasets (names) to run the experiment on. If empty, all datasets will be used.",
    )
    parser.add_argument(
        "--R", type=int, default=default_config.R, help="R hyperparameter for SEMF"
    )
    parser.add_argument(
        "--nodes_per_feature",
        type=int,
        default=default_config.nodes_per_feature,
        help="Nodes per feature hyperparameter for SEMF",
    )
    parser.add_argument(
        "--x_group_size_prop",
        type=str,
        default=default_config.x_group_size_prop,
        help="Proportion of number of columns per input",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default=default_config.model_class,
        help="Model class hyperparameter for SEMF",
    )
    parser.add_argument(
        "--tree_config",
        type=str,
        default=json.dumps(default_config.tree_config), 
        help="Settings for For XGBoost & Random Forest"
    )
    parser.add_argument(
        "--nn_config",
        type=str,
        default=json.dumps(default_config.nn_config),
        help="Hyperparameters for the neural network model which includes nn_batch_size, nn_load_into_memory, nn_epochs, nn_lr in JSON format.",
    )
    parser.add_argument(
        "--parallel_type",
        type=str,
        default=default_config.parallel_type,
        help="Type of parallelization for $\\phi$ & $\\theta$ training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_config.device,
        help="Device to use for training",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=default_config.n_jobs,
        help="Number of jobs to run in parallel",
    )
    add_boolean_argument(
        parser, 'force_n_jobs', default_config.force_n_jobs,
        "Force the number of jobs to run in parallel.",
        "Do not force the number of jobs to run in parallel."
    )
    parser.add_argument(
        "--max_it",
        type=int,
        default=default_config.max_it,
        help="Max iterations hyperparameter for SEMF",
    )
    parser.add_argument(
        "--stopping_patience",
        type=int,
        default=default_config.stopping_patience,
        help="Stopping patience hyperparameter for SEMF",
    )
    parser.add_argument(
        "--stopping_metric",
        type=str,
        default=default_config.stopping_metric,
        help="Stopping metric hyperparameter for SEMF",
    )
    parser.add_argument(
        "--z_norm_sd",
        default=default_config.z_norm_sd,
        help="Std of for sampling the Z dimension and a hyperparameter for SEMF. This can either be the same value as `sigma`, `weighted_residuals` or a custom value (int, float).",
    )
    parser.add_argument(
        "--custom_sigma_R",
        default=default_config.custom_sigma_R,
        help="Custom sigma R hyperparameter for SEMF",
    )
    add_boolean_argument(
        parser, 'use_constant_weights', default_config.use_constant_weights,
        "Use constant weights in SEMF.",
        "Do not use constant weights in SEMF."
    )
    add_boolean_argument(
        parser, 'verbose', default_config.verbose,
        "Enable verbose output for debugging.",
        "Disable verbose output."
    )
    add_boolean_argument(
        parser, 'return_mean_default', default_config.return_mean_default,
        "Return mean as default output.",
        "Do not return mean as default output."
    )
    parser.add_argument(
        "--mode_type",
        type=str,
        default=default_config.mode_type,
        help="Mode type hyperparameter for SEMF. If return_mean_default is True, this is ignored.",
    )
    parser.add_argument(
        "--initial_z_norm_sd", default=default_config.initial_z_norm_sd, help="."
    )
    parser.add_argument(
        "--fixed_z_norm_sd", default=default_config.fixed_z_norm_sd, help="."
    )
    add_boolean_argument(
        parser, 'return_point_benchmark', default_config.return_point_benchmark,
        "Return point benchmark results.",
        "Do not return point benchmark results."
    )
    add_boolean_argument(
        parser, 'return_interval_benchmark', default_config.return_interval_benchmark,
        "Return interval benchmark results.",
        "Do not return interval benchmark results."
    )
    parser.add_argument(
        "--alpha_certainty",
        type=float,
        default=default_config.alpha_certainty,
        help=".",
    )
    parser.add_argument(
        "--R_inference", type=int, default=default_config.R_inference, help="."
    )
    add_boolean_argument(
        parser, 'test_with_wide_intervals', default_config.test_with_wide_intervals,
        "Test with wide intervals in benchmarks.",
        "Do not test with wide intervals in benchmarks."
    )
    parser.add_argument(
        "--benchmark_knn_neighbors",
        type=int,
        default=default_config.benchmark_knn_neighbors,
        help=".",
    )
    add_boolean_argument(
        parser, 'save_models', default_config.save_models,
        "Save models after training. Only works for the local experiments and not wandb sweeps",
        "Do not save models after training."
    )
    add_boolean_argument(
        parser, 'benchmark_all', default_config.benchmark_all,
        "Run all benchmarks models from `model_class` and not only the base.",
        "Run only the base benchmark models. (e.g. if `model_class` is `MultiMLPs`, run only benchmark for `MLP`)"
    )

    args = parser.parse_args()
    if args.nn_config:
        args.nn_config = load_json_config(args.nn_config, default_config.nn_config)
    if args.tree_config:
        args.tree_config = load_json_config(args.tree_config, default_config.tree_config)

    if args.model_class:
        args.model_class = args.model_class.strip("'\"")

    for key, value in vars(args).items():
        if value is not None and hasattr(default_config, key):
            setattr(default_config, key, value)

    return args

if __name__ == "__main__":
    print(parse_args)
