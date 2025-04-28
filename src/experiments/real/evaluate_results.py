"""
Evaluate the results of the experiments with the best hyperparameters and all the datasets
"""
import os
import wandb
import pandas as pd
import sys
import time
import gc
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))
from shared.cmd_configs import default_config
from run_experiments import train_on_dataset_wandb
import json
from dotenv import load_dotenv


def run_experiments(seeds, start_index, end_index):
    for seed in seeds:
        for index, row in hyperparams_df.iloc[start_index:end_index].iterrows():
            dataset = row["dataset"]
            model_class = row["model_class"]
            config = vars(default_config).copy()
            row_dict = row.to_dict()

            # Handle nested JSON structures if necessary
            if 'tree_config' in row_dict:
                row_dict['tree_config'] = json.loads(row_dict['tree_config'])
            if 'nn_config' in row_dict:
                row_dict['nn_config'] = json.loads(row_dict['nn_config'])

            config.update(row_dict)
            config['seed'] = seed

            # for large datasets such as California, we set a lower number for R_inference
            if (dataset in ['california_housing', 'cpu_activity', 'miami_housing', 'naval_propulsion_plant']) or (dataset == 'QSAR_fish_toxicity' and model_class == 'MultiMLPs'):
                config['stopping_patience'] = 5
            if dataset in ['california_housing', 'cpu_activity']:
                config['R_inference'] = 30
            else:
                config['R_inference'] = 50

            if dataset not in ['cpu_activity', 'naval_propulsion_plant', 'energy_efficiency', 'california_housing']:
                if dataset == 'QSAR_fish_toxicity' and model_class == 'MultiXGBs':
                    pass
                else:
                    config['z_norm_sd'] = 'train_residual_models'
            config['test_with_wide_intervals'] = True
            config['baseline_interval_method'] = 'quantile'
            config['x_group_size_prop'] = 'one'
            # config['nodes_per_feature'] = 10
            # config['x_group_size_prop'] = 'all'

            # Debugging print to check the actual config
            print(f"Configuration for {model_class} on {dataset}:")
            print(config)

            wandb.init(project=project_name, entity=entity_name, config=config) #, anonymous="allow"
            train_on_dataset_wandb(config=wandb.config)
            wandb.finish()
            gc.collect()  # Manually collect garbage

if __name__ == "__main__":
    # NEW: Parse command-line arguments for filtering
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        help="List of dataset names to run experiments on (e.g. california_housing, cpu_activity)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="List of model classes to run experiments with (e.g. MultiXGBs, MultiETs, MultiMLPs)"
    )
    args = parser.parse_args()

    api = wandb.Api()

    # Load environment variables from .env file
    load_dotenv()
    project_name = os.getenv('WANDB_PROJECT')
    entity_name = os.getenv('WANDB_ENTITY')

    # Use the results from the sweep to evaluate the best hyperparameters
    new_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    file_path = os.path.join(os.path.join(new_dir, '..'), "results", "sweep_hyperparams_final.csv")
    hyperparams_df = pd.read_csv(file_path)

    # Select only the columns that are hyperparameters
    columns_to_select = [
        "dataset",
        "model_class",
        "R",
        "nodes_per_feature",
        "device",
        "n_jobs",
        "force_n_jobs",
        "z_norm_sd",
        "stopping_patience",
        "R_inference",
        "nn_n_epochs"
    ]
    hyperparams_df = hyperparams_df[columns_to_select]

    # NEW: Filter by dataset if specified
    if args.datasets:
        hyperparams_df = hyperparams_df[hyperparams_df["dataset"].isin(args.datasets)]
        print(f"Filtering to datasets: {args.datasets}")

    # NEW: Filter by model if specified
    if args.models:
        hyperparams_df = hyperparams_df[hyperparams_df["model_class"].isin(args.models)]
        print(f"Filtering to model classes: {args.models}")

    # Set all the experiment settings
    seeds = [0, 10, 20, 30, 40]
    start_index = 0
    end_index = len(hyperparams_df)
    print(f"Running experiments on {end_index} configurations...")

    run_experiments(seeds, start_index, end_index)
    gc.collect()  # Manually collect garbage
