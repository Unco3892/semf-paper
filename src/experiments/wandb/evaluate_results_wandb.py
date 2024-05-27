"""
Evaluate the results of the experiments with the best hyperparameters and all the datasets
"""
import os
import wandb
import pandas as pd
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))
from shared.cmd_configs import default_config
from run_experiments_wandb import train_on_dataset_wandb
import json
from dotenv import load_dotenv


if __name__ == "__main__":
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
        "simulator_architecture",
        "R_inference",
        "nn_n_epochs"
    ]
    hyperparams_df = hyperparams_df[columns_to_select]

    # Set all the experiment settings
    missing_rates = [0, 0.5]
    seeds = [0, 10, 20, 30, 40]    
    start_index = 0
    end_index = len(hyperparams_df)

    # Run the experiments
    for seed in seeds:
        for index, row in hyperparams_df.iloc[start_index:end_index].iterrows():
            dataset = row["dataset"]
            model_class = row["model_class"]
            for missing_rate in missing_rates:
                config = vars(default_config).copy()
                row_dict = row.to_dict()

                # Handle nested JSON structures if necessary
                if 'tree_config' in row_dict:
                    row_dict['tree_config'] = json.loads(row_dict['tree_config'])
                if 'nn_config' in row_dict:
                    row_dict['nn_config'] = json.loads(row_dict['nn_config'])
                if 'simulator_architecture' in row_dict:
                    row_dict['simulator_architecture'] = json.loads(row_dict['simulator_architecture'])


                config.update(row_dict)
                config['missing_rate'] = missing_rate
                config['seed'] = seed

                # for large datasets such as California, we set a lower number for R_inference
                if missing_rate > 0:
                    if dataset in ['california_housing', 'cpu_activity', 'miami_housing', 'naval_propulsion_plant']:
                        config['stopping_patience'] = 5
                    if dataset in ['california_housing', 'cpu_activity']:
                        config['R_inference'] = 30
                    else:
                        config['R_inference'] = 50

                # Debugging print to check the actual config
                print(f"Configuration for {model_class} on {dataset} with missing rate {missing_rate}:")
                print(config)

                wandb.init(project=project_name, entity=entity_name, config=config) #, anonymous="allow"
                train_on_dataset_wandb(config=wandb.config, SEED=config["seed"])
                wandb.finish()
