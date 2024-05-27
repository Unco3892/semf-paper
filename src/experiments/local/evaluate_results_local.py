import os
import pandas as pd
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))
from shared.cmd_configs import default_config
from local.run_experiments_local import train_on_dataset
import json
from argparse import Namespace

if __name__ == "__main__":
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

    for seed in seeds:
        for index, row in hyperparams_df.iloc[start_index:end_index].iterrows():
        # for index, row in hyperparams_df.iloc[start_index:].iterrows():
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
                config['missing_rate'] = missing_rate  # Explicitly set missing rate
                config['seed'] = seed
                # for large datasets such as California, we set the number for R_inference to 40
                if missing_rate > 0:
                    if dataset in ['california_housing', 'cpu_activity', 'miami_housing', 'naval_propulsion_plant']:
                        config['stopping_patience'] = 5
                    if dataset in ['california_housing', 'cpu_activity']:
                        config['R_inference'] = 30
                    else:
                        config['R_inference'] = 50

                # Debugging print to check the actual config
                print(f"Configuration for {model_class} on {dataset} with missing rate {missing_rate}:")
                config = Namespace(**config)
                print(config)

                train_on_dataset(ds_name=dataset, config = config, SEED=seed)
