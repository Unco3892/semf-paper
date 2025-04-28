import pandas as pd
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_project_runs(project_name: str, exclude_seeds: list = None) -> pd.DataFrame:
    """
    Fetches the runs data and configurations (hyperparameters) for a given project on wandb.

    Args:
        project_name (str): The name of the wandb project.
        exclude_seeds (list): A list of seed values to exclude from the DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the runs data and configurations.
    """
    api = wandb.Api()
    runs = api.runs(project_name)
    
    data = []
    for run in runs:
        run_data = run.summary_metrics
        run_data.update(run.config)
        run_data['run_name'] = run.name
        data.append(run_data)

    df = pd.DataFrame(data)
    
    # Columns to remove
    columns_to_remove = ['_settings', '_callback', 'boxplot_intervals', 'tr_val_metrics', 'kde_intervals', 'quantile_intervals', 'confidence_intervals']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    
    # Remove columns starting with '_'
    df = df.loc[:, ~df.columns.str.startswith('_')]
    
    # Define the order of hyperparameter columns
    hyperparameter_columns = [
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
        "nn_n_epochs",
        "seed"
    ]

    # Priority columns for each split
    priority_patterns = [
        '{split}_top1_rel_cwr_SEMF',
        '{split}_top1_rel_picp_SEMF',
        '{split}_top1_rel_nmpiw_SEMF',
        '{split}_top1_rel_crps_SEMF',
        '{split}_top1_rel_pinball_SEMF',
        '{split}_PICP_SEMF_Original',
        '{split}_MPIW_SEMF_Original',
        '{split}_NMPIW_SEMF_Original',
        '{split}_CWR_SEMF_Original',
        '{split}_CRPS_SEMF_Original',
        '{split}_Pinball_SEMF_Original',
        # '{split}_top1_rel_rmse_SEMF',
        # '{split}_top1_rel_mae_SEMF',
        # '{split}_R2_SEMF_Original'
    ]
    
    splits = ['valid', 'test', 'train']
    priority_columns = []
    for split in splits:
        for pattern in priority_patterns:
            col_name = pattern.format(split=split)
            if col_name in df.columns:
                priority_columns.append(col_name)

    # Collect remaining {split} columns
    valid_columns = [col for col in df.columns if col.startswith("valid_") and col not in priority_columns]
    test_columns = [col for col in df.columns if col.startswith("test_") and col not in priority_columns]
    train_columns = [col for col in df.columns if col.startswith("train_") and col not in priority_columns]
    
    # Remaining columns
    remaining_columns = [col for col in df.columns if col not in hyperparameter_columns + priority_columns + valid_columns + test_columns + train_columns + ["run_name"]]
    
    # Order the columns
    ordered_columns = ["run_name"] + hyperparameter_columns + priority_columns + valid_columns + test_columns + train_columns + remaining_columns
    
    # Ensure all ordered columns exist in the DataFrame
    ordered_columns = [col for col in ordered_columns if col in df.columns]
    
    # Reorder the DataFrame columns
    df = df[ordered_columns]

    # Filter out rows with excluded seed values
    if exclude_seeds:
        df = df[~df['seed'].isin(exclude_seeds)]

    return df

if __name__ == '__main__':
    wandb.login()

    project_name = os.getenv('WANDB_PROJECT')

    # Example of seeds to exclude
    # exclude_seeds = [30, 40]
    exclude_seeds = None

    data = fetch_project_runs(project_name, exclude_seeds=exclude_seeds)

    print(f"Data fetched successfully for project: {project_name}")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    data.to_csv(os.path.join(results_dir, 'sweep_results.csv'), index=False)

    print(data)
