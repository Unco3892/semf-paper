import pandas as pd
import os
import argparse

def sort_and_filter(group: pd.DataFrame, cwr_criteria: float = 0, picp_criteria: float = 0) -> pd.DataFrame:
    """
    Sorts and filters a group based on criteria for cwr and picp.

    Args:
        group (pd.DataFrame): The group to be sorted and filtered.
        cwr_criteria (float): The minimum value for cwr. Defaults to 0.
        picp_criteria (float): The minimum value for picp. Defaults to 0.

    Returns:
        pd.DataFrame: The sorted and filtered group.
    """
    return group[
        (group["valid_top1_rel_cwr_SEMF"] > cwr_criteria)
        & (group["valid_top1_rel_picp_SEMF"] > picp_criteria)
    ].sort_values(
        by=[
            "valid_top1_rel_cwr_SEMF",
            "valid_top1_rel_picp_SEMF",
        ],
        ascending=False,
    )


def custom_sort_with_fallback_and_criteria(group: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts and filters a group of data based on certain criteria, with fallback options.

    Args:
        group (pd.DataFrame): The group of data to be sorted and filtered.

    Returns:
        pd.DataFrame: The filtered group of data, with fallback options if necessary.
    """
    filtered_group = sort_and_filter(group)

    if filtered_group.empty:
        fallback_group = sort_and_filter(group, picp_criteria=-5)
        if fallback_group.empty:
            return sort_and_filter(group, cwr_criteria=float("-inf"), picp_criteria=float("-inf")).head(1)
        else:
            return fallback_group.head(1)
    else:
        return filtered_group.head(1)


def rearrange_columns(data: pd.DataFrame, desired_metrics: list, exclude_suffixes: list = None) -> pd.DataFrame:
    """
    Rearranges the columns of the given data DataFrame according to the desired order of metrics,
    with an option to exclude certain suffixes in column names.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be rearranged.
        desired_metrics (list): A list of metrics specifying the desired order.
        exclude_suffixes (list, optional): List of suffixes to exclude from the rearrangement.
                                           Defaults to None.

    Returns:
        pd.DataFrame: The rearranged DataFrame with columns ordered according to the desired metrics.
    """
    if exclude_suffixes is None:
        exclude_suffixes = []

    base_metric_names = set(
        col.replace("valid_top1_rel_", "")
        for col in data.columns
        if "top1_rel_" in col
    )

    metric_order_map = {metric: i for i, metric in enumerate(desired_metrics)}

    sorted_base_metrics = sorted(
        base_metric_names,
        key=lambda x: metric_order_map.get(x.split('_')[0], len(metric_order_map))
    )

    metric_columns = [
        f"valid_top1_rel_{metric}"
        for metric in sorted_base_metrics
        if not any(metric.endswith(suffix) for suffix in exclude_suffixes)
    ]

    metric_columns = [col for col in metric_columns if col in data.columns]

    config_columns = [col for col in data.columns if not any(col.startswith(f"valid_top1_rel_{metric}") for metric in sorted_base_metrics)]
    final_columns = config_columns + metric_columns

    return data.loc[:, final_columns]


def get_best_runs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the best runs for each dataset and model class.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be processed.

    Returns:
        pd.DataFrame: The DataFrame with the best runs.
    """
    best_runs = []
    for (model_class, dataset), group in data.groupby(["model_class", "dataset"]):
        best_run = custom_sort_with_fallback_and_criteria(group)
        best_runs.append(best_run)
    best_runs_df = pd.concat(best_runs).reset_index(drop=True)
    return best_runs_df


def order_datasets_and_models(data: pd.DataFrame, dataset_order: list, model_order: list) -> pd.DataFrame:
    """
    Orders the datasets and models according to the specified orders.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be ordered.
        dataset_order (list): The list specifying the desired order of datasets.
        model_order (list): The list specifying the desired order of models.

    Returns:
        pd.DataFrame: The ordered DataFrame.
    """
    data['model_order'] = data['model_class'].apply(lambda x: model_order.index(x) if x in model_order else len(model_order))
    data['dataset_order'] = data['dataset'].apply(lambda x: dataset_order.index(x) if x in dataset_order else len(dataset_order))
    ordered_data = data.sort_values(by=['model_order', 'dataset_order']).drop(columns=['model_order', 'dataset_order'])
    return ordered_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sweep results to find the best hyperparameters.")
    parser.add_argument('--include-metrics', action='store_true', help="Include metrics in the final CSV file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(base_dir, "..")
    file_path = os.path.join(base_dir, "results", "sweep_results.csv")
    data = pd.read_csv(file_path)

    best_runs_refactored = get_best_runs(data)

    columns_to_select = [
        "dataset",
        "model_class",
        "seed",
        "R",
        "nodes_per_feature",
        "nn_n_epochs",
        "device",
        "n_jobs",
        "force_n_jobs",
        "z_norm_sd",
        "stopping_patience",
        "R_inference",
    ]

    if args.include_metrics:
        columns_to_select += [col for col in best_runs_refactored.columns if col.startswith("valid_top1")]

    best_runs_refactored = best_runs_refactored[columns_to_select]

    if args.include_metrics:
        desired_metrics = ["cwr", "picp", "nmpiw", "rmse", "mae"]
        best_runs_refactored = rearrange_columns(best_runs_refactored, desired_metrics, exclude_suffixes=["Original", "imputation"])

    # Define the order of datasets
    dataset_order = [
        'space_ga', 'cpu_activity', 'naval_propulsion_plant', 'miami_housing', 
        'kin8nm', 'concrete_compressive_strength', 'cars', 'energy_efficiency', 
        'california_housing', 'airfoil_self_noise', 'QSAR_fish_toxicity'
    ]
    
    # Define the order of models
    model_order = ["MultiXGBs", "MultiETs", "MultiMLPs"]

    best_runs_refactored = order_datasets_and_models(best_runs_refactored, dataset_order, model_order)

    best_runs_refactored.to_csv(
        os.path.join(base_dir, "results", "sweep_hyperparams_final.csv"), index=False
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(best_runs_refactored)
