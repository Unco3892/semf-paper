import os
import pandas as pd

def load_csv(file_path):
    """Loads CSV data from a specified file path."""
    return pd.read_csv(file_path)

def filter_data(data, model_class, missing_rate):
    """Filters data based on model class and missing rate."""
    if model_class and missing_rate is not None:
        return data[(data['model_class'] == model_class) & (data['missing_rate'] == missing_rate)]
    elif model_class:
        return data[data['model_class'] == model_class]
    elif missing_rate is not None:
        return data[data['missing_rate'] == missing_rate]
    return data

def format_float(val, percentage=False, mean_std=False):
    """Formats floating point numbers for LaTeX output with percentage signs and rounding."""
    if pd.isna(val):
        return "---"
    try:
        if mean_std:
            mean_val, std_val = val
            if percentage:
                mean_val = round(mean_val)
                std_val = round(std_val)
                formatted_val = f"{mean_val}\\%$\\pm${std_val}\\%"
                if mean_val > 0:
                    formatted_val = f"\\textbf{{{formatted_val}}}"
            else:
                formatted_val = f"{mean_val:.2f}$\\pm${std_val:.2f}"
            return formatted_val
        else:
            if percentage:
                rounded_val = round(val)
                formatted_val = f"{rounded_val}\\%"
                if rounded_val > 0:
                    formatted_val = f"\\textbf{{{formatted_val}}}"
                return formatted_val
            return f"{val:.2f}"
    except ValueError:
        return val

def prepare_latex_table(df, aggregate=True):
    """Prepares dataframe for LaTeX conversion by formatting float columns."""
    if aggregate:
        mean_std_pairs = {
            'test_top1_relative_cwr_SEMF': True,
            'test_top1_relative_picp_SEMF': True,
            'test_top1_relative_nmpiw_SEMF': True,
            'test_PICP_SEMF_Original': False,
            'test_MPIW_SEMF_Original': False,
            'test_NMPIW_SEMF_Original': False,
            'test_top1_relative_rmse_SEMF': True,
            'test_top1_relative_mae_SEMF': True,
            'test_R2_SEMF_Original': False
        }

        for col_base, is_relative in mean_std_pairs.items():
            mean_col = col_base + '_mean'
            std_col = col_base + '_std'
            if mean_col in df.columns and std_col in df.columns:
                df[col_base] = df[[mean_col, std_col]].apply(lambda x: format_float((x.iloc[0], x.iloc[1]), percentage=is_relative, mean_std=True), axis=1)
                df = df.drop(columns=[mean_col, std_col])
    else:
        float_cols = df.select_dtypes(include=['float']).columns
        relative_cols = ['test_top1_relative_cwr_SEMF', 'test_top1_relative_picp_SEMF', 
                         'test_top1_relative_nmpiw_SEMF', 'test_top1_relative_rmse_SEMF', 
                         'test_top1_relative_mae_SEMF']
        
        for col in float_cols:
            if col in relative_cols:
                df[col] = df[col].apply(lambda x: format_float(x, percentage=True, mean_std=False))
            else:
                df[col] = df[col].apply(lambda x: format_float(x, mean_std=False))

    df['dataset'] = df['dataset'].apply(lambda x: x.replace('_', '\\_'))  # Escape underscores for LaTeX
    return df

def df_to_latex(df, caption, label, aggregate=True):
    """Converts DataFrame to a LaTeX table code with specific formatting."""
    df = prepare_latex_table(df, aggregate=aggregate)  # Prepare data
    column_format = 'l' + 'c' * (len(df.columns) - 1)  # Define the column format
    
    # Define the multi-column header for the table
    header = r"""
\multicolumn{1}{c}{} & \multicolumn{6}{c}{Interval Predictions} & \multicolumn{3}{c}{Point Predictions} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-10}
 & \multicolumn{3}{c}{Relative} & \multicolumn{3}{c}{Absolute} & \multicolumn{2}{c}{Relative} & Absolute \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9} \cmidrule(lr){10-10}
Dataset & $\Delta$CWR & $\Delta$PICP & $\Delta$NMPIW & PICP & MPIW & NMPIW & $\Delta$RMSE & $\Delta$MAE & $\text{R}^2$ \\
\midrule
"""
    # Convert DataFrame to LaTeX code without index and escape set to false
    body = df.to_latex(index=False, escape=False, header=False)
    body = body.replace('\\begin{tabular}{llllllllll}', '').replace('\\end{tabular}', '').replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', '').strip()
    
    # Full LaTeX table environment
    latex_table = f"""
\\begin{{table*}}[!htbp]
\\caption{{{caption}}}
\\label{{{label}}}
\\vskip 0.15in
\\begin{{small}}
\\begin{{sc}}
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{column_format}}}
\\toprule
{header}
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{sc}}
\\end{{small}}
\\end{{table*}}
"""
    return latex_table

def convert_list_columns(data):
    """Convert list columns to atomic values if they end with 'imputation' or 'Original' (except 'SEMF_Original')."""
    list_columns = [col for col in data.columns if col.endswith('imputation') or (col.endswith('_Original') and not col.endswith('SEMF_Original'))]
    for col in list_columns:
        data[col] = data[col].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    # Drop the columns from the DataFrame
    data = data.drop(columns=list_columns)
    return data

def aggregate_metrics(data):
    """Aggregates the metrics by calculating mean and standard deviation."""
    mean_data = data.groupby('dataset', observed=True).mean().reset_index()
    # Calculate standard deviation for each group
    std_data = data.groupby('dataset', observed=True).std(ddof=0).reset_index()
    agg_data = pd.merge(mean_data, std_data, on='dataset', suffixes=('_mean', '_std')).reset_index()
    return agg_data

def save_latex_table(latex_code, filename):
    """Saves the LaTeX table code to a .tex file."""
    with open(filename, 'w') as f:
        f.write(latex_code)

def generate_caption(model_class, missing_rate, aggregate, seeds_str, num_seeds):
    """Generates a caption for the LaTeX table."""
    data_type = "complete data" if missing_rate == 0 else "50\\% missing data"
    aggregate_text = "aggregated over {num_seeds} seeds, with rows ordered by seed (ascending). For each metric, the mean and standard deviation of the performance across the seeds are separated by $\\pm$." if aggregate else f"for seeds {seeds_str}, with rows ordered by seed (ascending)."
    caption = f"Test results for {model_class} with {data_type} at 95\\% quantiles {aggregate_text} Performance over the baseline is highlighted in bold."
    return caption

def generate_tables(data, aggregate=True, output_dir=None, interval_suffix = None):
    """
    Generate LaTeX tables from the given data.

    Args:
        data (pandas.DataFrame): The input data containing the results.
        aggregate (bool, optional): Whether to aggregate the data over different seeds or not. Defaults to True.
        output_dir (str, optional): The directory to save the generated tables. Defaults to None.
        interval_suffix (str, optional): The suffix to add to the table label. Defaults to None.

    Returns:
        None
    """
    model_classes = data['model_class'].unique()
    
    # Define the order of datasets
    dataset_order = [
        'space_ga', 'cpu_activity', 'naval_propulsion_plant', 'miami_housing', 
        'kin8nm', 'concrete_compressive_strength', 'cars', 'energy_efficiency', 
        'california_housing', 'airfoil_self_noise', 'QSAR_fish_toxicity'
    ]

    seeds = sorted(data['seed'].unique())
    seeds_str = ', '.join(map(str, seeds))
    num_seeds = len(seeds)

    for model_class in model_classes:
        for missing_rate, label_suffix in zip([0, 0.5], ['complete', 'missing']):
            filtered_data = filter_data(data, model_class, missing_rate)
            filtered_data = filtered_data.copy()  # Avoid SettingWithCopyWarning
            filtered_data['dataset'] = pd.Categorical(filtered_data['dataset'], categories=dataset_order, ordered=True)
            filtered_data = filtered_data.sort_values(by=['dataset', 'seed'], ascending=True)

            # Select only relevant columns before aggregation
            selected_columns = ['dataset', 'test_top1_relative_cwr_SEMF', 'test_top1_relative_picp_SEMF',
                                'test_top1_relative_nmpiw_SEMF', 'test_PICP_SEMF_Original', 'test_MPIW_SEMF_Original',
                                'test_NMPIW_SEMF_Original', 'test_top1_relative_rmse_SEMF', 'test_top1_relative_mae_SEMF',
                                'test_R2_SEMF_Original']
            filtered_data = filtered_data[selected_columns]
            filtered_data = convert_list_columns(filtered_data)
    
            if aggregate:
                aggregated_data = aggregate_metrics(filtered_data)
                selected_columns = ['dataset', 'test_top1_relative_cwr_SEMF_mean', 'test_top1_relative_cwr_SEMF_std',
                                    'test_top1_relative_picp_SEMF_mean', 'test_top1_relative_picp_SEMF_std',
                                    'test_top1_relative_nmpiw_SEMF_mean', 'test_top1_relative_nmpiw_SEMF_std',
                                    'test_PICP_SEMF_Original_mean', 'test_PICP_SEMF_Original_std',
                                    'test_MPIW_SEMF_Original_mean', 'test_MPIW_SEMF_Original_std',
                                    'test_NMPIW_SEMF_Original_mean', 'test_NMPIW_SEMF_Original_std',
                                    'test_top1_relative_rmse_SEMF_mean', 'test_top1_relative_rmse_SEMF_std',
                                    'test_top1_relative_mae_SEMF_mean', 'test_top1_relative_mae_SEMF_std',
                                    'test_R2_SEMF_Original_mean', 'test_R2_SEMF_Original_std']
                aggregated_data = aggregated_data[selected_columns]
                df_to_use = aggregated_data
            else:
                df_to_use = filtered_data

            caption = generate_caption(model_class, missing_rate, aggregate, seeds_str, num_seeds)
            if interval_suffix == "raw":
                label_suffix += "-raw"

            label = f"table-{model_class.lower().replace(':', '-').replace(';', '-').replace('_', '-')}-{label_suffix}"
            if aggregate:
                label += "-aggregated"
            latex_table = df_to_latex(df_to_use, caption, label, aggregate=aggregate)
            
            if output_dir:
                filename = os.path.join(output_dir, f"{label}.tex")
                save_latex_table(latex_table, filename)
            else:
                print(latex_table)  # For demonstration; you might want to save this to a file instead

if __name__ == '__main__':
    # Load best hyperparameters for each dataset from CSV
    new_dir = os.path.dirname(os.getcwd())
    ## Define the relative path to the file from new_dir
    file_path = os.path.join(
        new_dir, "results", "sweep_results_conformalized.csv"
        # new_dir, "results", "sweep_results.csv"
    )  # Or change it to your params of choice
    # Optionally, specify the suffix for the interval tables (e.g., "raw" for raw intervals)
    interval_suffix = None
    # interval_suffix = "raw"

    data = load_csv(file_path)
    
    # Define the output directory for LaTeX tables
    output_dir = "tex_tbls"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate LaTeX tables for all model classes, with aggregation
    generate_tables(data, aggregate=True, output_dir=output_dir, interval_suffix=interval_suffix)
    
    # Optionally, generate LaTeX tables for all model classes without aggregation
    generate_tables(data, aggregate=False, output_dir=output_dir, interval_suffix=interval_suffix)
