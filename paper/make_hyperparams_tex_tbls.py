import pandas as pd
import os

def load_csv(file_path):
    """Loads CSV data from a specified file path."""
    return pd.read_csv(file_path)

def generate_combined_latex_table(data, model_classes):
    """Generates a combined LaTeX table for all model classes.

    Args:
        data (pandas.DataFrame): The input data containing the hyperparameters for each model class.
        model_classes (list): A list of model classes.

    Returns:
        str: The generated LaTeX table.

    """
    combined_rows = []

    for i, model_class in enumerate(model_classes):
        filtered_df = data[data['model_class'] == model_class]
        escaped_model_class = model_class.replace('_', '\\_').replace('-', '\\-')
        if i > 0:
            combined_rows.append("\\cmidrule(lr){1-7}\n")
        # Fix: Remove unnecessary "&" symbols from model class header row
        combined_rows.append(f"\\textbf{{{escaped_model_class}}} \\\\ \\cmidrule(lr){{1-7}}\n")
        for _, row in filtered_df.iterrows():
            xi_nodes = eval(row['simulator_architecture'])[0]['units']
            escaped_dataset = row['dataset'].replace('_', '\\_')
            combined_rows.append(f"{escaped_dataset} & {row['R']} & {row['nodes_per_feature']} & {row['z_norm_sd']} & {row['stopping_patience']} & {row['R_inference']} & {xi_nodes} \\\\\n")

    latex_table = r"""
\begin{table*}[t]
\begin{center}
\caption{Hyper-parameters for MultiXGBs, MultiETs, and MultiMLPs used for both complete and missing data.}
\label{table-combined-hyperparameters}
\vskip 0.15in
\begin{small}
\begin{sc}
\resizebox{0.9\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{Dataset} & \multicolumn{5}{c}{Complete and Missing} & \multicolumn{1}{c}{Missing} \\
\cmidrule(lr){2-6} \cmidrule(lr){7-7}
 & R & $m_k$ & $\sigma_{k}$ & Patience & $R_\mathrm{infer}$ & $\xi_\mathrm{nodes}$ \\
\midrule
"""
    for row in combined_rows:
        latex_table += row

    latex_table += r"""
\bottomrule
\end{tabular}
}
\end{sc}
\end{small}
\end{center}
\end{table*}
"""
    return latex_table


def save_latex_table(latex_code, filename):
    """Saves the LaTeX table code to a .tex file."""
    with open(filename, 'w') as f:
        f.write(latex_code)

if __name__ == '__main__':
    # Load best hyperparameters for each dataset from CSV
    new_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(new_dir, "results", "sweep_hyperparams_final.csv")
    data = load_csv(file_path)
    
    # List of unique model classes
    model_classes = data['model_class'].unique()

    # Generate combined LaTeX table for all model classes
    combined_latex_table = generate_combined_latex_table(data, model_classes)

    # Define the output directory for LaTeX table
    output_dir = "tex_tbls"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the combined LaTeX table
    output_file = os.path.join(output_dir, "table-combined-hyperparameters.tex")
    save_latex_table(combined_latex_table, output_file)

    print(f"Combined hyperparameters table saved to {output_file}")
