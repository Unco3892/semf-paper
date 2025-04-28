import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))
from semf.preprocessing import DataPreprocessor
from semf.semf import SEMF
from semf import utils
from shared.benchmark import BenchmarkSEMF, display_results

# --- Using the unified synthetic data generator ---
from synthetic_data_generator import generate_synthetic_data

# Import wandb (will only be used if --use_wandb is passed)
try:
    import wandb
except ImportError:
    wandb = None

def run_synthetic_experiment(distribution, baseline="MultiXGBs", model_seed=0, data_seed=0, n=1000, num_variables=2, baseline_interval_method="quantile", data_gen_method="cosine"):
    """
    Runs a synthetic experiment for a given distribution and baseline model using 
    the unified synthetic data generator. The generated data follow the model 
    $$y = f(x) + \\epsilon$$ where $$f(x)$$ is defined as $$\\sum_{i=1}^{k}\\cos(x_i)$$ for the "cosine" setup (isolating noise effects) or as $$\\sum_{i=1}^{k}\\Bigl(x_i^2+0.5\\sin(3x_i)\\Bigr)$$ for the "quadratic"  setup (introducing nonlinearity and heteroscedasticity). $$\\epsilon$$ is drawn from one of four distributions. 
    
    Parameters:
        data_gen_method (str): Data generation method to use. Options: "cosine" or "quadratic".
    """
    print(f"\nRunning experiment with settings: baseline={baseline}, distribution={distribution}, " 
          f"data_seed={data_seed}, model_seed={model_seed}, n={n}, num_variables={num_variables}, "
          f"baseline_interval_method={baseline_interval_method}, data_gen_method={data_gen_method}\n")
    
    utils.set_seed(data_seed)

    # Use the appropriate unified synthetic data generator function based on data_gen_method.
    df = generate_synthetic_data(n, num_variables, seed=data_seed, noise_distribution=distribution, method=data_gen_method)
    
    # Reset seed for model training
    utils.set_seed(model_seed)
    
    base_mapping = {
        "MultiXGBs": "XGB",
        "MultiETs": "ET",
        "MultiMLPs": "MLP"
    }
    base_model = base_mapping.get(baseline, "XGB")
    
    data_preprocessor = DataPreprocessor(df, y_col='Output', train_size=0.7, valid_size=0.15, seed=model_seed)
    data_preprocessor.split_data()
    data_preprocessor.scale_data(scale_output=True)
    
    if baseline == "MultiMLPs":
        device = "gpu"
        n_jobs = 1
        force_n_jobs = True
    else:
        device = "cpu"
        n_jobs = 2
        force_n_jobs = False

    z_norm_sd = "train_residual_models"
    stopping_patience = 10

    if baseline == "MultiXGBs":
        R = 10
        nodes_per_feature = np.array([10] * num_variables)
    elif baseline == "MultiETs":
        R = 10 
        nodes_per_feature = np.array([10] * num_variables)
    else:  # MultiMLPs
        R = 5
        nodes_per_feature = np.array([20] * num_variables)
    
    # Define common hyperparameters for both SEMF and its baselines
    common_tree_config = {
        "tree_n_estimators": 100,
        "xgb_max_depth": None,
        "xgb_patience": 10,
        "et_max_depth": 10
    }
    common_nn_config = {
        "nn_batch_size": None,
        "nn_load_into_memory": True,
        "nn_epochs": 1000,
        "nn_lr": 0.001,
        "nn_patience": 100
    }
    
    semf = SEMF(
        data_preprocessor,
        R=R,
        nodes_per_feature=nodes_per_feature,
        model_class=baseline,
        tree_config=common_tree_config,            # Use common tree hyperparameters
        nn_config=common_nn_config,                # Use common NN hyperparameters
        models_val_split=0.15,
        parallel_type="semf_joblib",
        device=device,
        n_jobs=n_jobs,
        force_n_jobs=force_n_jobs,
        max_it=100,
        stopping_patience=stopping_patience,
        stopping_metric="RMSE",
        custom_sigma_R=None,
        z_norm_sd=z_norm_sd,
        initial_z_norm_sd=None,
        fixed_z_norm_sd=None,
        return_mean_default=True,
        mode_type="approximate",
        use_constant_weights=False,
        verbose=False,
        x_group_size=1,
        seed=model_seed
    )
    semf.train_semf()
    
    df_train, df_valid, df_test = data_preprocessor.get_train_valid_test()
    
    if baseline in ["MultiXGBs", "MultiETs"]:
        R_inference = R
    else:
        R_inference = R
    
    benchmark = BenchmarkSEMF(
        df_train,
        df_valid,
        df_test,
        y_col='Output',
        semf_model=semf,
        alpha=0.05,
        base_model=base_model,
        baseline_interval_method=baseline_interval_method,
        test_with_wide_intervals=False,
        seed=model_seed,
        inference_R=R_inference,
        tree_n_estimators=common_tree_config["tree_n_estimators"],  # Use same hyperparameters
        xgb_max_depth=common_tree_config["xgb_max_depth"],
        et_max_depth=common_tree_config["et_max_depth"],
        nn_batch_size=common_nn_config["nn_batch_size"],
        nn_epochs=common_nn_config["nn_epochs"],
        nn_lr=common_nn_config["nn_lr"],
        nn_load_into_memory=common_nn_config["nn_load_into_memory"],
        device=device,
        models_val_split=0.15,
        xgb_patience=common_tree_config["xgb_patience"],
        nn_patience=common_nn_config["nn_patience"]
    )
    point_results = benchmark.run_pointpred()
    interval_results = benchmark.run_intervals()

    print("\nSynthetic Experiment - Point Prediction Benchmark Results:")
    display_results(point_results, sort_descending_by="MAE", include_imputation=True)
    print("\nSynthetic Experiment - Interval Prediction Benchmark Results:")
    display_results(interval_results, sort_descending_by="CWR", include_imputation=True)

    return {
        'distribution': distribution,
        'point_benchmark': point_results,
        'interval_benchmark': interval_results
    }

def aggregate_results(baseline, distribution, model_seeds, baseline_interval_method="quantile", 
                      data_seed=42, n=1000, num_variables=2, data_gen_method="cosine"):
    # Create a list to store the raw results from every run.
    raw_runs = []
    results = {
        "ΔCWR": [], "ΔPICP": [], "ΔNMPIW": [], "ΔCRPS": [],
        "PICP": [], "NMPIW": [], "CWR": [], "CRPS": [],
        "ΔRMSE": [], "ΔMAE": [], "R2": [],
        "Pinball": [], "ΔPinball": []
    }
    
    for model_seed in model_seeds:
        res = run_synthetic_experiment(
            distribution, 
            baseline=baseline, 
            model_seed=model_seed,
            data_seed=data_seed, 
            n=n, 
            num_variables=num_variables,
            baseline_interval_method=baseline_interval_method,
            data_gen_method=data_gen_method
        )
        if res is None:
            continue
        
        # Save the raw result for this run.
        raw_runs.append(res)
        
        test_results = res['interval_benchmark'].get('test', [])
        semf_result = next((r for r in test_results if 'SEMF' in r.get('Model', '')), None)
        if semf_result:
            # Extract relative metrics, trying "top1_rel_" keys first, fallback if NaN.
            cwr_val = semf_result.get('top1_rel_cwr', (np.nan,))[0]
            if np.isnan(cwr_val):
                cwr_val = semf_result.get('ΔCWR', (np.nan,))[0]
            results["ΔCWR"].append(cwr_val)
            
            picp_val = semf_result.get('top1_rel_picp', (np.nan,))[0]
            if np.isnan(picp_val):
                picp_val = semf_result.get('ΔPICP', (np.nan,))[0]
            results["ΔPICP"].append(picp_val)
            
            nmpiw_val = semf_result.get('top1_rel_nmpiw', (np.nan,))[0]
            if np.isnan(nmpiw_val):
                nmpiw_val = semf_result.get('ΔNMPIW', (np.nan,))[0]
            results["ΔNMPIW"].append(nmpiw_val)
            
            crps_val = semf_result.get('top1_rel_crps', (np.nan,))[0]
            if np.isnan(crps_val):
                crps_val = semf_result.get('ΔCRPS', (np.nan,))[0]
            results["ΔCRPS"].append(crps_val)
            
            pinball_val = semf_result.get('top1_rel_pinball', (np.nan,))[0]
            if np.isnan(pinball_val):
                pinball_val = semf_result.get('ΔPinball', (np.nan,))[0]
            results["ΔPinball"].append(pinball_val)

            # Extract absolute metrics
            results["PICP"].append(semf_result.get('PICP', np.nan))
            results["NMPIW"].append(semf_result.get('NMPIW', np.nan))
            results["CWR"].append(semf_result.get('CWR', np.nan))
            results["CRPS"].append(semf_result.get('CRPS', np.nan))
            results["Pinball"].append(semf_result.get('Pinball', np.nan))

            # Extract point prediction metrics from test set
            test_point_results = res['point_benchmark'].get('test', [])
            semf_point_result = next((r for r in test_point_results if r['Model'] == 'SEMF'), None)
            if semf_point_result:
                results["ΔRMSE"].append(semf_point_result.get('top1_rel_rmse', (np.nan,))[0])
                results["ΔMAE"].append(semf_point_result.get('top1_rel_mae', (np.nan,))[0])
                results["R2"].append(semf_point_result.get('R2', np.nan))
    
    # Calculate aggregated metrics.
    aggregated = {}
    for metric, values in results.items():
        if values:  # Only process if we have values
            clean_values = [v for v in values if not np.isnan(v)]
            if clean_values:
                if metric.startswith("Δ"):
                    mean_val = np.mean(clean_values)
                    min_val = min(clean_values)
                    max_val = max(clean_values)
                    aggregated[metric] = (mean_val, min_val, max_val)
                else:
                    mean_val = np.mean(clean_values)
                    std_val = np.std(clean_values) if len(clean_values) > 1 else 0
                    aggregated[metric] = (mean_val, std_val)
            else:
                aggregated[metric] = (np.nan, np.nan) if not metric.startswith("Δ") else (np.nan, np.nan, np.nan)
        else:
            aggregated[metric] = (np.nan, np.nan) if not metric.startswith("Δ") else (np.nan, np.nan, np.nan)
    
    # Return both the aggregated metrics and the raw run results.
    return {"aggregated": aggregated, "runs": raw_runs}


# Add ability to load existing results when some methods aren't being run
def load_existing_results(args):
    """Load results from existing JSON file if available"""
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            results = json.load(f)
            print(f"Loaded existing results from {args.output_file}")
            return results
    return {}

def create_separate_comparison_tables(aggregated_results, args):
    """
    Creates separate LaTeX tables comparing delta metrics for each number of predictor variables.
    
    Returns:
        dict: Mapping from a table key (e.g., "x2", "x3") to LaTeX table code.
    """
    # Define the metric keys to display.
    delta_keys = ["ΔCWR", "ΔNMPIW", "ΔCRPS", "ΔPinball", "ΔPICP", "PICP"]
    # Create header names (replace Unicode Δ with '$\Delta$' for LaTeX)
    converted_delta_keys = [f"$\\Delta${key[1:]}" if key.startswith("Δ") else key for key in delta_keys]
    
    tables = {}
    for num_vars in args.num_variables:
        if num_vars not in aggregated_results:
            print(f"Missing num_vars={num_vars} in results")
            continue
            
        rows = []
        # For every baseline, add a header row and then a row for each distribution.
        for baseline in args.model_class:
            # Baseline grouping header row
            row_dict = {"Model+Distribution": baseline}
            for key in delta_keys:
                row_dict[key] = ""
            rows.append(row_dict)
            
            for dist in args.distributions:
                row = {"Model+Distribution": dist}
                try:
                    # Access the 'aggregated' key from the results dictionary
                    if baseline in aggregated_results[num_vars] and dist in aggregated_results[num_vars][baseline]:
                        if 'aggregated' in aggregated_results[num_vars][baseline][dist]:
                            agg = aggregated_results[num_vars][baseline][dist]['aggregated']
                        
                            for key in delta_keys:
                                if key in agg:
                                    value = agg[key]
                                    if key.startswith("Δ"):
                                        # Expecting value to be a tuple: (mean, min, max)
                                        if isinstance(value, tuple) and len(value) == 3:
                                            mean_val, min_val, max_val = value
                                            if np.isnan(mean_val) or np.isnan(min_val) or np.isnan(max_val):
                                                formatted = "-"
                                            else:
                                                formatted = f"{mean_val:.1f}\\% [{min_val:.1f}:{max_val:.1f}]"
                                                if mean_val > 0:
                                                    formatted = f"\\textbf{{{formatted}}}"
                                        else:
                                            formatted = "-"
                                    else:
                                        # For absolute metrics, use a special formatting for PICP (two decimals) and one decimal for the rest.
                                        if isinstance(value, tuple) and len(value) == 2:
                                            mean_val, std_val = value
                                            if np.isnan(mean_val) or np.isnan(std_val):
                                                formatted = "-"
                                            else:
                                                if key == "PICP":
                                                    # Format PICP with two decimal places without a percent sign (or remove it if you prefer).
                                                    formatted = f"{mean_val:.2f}$\\pm${std_val:.2f}"
                                                else:
                                                    formatted = f"{mean_val:.1f}\\%$\\pm${std_val:.1f}\\%"
                                        else:
                                            formatted = "-"
                                    row[key] = formatted
                                else:
                                    row[key] = "-"
                        else:
                            for key in delta_keys:
                                row[key] = "-"
                    else:
                        for key in delta_keys:
                            row[key] = "-"
                except Exception as e:
                    for key in delta_keys:
                        row[key] = "-"
                rows.append(row)
        
        caption = (
            f"Test results for {args.n} observations generated using {args.data_gen_method} "
            f"$f(x)$ with {num_vars} predictor{'s' if num_vars != 1 else ''}, and an additive noise ($\\epsilon$) belonging to one of the four distributions. "
            f"Relative metrics are shown over {len(args.model_seeds)} seeds as "
            f"`mean [min:max]', and absolute metrics as `mean $\\pm$ std'. Average performance over the baseline is highlighted in bold."
        )
        
        header = (
            "\\begin{table}[t]\n"
            "\\centering%\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{tab:synthetic-{args.data_gen_method}-x{num_vars}-results}}\n"
            "\\vskip 0.15in\n"
            "\\begin{small}\n"
            "\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{" + "l" + "c" * len(delta_keys) + "}\n"
            "\\toprule\n"
            " & " + " & ".join(converted_delta_keys) + " \\\\\n"
            "\\midrule\n"
        )
        
        body = ""
        for row in rows:
            if row["Model+Distribution"] in args.model_class:
                body += f"\\multicolumn{{{len(delta_keys)+1}}}{{l}}{{\\textbf{{{row['Model+Distribution']}}}}} \\\\\n"
            else:
                body += f"{row['Model+Distribution']}"
                for key in delta_keys:
                    body += f" & {row[key]}"
                body += " \\\\\n"
        
        footer = (
            "\\bottomrule\n"
            "\\end{tabular}%%\n"
            "}\n"
            "\n"
            # "\\vspace{-0.2in}\n"
            "\\end{small}\n"
            "\\end{table}\n"
        )
        
        table_key = f"{args.data_gen_method}_x{num_vars}"
        tables[table_key] = header + body + footer
    return tables

def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic experiments with SEMF"
    )
    parser.add_argument("--output_file", type=str, default="../../../results/synthetic_experiments_results.json",
                        help="JSON file to save aggregated results.")
    parser.add_argument("--table_file", type=str, default="../../../paper/assets/tex_tbls/synthetic_experiments_results_table.tex",
                        help="LaTeX table file name.")
    parser.add_argument("--data_seed", type=int, default=0,
                        help="Seed for synthetic data generation")
    parser.add_argument("--model_class", type=str, nargs="+", 
                        default=["MultiXGBs", "MultiETs", "MultiMLPs"],
                        help="model_class to run")
    parser.add_argument("--distributions", type=str, nargs="+", 
                        default=["normal", "uniform", "lognormal", "gumbel"],
                        help="Distributions to run")
    parser.add_argument("--model_seeds", type=int, nargs="+", default=[0, 10, 20, 30, 40],
                        help="Seeds for model initialization/training")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of observations")
    parser.add_argument("--num_variables", type=int, nargs="+", default=[2],
                        help="List of numbers of predictor variables")
    parser.add_argument("--baseline_interval_method", type=str, default="quantile",
                        help="Baseline interval method: 'quantile' or 'conformal_point'")
    parser.add_argument("--data_gen_method", type=str, choices=["cosine", "quadratic"], default="cosine",
                        help="Synthetic data generation method: 'cosine' or 'quadratic'")
    parser.add_argument("--use_existing", action="store_true",
                        help="Use existing results where available instead of rerunning")
    # Add optional wandb arguments:
    parser.add_argument("--use_wandb", action="store_true",
                        help="If set, log experiment results with wandb.")
    parser.add_argument("--wandb_project", type=str, default="synthetic_experiments",
                        help="WandB project name (only used if --use_wandb is set).")
    parser.add_argument("--wandb_entity", type=str, default="",
                        help="WandB entity name (optional).")
    
    args = parser.parse_args()
    
    # If wandb logging is enabled, initialize wandb
    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    aggregated_results = {}
    if args.use_existing:
        aggregated_results = load_existing_results(args)
    
    for num_vars in args.num_variables:
        if num_vars not in aggregated_results:
            aggregated_results[num_vars] = {}
        for baseline in args.model_class:
            if baseline not in aggregated_results[num_vars]:
                aggregated_results[num_vars][baseline] = {}
            for dist in args.distributions:
                if dist not in aggregated_results[num_vars][baseline]:
                    agg = aggregate_results(
                        baseline, 
                        dist, 
                        args.model_seeds, 
                        baseline_interval_method=args.baseline_interval_method,
                        data_seed=args.data_seed, 
                        n=args.n, 
                        num_variables=num_vars,
                        data_gen_method=args.data_gen_method
                    )
                    aggregated_results[num_vars][baseline][dist] = agg
                    print(f"\nVariables: {num_vars}, Baseline: {baseline}, Distribution: {dist}")
                    for key, value in agg['aggregated'].items():
                        if isinstance(value, tuple):
                            print(f"{key}: {value[0]:.3f} ± {value[1]:.3f}")
                        else:
                            print(f"{key}: {value:.3f}")
                    # Print raw runs for double-checking:
                    print("Raw runs:")
                    for i, run in enumerate(agg['runs']):
                        print(f"Run {i}:")
                        print(run)
                        print("---------------")
                    print("\n --------------------------------------------------------------")
                    
                    # Log aggregated results (and optionally raw run summaries) to wandb
                    if args.use_wandb and wandb is not None:
                        wandb.log({
                            "aggregated_results": agg["aggregated"],
                            "distribution": dist,
                            "baseline": baseline,
                            "num_variables": num_vars,
                        })
    
    # Save results to JSON with updated file name including data generation method and x variables
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # Split the current output file into its directory, base name and extension
    dirname, filename = os.path.split(args.output_file)
    base, ext = os.path.splitext(filename)
    # Create a string for the number of predictor variables.
    # If args.num_variables is a list, join them by underscores.
    num_vars_str = "_".join(str(x) for x in args.num_variables)
    # Construct the new output file name.
    new_output_file = os.path.join(dirname, f"{base}_{args.data_gen_method}_x{num_vars_str}{ext}")

    with open(new_output_file, "w") as f:
        json.dump(aggregated_results, f, indent=4, default=lambda o: o.item() if hasattr(o, "item") else o)
    print(f"Results saved to {new_output_file}")
    
    # Create and save comparison tables
    separate_tables = create_separate_comparison_tables(aggregated_results, args)
    for table_key, table_tex in separate_tables.items():
        base_dir = os.path.dirname(args.table_file)
        # table_key is of the form: "{args.data_gen_method}_x{num_vars}" (e.g., "cosine_x2")
        new_name = f"table-synthetic-experiments-{table_key.replace('_', '-')}.tex"
        table_file = os.path.join(base_dir, new_name)
        os.makedirs(os.path.dirname(table_file), exist_ok=True)
        with open(table_file, 'w') as f:
            f.write(table_tex)
        print(f"Table for {table_key} saved to {table_file}")
    
    # Finish the wandb run if it's active.
    if args.use_wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main() 

# python synthetic_experiments.py --num_variables 2 --model_class MultiMLPs --n 5000 --distributions normal