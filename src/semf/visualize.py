import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pandas as pd
import functools
import matplotlib.patches as mpatches

def plot_decorator(func):
    """
    A decorator to handle common plotting functionalities.

    Args:
        func (function): The plotting function to be decorated.

    Returns:
        function: The wrapped function with additional functionality.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return_fig = kwargs.pop('return_fig', False)
        save_fig = kwargs.pop('save_fig', False)
        fig_path = kwargs.pop('fig_path', None)
        result = func(*args, **kwargs)
        plt.tight_layout()
        if save_fig and fig_path:
            plt.savefig(fig_path, dpi=300)
        if return_fig:
            return plt.gcf()
        else:
            plt.show()
        return result
    return wrapper

@plot_decorator
def visualize_prediction_intervals(y_preds, y_true=None, central_tendency="mean"):
    """
    Visualizes prediction intervals for multiple test instances using boxplots.

    Args:
        y_preds (np.ndarray): The matrix of predicted y-values, where each row represents a test instance and each column is a prediction from an ensemble member.
        y_true (np.ndarray, optional): The true y-values corresponding to the instances. Default is None.
        central_tendency (str, optional): A string indicating whether to plot 'mean' or 'median'. Default is 'mean'.
    """
    plt.figure(figsize=(12, 6))

    df = pd.DataFrame(y_preds)
    df['instance'] = df.index
    long_data = df.melt(id_vars='instance', var_name='ensemble_member', value_name='prediction')
    
    if central_tendency == "mean":
        y_central = np.mean(y_preds, axis=1)
        label = "Predicted Mean"
    else:
        y_central = np.median(y_preds, axis=1)
        label = "Predicted Median"

    sns.boxplot(x=long_data['instance'], y=long_data['prediction'], color='skyblue', fliersize=0, whis=[5, 95])
    
    if y_true is not None:
        plt.scatter(np.arange(y_true.shape[0]), y_true, color='red', s=10, marker="x", label='True Values')
    
    plt.scatter(np.arange(y_central.shape[0]), y_central, color='blue', marker='o', s=10, label=label)

    data_length = y_true.shape[0] if y_true is not None else y_preds.shape[0]
        
    if data_length <= 10:
        interval = 1
    elif data_length <= 50:
        interval = 10
    else:
        interval = 20
    
    ticks = list(np.arange(0, data_length, interval))
    if data_length - 1 not in ticks:
        ticks.append(data_length - 1)
    tick_labels = [str(tick) for tick in ticks]
    tick_labels[-1] = str(int(tick_labels[-1]) + 1)
    plt.xticks(ticks, tick_labels)
    
    plt.title('Prediction Intervals (95%) and Actual Values')
    plt.xlabel('Observation')
    plt.ylabel('Value')
    plt.legend(loc='upper right')


@plot_decorator
def visualize_prediction_intervals_kde(y_preds, y_true=None, central_tendency="mean"):
    """
    Visualizes prediction intervals for the data using KDE plots.

    Args:
        y_preds (list): The list of predicted y-values for that test case.
        y_true (float, optional): The true y-values (should be a scalar for a single test case). Defaults to None.
        central_tendency (str, optional): A string indicating whether to plot 'mean' or 'median'. Default is 'mean'.
    """
    plt.figure(figsize=(12, 12))

    sns.kdeplot(y_preds, fill=True, label='Predicted Distribution', color='blue')
    
    if central_tendency == "mean":
        y_central = np.mean(y_preds)
        label = "Predicted Mean"
    else:
        y_central = np.median(y_preds)
        label = "Predicted Median"
    
    y_mode = st.mode(y_preds).mode
    
    plt.axvline(y_central, color='blue', linestyle='--', label=label)
    plt.axvline(y_mode, color='purple', linestyle='--', label='Predicted Mode')
    
    min_limit = min(np.min(y_preds), y_true) if y_true is not None else np.min(y_preds)
    max_limit = max(np.max(y_preds), y_true) if y_true is not None else np.max(y_preds)
    
    if y_true is not None:
        plt.axvline(y_true, color='red', linestyle='--', label='True Value')

    padding = 0.25 * (max_limit - min_limit)
    plt.xlim([min_limit - padding, max_limit + padding])
    plt.title('Kernel Density Estimation of Prediction Intervals and Actual (True) Values')
    plt.legend()

@plot_decorator
def visualize_prediction_intervals_kde_multiple(y_preds, y_true=None, n_instances=None, central_tendency="mean"):
    """
    Visualizes horizontal prediction intervals for multiple instances using KDE plots.

    Args:
        y_preds (np.ndarray): The matrix of predicted y-values, where each row represents a test instance and each column is a prediction from an ensemble member.
        y_true (np.ndarray, optional): The true y-values corresponding to the instances. Default is None.
        n_instances (int, optional): The number of instances to plot. If None, all instances will be plotted. Default is None.
        central_tendency (str, optional): A string indicating whether to plot 'mean' or 'median'. Default is 'mean'.
    """
    if n_instances is None:
        n_instances = y_preds.shape[0]
    else:
        n_instances = min(n_instances, y_preds.shape[0])

    fig, axes = plt.subplots(n_instances, figsize=(12, 2 * n_instances))

    if n_instances == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        instance_preds = y_preds[idx, :]
        
        sns.kdeplot(y=instance_preds, fill=True, label='Predicted Distribution', color='blue', ax=ax)
        
        if central_tendency == "mean":
            y_central = np.mean(instance_preds)
            label = "Predicted Mean"
        else:
            y_central = np.median(instance_preds)
            label = "Predicted Median"
        
        ax.axhline(y_central, color='blue', linestyle='--', label=label)
        
        if y_true is not None:
            ax.axhline(y_true[idx], color='red', linestyle='--', label='True Value')

        ax.set_title(f'Instance {idx+1}')
        ax.legend()
        
    plt.tight_layout()
    return fig

def plot_violin(y_preds, y_true, n_instances=None, save = False):
    """
    Plots violin plots for prediction intervals.

    Args:
        y_preds (np.ndarray): The matrix of predicted y-values.
        y_true (np.ndarray): The true y-values.
        n_instances (int, optional): The number of instances to plot. If None, all instances will be plotted. Default is None.
    """
    n_instances = n_instances or y_preds.shape[0]

    instance_ids = np.repeat(np.arange(n_instances), y_preds.shape[1])
    predictions = y_preds[:n_instances].flatten()
    data = pd.DataFrame({'Instance': instance_ids, 'Prediction': predictions})

    sns.set_theme(style="whitegrid")
    
    f, ax = plt.subplots(figsize=(12, 10))

    legend_elements = []

    violin = sns.violinplot(x="Instance", y="Prediction", data=data, inner=None, palette="Set3", linewidth=3)

    stripplot = sns.stripplot(x="Instance", y="Prediction", data=data, size=4, color=".3", linewidth=0, jitter=True)

    if isinstance(y_true, pd.Series):
        y_axis_label = y_true.name
    elif isinstance(y_true, pd.DataFrame) and y_true.columns.size == 1:
        y_axis_label = y_true.columns[0]
    else:
        y_axis_label = "Values"

    if y_true is not None:
        true_points = plt.scatter(np.arange(n_instances), y_true[:n_instances], color="#566D93", label='True Value')
        legend_elements.append(mpatches.Patch(color='#566D93', label='True Value'))

    legend_elements.append(mpatches.Patch(color='.3', label='Predicted Values'))

    ax.legend(handles=legend_elements, loc='upper right')

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
    ax.minorticks_on()

    sns.despine(left=True, bottom=True)

    ax.set_xlabel("Test Instance", size=16, alpha=0.7)
    ax.set_ylabel("gt_compressor_decay_state_coefficient", size=16, alpha=0.7)

    if save:
        plt.savefig('violin_plot.png')
    plt.show()

def get_confidence_intervals(y_preds, y_true):
    """
    Calculate confidence intervals based on SEMF predictions.

    Args:
        y_preds (np.ndarray): A 2D numpy array where each row corresponds to the multiple predictions for a single true value.
        y_true (np.ndarray): A numpy array of true y-values.

    Returns:
        pd.DataFrame: A DataFrame with columns ['observation', 'y_test', 'y_mean', 'y_median', 'y_lower', 'y_upper'].
    """
    y_mean = np.mean(y_preds, axis=1)
    y_median = np.median(y_preds, axis=1)
    y_lower = np.percentile(y_preds, 2.5, axis=1)
    y_upper = np.percentile(y_preds, 97.5, axis=1)

    df = pd.DataFrame({
        'observation': np.arange(1, len(y_true) + 1),
        'y_test': y_true.flatten(),
        'y_mean': y_mean,
        'y_median': y_median,
        'y_lower': y_lower,
        'y_upper': y_upper
    })
    
    return df


@plot_decorator
def plot_confidence_intervals(df, n_instances, central_tendency="mean"):
    """
    Plots the true values, the central tendency of predicted values, and their confidence intervals for a specified number of test data points.

    Args:
        df (pd.DataFrame): A DataFrame containing columns ['observation', 'y_test', 'y_mean', 'y_median', 'y_lower', 'y_upper'].
        n_instances (int): An integer indicating the number of test data points to plot.
        central_tendency (str, optional): A string indicating whether to plot 'mean' or 'median'. Default is 'mean'.
    """
    n_instances = min(n_instances, df.shape[0])
    
    plt.figure(figsize=(6, 6))

    data = df.iloc[:n_instances].copy()
    
    data['outside_ci'] = (data['y_test'] < data['y_lower']) | (data['y_test'] > data['y_upper'])
    
    plt.fill_between(data['observation'], data['y_lower'], data['y_upper'], color='skyblue', alpha=0.15, label='Confidence Interval')
    
    plt.scatter(data['observation'], data['y_test'], color='red', marker='x', s=14, label='True Value')
    
    if central_tendency == "mean":
        plt.scatter(data['observation'], data['y_mean'], color='blue', marker='o', s=14, label='Predicted Mean')
    else:
        plt.scatter(data['observation'], data['y_median'], color='blue', marker='o', s=14, label='Predicted Median')

    plt.title('Predictions with Confidence Intervals (95%)')
    plt.xlabel('Observation')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

@plot_decorator
def plot_tr_val_metrics(semf, optimal_i_value=None, metrics=['MAE', 'RMSE', 'R2']):
    """
    Plots MAE, RMSE, and R2 metrics for training and validation sets.

    Args:
        semf: The SEMF object containing train_perf and valid_perf attributes.
        optimal_i_value (int, optional): The iteration after which early stopping is triggered. None if not provided.
        metrics (list, optional): The metrics to show for the plots. Defaults to `MAE`, `RMSE` and `R2`.
    """
    _, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))

    for i, metric in enumerate(metrics):
        train_values = [item[metric] for item in semf.train_perf]
        axs[i].plot(range(1, len(train_values) + 1), train_values, '-o', label='Train')
        
        valid_values = [item[metric] for item in semf.valid_perf]
        axs[i].plot(range(1, len(valid_values) + 1), valid_values, '-o', label='Validation')
        
        if optimal_i_value is not None:
            axs[i].axvline(optimal_i_value + 1, color='red', linestyle='--')
        
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].legend()
        
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[len(axs) - 1].set_xlabel('Iteration')

if __name__ == "__main__":
    # Example data
    y_preds = np.random.rand(10, 100)
    y_true = np.random.rand(10)

    # Visualize prediction intervals
    visualize_prediction_intervals(y_preds, y_true)

    # Visualize KDE of prediction intervals
    visualize_prediction_intervals_kde(y_preds[0], y_true[0])

    # Visualize KDE for multiple instances
    visualize_prediction_intervals_kde_multiple(y_preds, y_true, n_instances=5)

    # Plot violin plot
    plot_violin(y_preds, y_true, n_instances=5)

    # Get confidence intervals
    df_confidence_intervals = get_confidence_intervals(y_preds, y_true)

    # Plot confidence intervals
    plot_confidence_intervals(df_confidence_intervals, n_instances=5)
