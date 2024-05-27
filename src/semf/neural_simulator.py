import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
import warnings
import keras.backend as K
import tensorflow as tf
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MissingDataSimulator:
    """
    A class for simulating missing data in a given dataset and training a neural network to impute the missing values.

    Args:
        df (pandas.DataFrame or numpy.ndarray): The input dataset with missing values.
        R (int): The number of times to repeat the simulation and training process.
        layers (list of int, optional): The number of neurons in each hidden layer of the neural network.
        keras_model (tensorflow.keras.Model, optional): A user-provided Keras model to use instead of the default model.
        optimizer (str, optional): The optimizer to use for training the neural network.
        lr (float, optional): The learning rate for the optimizer.
        epochs (int, optional): The maximum number of epochs to train the neural network.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped.
        val_split (float, optional): The fraction of the data to use for validation during training.
        metrics (list of str, optional): The evaluation metrics to use during training.

    Attributes:
        df (pandas.DataFrame or numpy.ndarray): The input dataset with missing values.
        R (int): The number of sampling operations for the training process.
        layers (list of int or None): The number of neurons in each hidden layer of the neural network.
        user_model (tensorflow.keras.Model or None): A user-provided Keras model to use instead of the default model.
        optimizer (str): The optimizer to use for training the neural network.
        lr (float): The learning rate for the optimizer.
        epochs (int): The maximum number of epochs to train the neural network.
        val_split (float): The fraction of the data to use for validation during training.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): The early stopping callback for training the neural network.
        xi_model (tensorflow.keras.Model or None): The trained neural network for imputing missing values.
        complete_col_names (list of str or None): The names of the columns in the input dataset that have no missing values.
        missing_col_names (list of str or None): The names of the columns in the input dataset that have missing values.
        weights_col_names (list of str or None): The names of the columns in the input dataset that represent weights.
        metrics (list of tensorflow.keras.metrics.Metric): The evaluation metrics to use during training.
        loss (tensorflow.keras.losses): The training loss function.

    Notes:
        - The `test_size` argument is not available here and is essentially 1 - `train_size`.
        - If both `user_model` and `layers` are provided, a ValueError is raised.
    """

    def __init__(
        self,
        df,
        R=10,
        xi_model=None,
        layers=None,
        keras_model=None,
        optimizer="Adam",
        lr=0.2,
        epochs=100,
        batch_size=128,
        patience=5,
        val_split=0.15,
        metrics=["Accuracy"],
        loss=CategoricalCrossentropy(),
    ):
        self.df = df
        self.R = R
        self.layers = layers
        self.user_model = keras_model  # User-provided model
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Check if both user_model and layers are provided
        if self.user_model is not None and self.layers is not None:
            raise ValueError("Only one of 'keras_model' and 'layers' can be provided.")
        # Add early stopping
        self.val_split = val_split
        self.early_stopping = EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        )

        # Initialize other necessary attributes
        self.xi_model = xi_model
        self.complete_col_names = None
        self.missing_col_names = None
        self.weights_col_names = None
        # Use the metric(s) desired by the user
        self.metrics = metrics
        self.metrics = [
            getattr(importlib.import_module("tensorflow.keras.metrics"), metric)
            for metric in self.metrics
        ]
        self.loss = loss

    @staticmethod
    def convert_to_df(data):
        """
        Converts a numpy array to a pandas DataFrame.

        Args:
            data (numpy.ndarray or pandas.DataFrame): The input data.

        Returns:
            pandas.DataFrame: The input data as a DataFrame.
        """
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            return data

    def check_col_names(self, new_df):
        """
        Check and rename the columns of the given DataFrame to match self.df.

        Args:
            new_df (pd.DataFrame): The new DataFrame whose columns need to be checked and
                                   possibly renamed.

        Returns:
            pd.DataFrame: DataFrame with columns renamed if necessary.

        Raises:
            ValueError: If the number of columns in new_df and expected columns do not match.
        """
        remaining_cols = [
            col for col in self.df.columns if col not in self.weights_col_names
        ]
        new_column_order = [
            col
            for col in remaining_cols
            if col in self.complete_col_names or col in self.missing_col_names
        ]
        if set(new_df.columns) != set(new_column_order + self.weights_col_names):
            new_df_cols_without_weight = [
                col for col in new_df.columns if col not in self.weights_col_names
            ]
            if len(new_df_cols_without_weight) != len(new_column_order):
                raise ValueError(
                    "Length of new DataFrame columns and expected columns do not match."
                )
            rename_dict = {
                old: new
                for old, new in zip(new_df_cols_without_weight, new_column_order)
            }
            new_df.rename(columns=rename_dict, inplace=True)
        return new_df

    def _transform_data(self, df):
        """
        Transforms the input dataset into complete and missing data subsets.

        Args:
            df (pandas.DataFrame or numpy.ndarray): The input dataset with missing values.

        Returns:
            dict: A dictionary containing the complete and missing data subsets, as well as the names of the columns with missing values, complete values, and weights.
        """
        df = self.convert_to_df(df)
        missing_col_names = df.columns[df.isnull().any()].tolist()
        weights_col_names = [col for col in df.columns if col.startswith("W_")]
        complete_col_names = list(
            set(df.columns) - set(missing_col_names) - set(weights_col_names)
        )
        index_complete = df[missing_col_names].notnull().all(axis=1)
        complete_data = df.loc[index_complete]
        missing_data = df.loc[~index_complete]
        complete_data = complete_data.reset_index(drop=True)
        missing_data = missing_data.reset_index(drop=True)

        return {
            "complete_data": complete_data,
            "missing_data": missing_data,
            "missing_col_names": missing_col_names,
            "complete_col_names": complete_col_names,
            "weights_col_names": weights_col_names,
        }

    def prepare_xi_data(self):
        """
        Prepares the input data for training the neural network to impute missing values.
        """
        data = self._transform_data(self.df.copy())
        self.complete_data = data["complete_data"]
        self.missing_data = data["missing_data"]
        self.missing_col_names = data["missing_col_names"]
        self.complete_col_names = data["complete_col_names"]
        self.weights_col_names = data["weights_col_names"]

    def transform_new_data(self, new_data):
        """
        Transforms a new dataset into complete and missing data subsets.

        Args:
            new_data (pandas.DataFrame or numpy.ndarray): The new dataset with missing values.

        Returns:
            dict: A dictionary containing the complete and missing data subsets, as well as the names of the columns with missing values, complete values, and weights.
        """
        return self._transform_data(new_data)

    def run_xi_sampling(self, sampling_seed=0):
        """
        Runs the simulation and training process for imputing missing values.

        Args:
            sampling_seed (int, optional): The random seed for sampling. Defaults to 0.

        Returns:
            dict: A dictionary containing the simulated data for training the neural network.
        """
        X2_mat = self.missing_data[self.complete_col_names].values
        X2_mat = np.tile(X2_mat, (self.R, 1))
        if self.xi_model is None:
            self.initialize_xi_model(X2_mat.shape[1], len(self.complete_data))
        j_sim = self.simulate_x_missing(X2_mat, sampling_seed=sampling_seed)["j_star"]
        X1_mat = to_categorical(j_sim, num_classes=len(self.complete_data))

        W_vec = None
        if self.weights_col_names:
            W_vec = self.missing_data[self.weights_col_names].values.flatten()

        return {"X2_mat": X2_mat, "X1_mat": X1_mat, "W_vec": W_vec}

    @staticmethod
    def clear_memory():
        """
        Clears the Keras session and garbage collects.
        """
        K.clear_session()
        gc.collect()

    def simulate_x_missing(self, X2_mat, sampling_seed=None, pred_batch_size=256):
        """
        Simulates missing data in the input dataset.

        Args:
            X2_mat (numpy.ndarray): The input dataset with complete values.
            sampling_seed (int, optional): The random seed to use for generating the uniform samples. Default is None.
            pred_batch_size (int, optional): The batch size to use for predicting the probabilities. Default is 256.

        Returns:
            dict: A dictionary containing the simulated missing data, as well as the indices of the rows and columns with missing values.
        """
        self.clear_memory()
        try:
            proba = self.xi_model.predict(X2_mat, verbose=0, batch_size=pred_batch_size)
        except tf.errors.ResourceExhaustedError:
            print("ResourceExhaustedError: Reducing batch size")
            self.clear_memory()
            proba = self.xi_model.predict(X2_mat, verbose=0, batch_size=32)

        cumulative_proba = np.cumsum(proba, axis=1)
        np.random.seed(sampling_seed)
        uniform_samples = np.random.rand(len(proba), 1)

        j_star = np.sum(cumulative_proba < uniform_samples, axis=1)

        if np.max(j_star) >= len(self.complete_data):
            warnings.warn("Index in j_star is out-of-bounds for self.complete_data. Clipping to the maximum index.")
            j_star = np.clip(j_star, 0, len(self.complete_data) - 1)

        simulated_missing_data = self.complete_data.iloc[j_star][self.missing_col_names].values

        unique_j_sims = len(np.unique(j_star))
        percentage_unique = (unique_j_sims / len(self.complete_data)) * 100
        print(
            f"      *** There are {unique_j_sims} unique simulated rows ({percentage_unique:.2f}% of the training data). The indices are {', '.join(map(str, np.unique(j_star)[:max(1, 3)]))},..., {', '.join(map(str, np.unique(j_star)[-max(1, 3):])) if len(np.unique(j_star)) > 1 else 'Only one unique index: ' + str(np.unique(j_star)[0])}"
        )

        return {"simulated_missing_data": simulated_missing_data, "j_star": j_star}

    def initialize_xi_model(self, input_shape, output_shape):
        """
        Initializes the neural network model for imputing missing values.

        Args:
            input_shape (int): The number of input features.
            output_shape (int): The number of output classes.
        """
        inputs = Input(shape=(input_shape,))
        if self.user_model is not None:
            self.xi_model = self.user_model
        elif self.layers is not None:
            x = inputs
            for layer in self.layers:
                x = Dense(layer["units"], activation=layer["activation"])(x)
            outputs = Dense(output_shape, activation="softmax")(x)
            self.xi_model = Model(inputs=inputs, outputs=outputs)
        else:
            x = Dense(100, activation="selu")(inputs)
            outputs = Dense(output_shape, activation="softmax")(x)
            self.xi_model = Model(inputs=inputs, outputs=outputs)

    def train_xi_model(self, X2_mat, X1_mat, W_vec=None):
        """
        Trains the neural network model for imputing missing values.

        Args:
            X2_mat (numpy.ndarray): The input features for training.
            X1_mat (numpy.ndarray): The target values for training.
            W_vec (numpy.ndarray, optional): The sample weights for training. Default is None.
        """
        self.initialize_xi_model(X2_mat.shape[1], X1_mat.shape[1])
        try:
            Opt = getattr(importlib.import_module("tensorflow.keras.optimizers"), self.optimizer)
        except AttributeError:
            raise ValueError(f"Invalid optimizer. Check if the optimizer '{self.optimizer}' exists in tensorflow.keras.optimizers.")
        opt = Opt(learning_rate=self.lr)
        self.xi_model.compile(
            optimizer=opt,
            loss=self.loss,
            weighted_metrics=[metric() for metric in self.metrics],
        )
        indented_print_callback = self.CustomPrintCallback()
        self.xi_model.fit(
            X2_mat,
            X1_mat,
            sample_weight=W_vec,
            epochs=self.epochs,
            validation_split=self.val_split,
            callbacks=[self.early_stopping, indented_print_callback],
            batch_size=self.batch_size,
            verbose=0,
        )

    def replace_na_in_df(self, new_data=None, sampling_seed=None):
        """
        Replaces missing values in a dataset with imputed values.

        Args:
            new_data (pandas.DataFrame or numpy.ndarray, optional): The new dataset with missing values. Default is None.
            sampling_seed (int, optional): The random seed for sampling. Note that if you fix this, the missing data simulator will always draw the same random values which is not ideal. To reproduce the results in our experiments, the iteration `i` in `semf.SEMF` increments the `seed`+`i`. Default is None.

        Returns:
            pandas.DataFrame: The dataset with simulated (imputed) values.
        """
        if new_data is not None:
            df_copy = new_data.copy()
            df_copy = self.convert_to_df(df_copy)
            df_copy = self.check_col_names(df_copy)
        else:
            df_copy = self.df.copy()
            df_copy = self.df.drop(columns=self.weights_col_names)
        missing_mask = df_copy[self.missing_col_names].isnull()
        missing_rows = missing_mask.any(axis=1).values
        missing_data = df_copy.loc[missing_rows, self.complete_col_names]
        simulated_missing_data = self.simulate_x_missing(missing_data.values, sampling_seed=sampling_seed)["simulated_missing_data"]

        simulated_df = pd.DataFrame(
            simulated_missing_data,
            index=missing_data.index,
            columns=self.missing_col_names,
        )
        df_copy.fillna(simulated_df, inplace=True)
        return df_copy

    def plot_predictions(self, an_instance, pred_batch_size=256):
        """
        Plots the predicted and actual values for a given column in the input dataset.

        Args:
            an_instance (pandas.DataFrame or numpy.ndarray): The input dataset with missing values.
            pred_batch_size (int, optional): The batch size to use for predicting the probabilities. Default is 256.

        Returns:
            matplotlib.pyplot.figure: The figure object containing the plot.

        Raises:
            ValueError: If no model is found or if there are no missing inputs to plot.
        """
        if self.xi_model is None:
            raise ValueError("No model found. Please train the model first.")
        an_instance = self.transform_new_data(an_instance)
        missing_rows = an_instance["missing_data"].any(axis=1).values
        if not np.any(missing_rows):
            raise ValueError("No missing inputs found, cannot proceed with plotting.")
        an_instance = an_instance["missing_data"].loc[missing_rows, self.complete_col_names].values
        probabilities = self.xi_model.predict(an_instance, verbose=0, batch_size=pred_batch_size)
        for i, prob in enumerate(probabilities):
            flat_prob = np.array(prob).flatten()
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(flat_prob)), flat_prob, color="blue")
            plt.title(f"Predicted probabilities for instance {i+1}")
            plt.xlabel("Instance")
            plt.ylabel("Probability")
            plt.show()

    class CustomPrintCallback(Callback):
        """
        A custom callback to provide indented, human-readable logs during training.

        This callback is designed to run at the end of each epoch and prints an indented summary of key metrics like
        loss and accuracy for both training and validation sets.

        Methods:
            on_epoch_end(epoch, logs): Prints indented logs for training and validation metrics.
        """

        def on_epoch_end(self, epoch, logs=None):
            """
            Called at the end of each epoch during training.

            Args:
                epoch (int): The current epoch number.
                logs (dict, optional): A dictionary containing the training and validation metrics. Default is None.
            """
            logs = logs or {}
            train_loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            train_acc = logs.get("accuracy")
            val_acc = logs.get("val_accuracy")

            print(
                f"      *** Epoch {epoch+1} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, train_accuracy: {train_acc:.4f}, val_accuracy: {val_acc:.4f}"
            )


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.normal(size=(1000, 4))
    mask = np.random.rand(*X.shape) < 0.5
    keep_columns = [0, 2]
    mask[:, keep_columns] = False
    X[mask] = np.nan
    df = pd.DataFrame(X, columns=[f"col{i+1}" for i in range(X.shape[1])])
    print(df)
    simulator = MissingDataSimulator(df)
    simulator.prepare_xi_data()
    data = simulator.run_xi_sampling()
    simulator.train_xi_model(data["X2_mat"], data["X1_mat"])
    df_simulated = simulator.replace_na_in_df()
    print(df)
    print("--------------------")
    np.random.seed(1)
    X = np.random.normal(size=(1000, 4))
    mask = np.random.rand(*X.shape) < 0.5
    keep_columns = [0, 2]
    mask[:, keep_columns] = False
    X[mask] = np.nan
    df_2 = pd.DataFrame(X)
    print(df_2)
    print(simulator.replace_na_in_df(df_2))
