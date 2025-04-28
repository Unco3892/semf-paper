import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
)
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin


class DataPreprocessor:
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        y_col: Union[str, int] = None,
        train_size: float = 0.5,
        valid_size: float = 0.25,
        seed: int = 0
    ):
        """
        This class is used for processing the data in such a way that our framework accepts.

        Args:
            df (Union[pd.DataFrame, np.ndarray]): The input data to be preprocessed.
            y_col (Union[str, int]): The name of the target variable column.
            train_size (float, optional): The proportion of the dataset to include in the train split (default is 0.5).
            valid_size (float, optional): The proportion of the training dataset to include in the validation split (default is 0.25).
            seed (int, optional): Random seed for data splitting. If None, the split will be random each time.
        Notes:
            The `test_size` argument is not available here and is essentially 1 - `train_size`.
        """
        self.df = pd.DataFrame(df) if isinstance(df, np.ndarray) else df.copy()
        self.y_col = y_col
        self.train_size = train_size
        self.valid_size = valid_size
        self.seed = seed
        self.df_train = None
        self.df_valid = None
        self.df_test = None
        self.check_for_duplicates()

    def check_for_duplicates(self):
        """
        Checks for duplicated rows in the DataFrame and removes them.
        """
        duplicate_rows = self.df[self.df.duplicated()]
        if not duplicate_rows.empty:
            print("Warning: The following duplicated rows are found in the dataset:")
            print(duplicate_rows)
            # Removing duplicates
            self.df = self.df.drop_duplicates()
            print("Duplicated rows have been removed.")
        else:
            print("No duplicated rows found.")

    def split_data(self, df_test: Union[pd.DataFrame, np.ndarray] = None):
        """
        Splits the input dataframe into training, validation, and testing datasets.

        Args:
            df_test (Union[pd.DataFrame, np.ndarray], optional): The testing set data (default is None).
        """
        if df_test is None:
            train_idx, test_idx = train_test_split(
                self.df.index, train_size=self.train_size, random_state=self.seed
            )
            self.df_train = self.df.loc[train_idx]
            self.df_test = self.df.loc[test_idx]
        else:
            df_test = (
                pd.DataFrame(df_test)
                if isinstance(df_test, np.ndarray)
                else df_test.copy()
            )
            self.df_train = self.df
            self.df_test = df_test

        train_idx, valid_idx = train_test_split(
            self.df_train.index, train_size=(1 - self.valid_size), random_state=self.seed
        )
        self.df_valid = self.df_train.loc[valid_idx].reset_index(drop=True)
        self.df_train = self.df_train.loc[train_idx].reset_index(drop=True)

    def scale_data(
        self,
        scaler: TransformerMixin = None,
        scale_output: bool = False,
        output_scaler: TransformerMixin = None,
    ):
        """
        Scales the data in the dataframe using a specified scaler for predictors and optionally a different scaler for the output.

        Args:
            scaler (sklearn.base.TransformerMixin, optional): The scaler to use for predictors. If None, StandardScaler is used. (default is None)
            scale_output (bool, optional): Whether to scale the output column. (default is False)
            output_scaler (sklearn.base.TransformerMixin, optional): The scaler to use for the output column. If None and scale_output is True, StandardScaler is used. (default is None)

        Raises:
            ValueError: If the data has not been split yet.
        """
        if self.df_train is None or self.df_valid is None or self.df_test is None:
            raise ValueError(
                "Data has not been split yet. Please call split_data method before calling scale_data."
            )

        if scaler is None:
            scaler = StandardScaler()

        # Scale predictors
        for column in self.df_train.columns.difference([self.y_col]):
            scaler.fit(self.df_train[column].dropna().values.reshape(-1, 1))
            for df in [self.df_train, self.df_valid, self.df_test]:
                non_na_idx = df[column].notna()
                df[column] = df[column].astype(float)
                df.loc[non_na_idx, column] = scaler.transform(
                    df.loc[non_na_idx, column].values.reshape(-1, 1)
                ).ravel()

        # Optionally scale the output column
        if scale_output:
            if output_scaler is None:
                output_scaler = StandardScaler()

            output_scaler.fit(self.df_train[self.y_col].dropna().values.reshape(-1, 1))
            for df in [self.df_train, self.df_valid, self.df_test]:
                non_na_idx = df[self.y_col].notna()
                df[self.y_col] = df[self.y_col].astype(float)
                df.loc[non_na_idx, self.y_col] = output_scaler.transform(
                    df.loc[non_na_idx, self.y_col].values.reshape(-1, 1)
                ).ravel()

    def get_train_valid_test(self):
        """
        Returns the training, validation, and testing datasets.

        Returns:
            pd.DataFrame: The training dataset.
            pd.DataFrame: The validation dataset.
            pd.DataFrame: The testing dataset.

        Raises:
            ValueError: If the data has not been split yet.
        """
        if self.df_train is None or self.df_valid is None or self.df_test is None:
            raise ValueError(
                "Data has not been split yet. Please call split_data method before calling get_train_valid_test."
            )
        return self.df_train, self.df_valid, self.df_test

    def split_X_y(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Splits a dataframe into X (features) and y (target) dataframes.

        Args:
            df (Union[pd.DataFrame, np.ndarray]): The dataframe to split.

        Returns:
            pd.DataFrame: The X (features) dataframe.
            pd.DataFrame: The y (target) dataframe.
        """
        df_X = df.drop(columns=[self.y_col])
        df_y = df[[self.y_col]]
        return df_X, df_y


# Consider using numpy arrays here instead
if __name__ == "__main__":
    np.random.seed(1)

    # Simulate data
    data_sim = pd.DataFrame(
        np.random.randint(0, 10000, size=(10000, 4)), columns=list("ABCD")
    )

    # Let's check the first few rows of the data
    data_sim.head()

    def print_data_completeness(df_train, df_valid, df_test):
        """
        Prints the completeness of the training, validation, and testing datasets.

        Args:
            df_train (pd.DataFrame): The training dataset.
            df_valid (pd.DataFrame): The validation dataset.
            df_test (pd.DataFrame): The testing dataset.
        """
        print("Train data completeness: ", len(df_train.dropna()) / len(df_train))
        print("Validation data completeness: ", len(df_valid.dropna()) / len(df_valid))
        print("Test data completeness: ", len(df_test.dropna()) / len(df_test))

    def print_missing_rate(df_train, df_valid, df_test):
        """
        Prints the missing rate of the training, validation, and testing datasets.

        Args:
            df_train (pd.DataFrame): The training dataset.
            df_valid (pd.DataFrame): The validation dataset.
            df_test (pd.DataFrame): The testing dataset.
        """
        print("Train data missing rate: ", df_train.isna().mean().mean())
        print("Validation data missing rate: ", df_valid.isna().mean().mean())
        print("Test data missing rate: ", df_test.isna().mean().mean())

    print("===============================================================")

    data = data_sim.copy()
    rate = 0

    # with only training data
    preprocessor = DataPreprocessor(data, y_col="D")
    preprocessor.split_data()
    preprocessor.scale_data()
    # Get the train, validation, and test sets
    df_train, df_valid, df_test = preprocessor.get_train_valid_test()
    # print(df_train, df_valid, df_test)
    print_data_completeness(df_train, df_valid, df_test)
    print_missing_rate(df_train, df_valid, df_test)

    print("===============================================================")

    data = data_sim.copy()

    # missing rate at 50%
    rate = 0.2

    # with test data
    preprocessor = DataPreprocessor(data, y_col="D")
    preprocessor.split_data()
    # Get the train, validation, and test sets
    df_train, df_valid, df_test = preprocessor.get_train_valid_test()
    # print(df_train, df_valid, df_test)
    print_data_completeness(df_train, df_valid, df_test)
    # print_missing_rate(df_train, df_valid, df_test)

    print("===============================================================")
    # test whether the splitting method works
    # x_train, y_train = preprocessor.split_X_y(df_train)
    # x_valid, y_valid = preprocessor.split_X_y(df_valid)
    # x_test, y_test = preprocessor.split_X_y(df_test)
    # print(x_test, y_test)

