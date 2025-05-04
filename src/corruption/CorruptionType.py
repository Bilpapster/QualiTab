from enum import Enum
import pandas as pd
import numpy as np

from jenga.corruptions.generic import MissingValues
from jenga.corruptions.numerical import Scaling
from jenga.corruptions.generic import CategoricalShift

class CorruptionType(Enum):
    """
    Enum for corruption types. Currently, it includes:
    - MCAR: Missing Completely At Random
    - SCAR: Scaling Completely At Random
    - CSCAR: Categorial Shift Completely At Random
    The corruption types are inspired by the work of Schelter et al. (Jenga)
    https://github.com/schelterlabs/jenga
    """
    MCAR = 'MCAR' # Missing Completely At Random
    SCAR = 'SCAR' # Scaling Completely At Random
    CSCAR = 'CSCAR' # Categorial Shift Completely At Random

    NONE = 'NONE' # No corruption. Introduced to make the code more readable, especially for the clean-clean mode.


    def get_compatible_corruption_percents_from_candidates(self, candidate_percents: list[int | float]) -> list[int | float]:
        """
        Get the compatible corruption percents from the candidate percents.
        Args:
            candidate_percents (list): List of candidate corruption percents.
        Returns:
            list: List of compatible corruption percents.
        """
        match self:
            case CorruptionType.NONE:
                return [0]
            case _:
                return [percent for percent in candidate_percents if 0 < percent <= 100]


    def corrupt(self, data: pd.DataFrame, random_seed: int | float,
                row_percent: int | float, column_percent: int | float) -> tuple:

        if not 0 <= row_percent <= 100:
            raise ValueError(f"Row corruption percent must be between 0 and 100. Got {row_percent}.")
        if not 0 <= column_percent <= 100:
            raise ValueError(f"Column corruption percent must be between 0 and 100. Got {column_percent}.")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("Data must not be empty.")
        if row_percent == 0 or column_percent == 0:
            # If either row or column percent is 0, return the original data
            return data, [], [] # -> original_data, corrupted_rows (empty), corrupted_columns (empty)
        if self == CorruptionType.NONE:
            # If no corruption should be applied, return the original data
            return data, [], []

        corrupted_data = data.copy(deep=True)

        # setting the np.random.seed (numpy's seed) ensures pandas randomness is reproducible (pd is used within Jenga)
        np.random.seed(random_seed)

        categorical_columns = [column for column in data.columns if isinstance(data[column].dtype, pd.CategoricalDtype)]
        numeric_columns = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column].dtype)]

        corrupted_rows = np.random.choice(data.index.to_list(), max(int(row_percent * data.shape[0] / 100), 1), replace=False)

        # to be used in corruptions that can be applied to any column (e.g. MCAR)
        corrupted_columns_generic = np.random.choice(
            data.columns, max(int(column_percent * data.shape[1] / 100), 1), replace=False
        )
        corrupted_columns_categorical, corrupted_columns_numeric = [], []
        if len(categorical_columns) > 0:
            # to be used in corruptions that can only be applied to categorical columns (e.g. CSCAR)
            corrupted_columns_categorical = np.random.choice(
                categorical_columns, max(int(column_percent * len(categorical_columns) / 100),1),  replace=False
            )

        if len(numeric_columns) > 0:
            # to be used in corruptions that can only be applied to numeric columns (e.g. SCAR)
            corrupted_columns_numeric = np.random.choice(
                numeric_columns, max(int(column_percent * len(numeric_columns) / 100), 1), replace=False
            )

        """
        Dev Note: If we want to add more sampling strategies (and not only completely at random)
        we should use fraction 0.99, because 1 is not a valid fraction for anything else than *Completely* AR.
        """
        match self:
            # corruptions code based on: https://github.com/schelterlabs/jenga/blob/master/notebooks/corruptions.ipynb
            case CorruptionType.MCAR:
                # missing values is generic and can be applied to any column (categorical or numeric)
                corrupted_columns = corrupted_columns_generic
                corrupted_subset = data.loc[corrupted_rows, corrupted_columns]
                for column in corrupted_columns:
                    corrupted_subset[column] = MissingValues(column=column, fraction=1, missingness='MCAR') \
                        .transform(corrupted_subset)[column]

            case CorruptionType.SCAR:
                # scaling is only for numeric columns
                if len(corrupted_columns_numeric) == 0:
                    return data, [], []  # no numeric columns to corrupt; this should be handled in the experiment mode

                corrupted_columns = corrupted_columns_numeric
                corrupted_subset = data.loc[corrupted_rows, corrupted_columns]
                for column in corrupted_columns:
                    corrupted_subset[column] = Scaling(column=column, fraction=1, sampling='MCAR') \
                        .transform(corrupted_subset)[column]

            case CorruptionType.CSCAR:
                # categorical shift is only for categorical columns
                if len(corrupted_columns_categorical) == 0:
                    return data, [], []  # no categorical columns to corrupt; this should be handled in the experiment mode

                corrupted_columns = corrupted_columns_categorical
                corrupted_subset = data.loc[corrupted_rows, corrupted_columns]
                """ We need a (deep) copy, so that (after corruption) we can find the actually corrupted rows. 
                Missing Values & Scaling do not need this; corruption to all specified rows is guaranteed. """
                corrupted_subset_copy = corrupted_subset.copy(deep=True)
                for column in corrupted_columns:
                    corrupted_subset[column] = CategoricalShift(column=column, fraction=1, sampling='MCAR') \
                        .transform(corrupted_subset)[column]

                # find the actually corrupted rows (rows where `corrupted_subset` not_equals `corrupted_subset_copy`)
                corrupted_rows = corrupted_subset[corrupted_subset.ne(corrupted_subset_copy).any(axis=1)].index.to_list()

            case _:
                raise NotImplementedError(f"Corruption for type {self} is not implemented. "
                                          f"Please open us a GitHub issue or (even better) a pull request.")

        corrupted_data.loc[corrupted_rows, corrupted_columns] = corrupted_subset
        try:
            corrupted_rows, corrupted_columns = corrupted_rows.tolist(), corrupted_columns.tolist()
        except AttributeError:
            # it might be the case that corrupted_rows and corrupted_columns are already plain lists (not ndarrays)
            # print(f"DEBUG: Corrupted rows were already lists: {corrupted_rows}")
            # print(f"DEBUG: Corrupted columns were already lists: {corrupted_columns}")
            corrupted_rows, corrupted_columns = list(corrupted_rows), list(corrupted_columns)

        return corrupted_data, corrupted_rows, corrupted_columns