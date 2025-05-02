from .CorruptionType import CorruptionType


class Corruption():
    def __init__(
            self,
            type: CorruptionType,
            percent: int | float = 100,
            random_seed: int | None = None,
            name: str | None = None,
            description: str | None = None
    ):
        self.corruption_type = type
        self.corruption_percent = percent
        self.corruption_seed = random_seed

        self.corruption_name = name
        self.corruption_description = description

    @property
    def corruption_type(self):
        return self._corruption_type

    @corruption_type.setter
    def corruption_type(self, corruption_type):
        """
        Set the corruption type. The value must be an instance of CorruptionType enum.
        """
        try:
            if corruption_type not in CorruptionType:
                raise ValueError(
                    f"Invalid corruption type: {corruption_type}. Currently supported types: {list(CorruptionType)}.")
            self._corruption_type = corruption_type
        except TypeError:
            raise TypeError("Corruption type must be an instance of CorruptionType enum.")
        except ValueError:
            raise ValueError(
                f"Invalid corruption type: {corruption_type}. Currently supported types: {list(CorruptionType)}.")

    @property
    def corruption_percent(self):
        """
        Get the corruption percentages.
        Returns a copy of the list to prevent external modification.
        """
        return self._corruption_percent

    @corruption_percent.setter
    def corruption_percent(self, corruption_percent: int | float):
        """
        Set the corruption percentages. The value must be an integer or float in the range (0, 100].
        """
        if not isinstance(corruption_percent, int | float):
            raise TypeError("Corruption percentage must be an integer or float.")
        if not (0 < corruption_percent <= 100):
            raise ValueError("Corruption percentage must be in the range (0, 100].")
        self._corruption_percent = corruption_percent

    @property
    def corruption_seed(self):
        """
        Get the corruption seed.
        """
        return self._corruption_seed

    @corruption_seed.setter
    def corruption_seed(self, corruption_seed: int | float):
        """
        Set the corruption seed. The value must be an integer or float.
        """
        self._corruption_seed = corruption_seed

    @property
    def corruption_name(self):
        """
        Get the name of the corruption configuration.
        """
        return self._corruption_name

    @corruption_name.setter
    def corruption_name(self, name: str):
        """
        Set the name of the corruption configuration. The name must be a string.
        """
        self._corruption_name = name

    @property
    def corruption_description(self):
        """
        Get the description of the corruption configuration.
        """
        return self._corruption_description

    @corruption_description.setter
    def corruption_description(self, description: str):
        """
        Set the description of the corruption configuration. The description must be a string.
        """
        self._corruption_description = description

    def corrupt(self, data, column: str):
        if self.corruption_type is None:
            raise ValueError("Corruption type must be set before corrupting data.")

        if self.corruption_percent is None:
            raise ValueError("Corruption percentage must be set before corrupting data.")

        match self.corruption_type:
            case CorruptionType.MCAR:
                self._corrupt_with_mcar(data, column)
            case CorruptionType.SCAR:
                self._corrupt_with_scar(data, column)
            case CorruptionType.CSCAR:
                self._corrupt_with_cscar(data, column)
            case _:
                raise ValueError(f"Unsupported corruption type: {self.corruption_type}.")

    def _corrupt_with_mcar(self, data, column: str):
        """
        Apply MCAR corruption to the data using Jenga.
        MCAR stands for Missing Completely At Random.
        Based on: https://github.com/schelterlabs/jenga/blob/master/notebooks/corruptions.ipynb

        Raises:
            ValueError: If the column is not found in the data.
        """
        from jenga.corruptions.generic import MissingValues

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        data[column] = MissingValues(
            column=column,
            fraction=self.corruption_percent,
            missingness="MCAR"
        ).transform(data)[column]
        return data

    def _corrupt_with_scar(self, data, column: str):
        """
        Apply SCAR corruption to the data using Jenga.
        SCAR stands for Scaling Completely At Random.
        Corruption implementation can be found here:
        https://github.com/schelterlabs/jenga/blob/a8bd74a588176e64183432a0124553c774adb20d/src/jenga/corruptions/numerical.py#L29
        """
        from jenga.corruptions.numerical import Scaling

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        data[column] = Scaling(
            column=column,
            fraction=self.corruption_percent,
            sampling="CAR"
        )
        return data

    def _corrupt_with_cscar(self, data, column: str):
        """
        Apply CSCAR corruption to the data using Jenga.
        CSCAR stands for Categorical Shift Completely At Random.
        Corruption implementation can be found here:
        https://github.com/schelterlabs/jenga/blob/a8bd74a588176e64183432a0124553c774adb20d/src/jenga/corruptions/generic.py#L91
        """
        from jenga.corruptions.generic import CategoricalShift

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        data[column] = CategoricalShift(
            column=column,
            fraction=self.corruption_percent,
            sampling="CSCAR"
        ).transform(data)[column]
        return data
