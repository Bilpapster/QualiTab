from enum import Enum

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
        if self == CorruptionType.NONE:
            return [0]
        else:
            return [percent for percent in candidate_percents if 0 < percent <= 100]