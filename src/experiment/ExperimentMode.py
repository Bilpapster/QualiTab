from enum import Enum

from corruption import CorruptionType


class ExperimentMode(Enum):
    """
    Enum for experiment modes. The first word corresponds to the state of the training set,
    and the second word corresponds to the state of the test set. For example, `CLEAN_DIRTY`
    means that the model is trained (fitted) on dirty data and the performance (inference) is
    measured on clean data.
    """
    CLEAN_CLEAN = 'CLEAN_CLEAN'
    CLEAN_DIRTY = 'CLEAN_DIRTY'
    DIRTY_CLEAN = 'DIRTY_CLEAN'
    DIRTY_DIRTY = 'DIRTY_DIRTY'

    def get_compatible_corruptions_from_candidates(self, candidate_corruptions: list[CorruptionType]) -> list[CorruptionType]:
        match self:
            case ExperimentMode.CLEAN_CLEAN:  # for CLEAN_CLEAN mode, return no corruptions
                return [CorruptionType.NONE]
            case _: # for any other experiment mode, return the intersection of candidate and compatible corruptions
                compatible_corruptions = {CorruptionType.MCAR, CorruptionType.SCAR, CorruptionType.CSCAR}
                return [corruption for corruption in candidate_corruptions if corruption in compatible_corruptions]


    def is_meaningful(self, corrupted_rows: list, corrupted_columns: list) -> bool:
        """
        Check if the experiment mode is meaningful based on the corrupted rows and columns.
        Args:
            corrupted_rows (list): List of corrupted rows.
            corrupted_columns (list): List of corrupted columns.
        Returns:
            bool: True if the experiment mode is meaningful, False otherwise.
        """
        from itertools import chain

        try:
            corrupted_rows = list(chain.from_iterable(corrupted_rows)) # flatten the list of lists
            corrupted_columns = list(chain.from_iterable(corrupted_columns)) # flatten the list of lists
        except TypeError:
            # Flattening fails if corrupted_rows or corrupted_columns are not lists of lists. We are OK with that.
            pass

        # print(f"DEBUG: Corrupted rows inside is_meaningful: {corrupted_rows}")
        # print(f"DEBUG: Corrupted columns inside is_meaningful: {corrupted_columns}")

        match self:
            case ExperimentMode.CLEAN_CLEAN:
                return len(corrupted_rows) == 0 and len(corrupted_columns) == 0
            case _:
                return len(corrupted_rows) > 0 and len(corrupted_columns) > 0
