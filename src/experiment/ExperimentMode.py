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


