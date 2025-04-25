from enum import Enum

class ExperimentMode(Enum):
    """
    Enum for experiment modes. The first word corresponds to the state of the training set,
    and the second word corresponds to the state of the test set. For example, `CLEAN_DIRTY`
    means that the model is trained (fitted) on dirty data and the performance (inference) is
    measured on clean data.
    """
    CLEAN_CLEAN = 1
    CLEAN_DIRTY = 2
    DIRTY_CLEAN = 3
    DIRTY_DIRTY = 4