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
