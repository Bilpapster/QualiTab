from .Corruption import Corruption


class CorruptionsManager:
    def __init__(self, corruptions: list[Corruption] = None):
        """
        Initialize the CorruptionManager with a list of Corruption objects.
        Args:
            corruptions (list[Corruption]): List of Corruption objects.
        """
        self._corruptions = corruptions

    @property
    def corruptions(self):
        """
        Get the list of corruptions.
        Returns:
            list[Corruption]: List of Corruption objects.
        """
        import copy

        return copy.deepcopy(self._corruptions) if self._corruptions else None

    @corruptions.setter
    def corruptions(self, corruptions: list[Corruption]):
        """
        Set the list of corruptions.
        Args:
            corruptions (list[Corruption]): List of Corruption objects.
        """
        if not isinstance(corruptions, list):
            raise TypeError("Corruptions must be a list.")

        if not all(isinstance(corruption, Corruption) for corruption in corruptions):
            raise TypeError("All elements in the list must be instances of the Corruption class.")

        self._corruptions = corruptions

    def add_corruption(self, corruption: Corruption):
        """
        Add a Corruption object to the list of corruptions.
        """
        if not isinstance(corruption, Corruption):
            raise TypeError("Corruption must be an instance of the Corruption class.")

        if self._corruptions is None:
            self._corruptions = []

        self._corruptions.append(corruption)
