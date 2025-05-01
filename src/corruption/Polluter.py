from abc import ABC, abstractmethod


class Polluter(ABC):

    def __init__(
            self, polluter_name: str = None,
            polluter_type: str = None,
            polluter_description: str = None
    ):
        self.polluter_name = polluter_name
        self.polluter_type = polluter_type
        self.polluter_description = polluter_description

    # todo here we should create a utility abstract class for JENGA polluters

    @abstractmethod
    def pollute(self):
        pass
