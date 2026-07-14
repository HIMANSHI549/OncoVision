from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all CNN architectures.
    """

    @abstractmethod
    def build(self):
        pass