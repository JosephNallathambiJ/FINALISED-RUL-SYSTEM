from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSensor(ABC):
    """Abstract base class for all agent sensors."""

    @abstractmethod
    def read(self) -> Dict[str, Any]:
        """Perceive current environment/system values."""
        pass
