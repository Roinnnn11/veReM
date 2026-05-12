from abc import ABC, abstractmethod


class BaseDLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        steps: int = 64,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        Returns:
            {
                "text": str,
                "latency": float,
                "num_steps": int,
                "num_forwards": int,
                "metadata": dict,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def infill(
        self,
        prompt: str,
        text_with_masks: str,
        steps: int = 32,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        raise NotImplementedError
