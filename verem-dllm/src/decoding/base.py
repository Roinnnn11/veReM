from abc import ABC, abstractmethod


class BaseDecoder(ABC):
    @abstractmethod
    def decode(self, example: dict) -> dict:
        """
        Args:
            example: {"id": str, "prompt": str, "gold": str}
        Returns:
            {
                "id": str,
                "prompt": str,
                "gold": str,
                "initial_output": str,
                "final_output": str,
                "initial_correct": bool,
                "final_correct": bool,
                "method": str,
                "latency": float,
                "num_model_forwards": int,
                "num_verifier_calls": int,
                "num_remasked_spans": int,
                "revision_trace": list[dict],
            }
        """
        raise NotImplementedError
