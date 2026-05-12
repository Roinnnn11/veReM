from dataclasses import dataclass


@dataclass
class Span:
    span_id: int
    start: int       # char offset in generation text
    end: int
    text: str
    span_type: str   # "sentence", "equation", "number", "answer"

    def __repr__(self):
        return f"Span({self.span_id}, [{self.start}:{self.end}], type={self.span_type!r}, text={self.text[:40]!r})"
