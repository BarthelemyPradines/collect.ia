from enum import Enum
from typing import Any

from pydantic import BaseModel, create_model


def parse_structured_output_spec(spec: str) -> type[BaseModel]:
    """Parse a structured output spec string and return a Pydantic model.

    Supported formats:
        - "bool"              -> model with a single `answer: bool` field
        - "enum=A,B,C"        -> model with a single `answer: Enum("A","B","C")` field
    """
    spec = spec.strip()

    if spec.lower() == "bool":
        return create_model("BoolAnswer", answer=(bool, ...))

    if spec.startswith("enum="):
        raw_values = spec[5:]  # everything after "enum="
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No enum values found in spec: {spec!r}")

        enum_cls = Enum("AnswerEnum", {v: v for v in values})
        return create_model("EnumAnswer", answer=(enum_cls, ...))

    raise ValueError(f"Unknown structured output spec: {spec!r}")
