from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fc.dsl.tokens import Value, validate_opcode


@dataclass(frozen=True)
class Instruction:
    opcode: str
    args: dict[str, Value] = field(default_factory=dict)
    dest: str | None = None

    def __post_init__(self) -> None:
        validate_opcode(self.opcode)

    def to_dict(self) -> dict[str, Any]:
        return {"opcode": self.opcode, "args": self.args, "dest": self.dest}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Instruction":
        return Instruction(opcode=data["opcode"], args=data.get("args", {}), dest=data.get("dest"))


@dataclass(frozen=True)
class Program:
    instructions: list[Instruction]

    def to_dict(self) -> dict[str, Any]:
        return {"instructions": [inst.to_dict() for inst in self.instructions]}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Program":
        return Program(instructions=[Instruction.from_dict(d) for d in data["instructions"]])

    def skeleton(self) -> list[str]:
        return [inst.opcode for inst in self.instructions]
