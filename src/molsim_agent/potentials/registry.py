"""Optional ASE calculator discovery; no model downloads are performed."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any


@dataclass(frozen=True, slots=True)
class PotentialStatus:
    name: str
    available: bool
    reason: str | None = None


class PotentialRegistry:
    def available_models(self) -> list[PotentialStatus]:
        statuses = [PotentialStatus("emt", True, "ASE built-in test calculator")]
        statuses.append(PotentialStatus(
            "mace", find_spec("mace") is not None,
            None if find_spec("mace") is not None else "missing dependency: mace-torch",
        ))
        statuses.append(PotentialStatus(
            "uma", find_spec("fairchem") is not None,
            None if find_spec("fairchem") is not None else "missing dependency: fairchem-core",
        ))
        return statuses

    def create_calculator(self, name: str, config: dict[str, Any] | None = None):
        if name.lower() == "emt":
            from ase.calculators.emt import EMT
            return EMT()
        raise RuntimeError(
            f"Potential {name!r} is not available in this installation; "
            "install its optional dependency and configure a checkpoint explicitly."
        )
