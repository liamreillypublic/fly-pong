"""Per-neuron annotations from the MaleCNS feather file, aligned to graph order."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

COLUMNS = ["bodyId", "somaSide", "superclass", "somaLocation", "assignedOlHex1", "assignedOlHex2"]


@dataclass
class Annotations:
    soma_side: np.ndarray   # str: L, R, M or ?
    superclass: np.ndarray  # str, "" when missing
    hex1: np.ndarray        # int32, -1 when missing
    hex2: np.ndarray        # int32, -1 when missing
    soma_xyz: np.ndarray    # float32 (n, 3), NaN rows when missing

    @property
    def n(self) -> int:
        return len(self.soma_side)


def from_rows(rows: list[dict], body_ids: np.ndarray) -> Annotations:
    """Align annotation rows (dicts keyed by COLUMNS) to the order of body_ids.
    Neurons absent from rows get the missing-value defaults."""
    by_id = {int(r["bodyId"]): r for r in rows}
    n = len(body_ids)
    side = np.full(n, "?", dtype="<U1")
    superclass = np.full(n, "", dtype="<U40")
    hex1 = np.full(n, -1, dtype=np.int32)
    hex2 = np.full(n, -1, dtype=np.int32)
    xyz = np.full((n, 3), np.nan, dtype=np.float32)
    for i, body in enumerate(body_ids):
        r = by_id.get(int(body))
        if r is None:
            continue
        if r.get("somaSide"):
            side[i] = str(r["somaSide"])[0]
        if r.get("superclass"):
            superclass[i] = r["superclass"]
        if r.get("assignedOlHex1") is not None:
            hex1[i] = int(r["assignedOlHex1"])
        if r.get("assignedOlHex2") is not None:
            hex2[i] = int(r["assignedOlHex2"])
        loc = r.get("somaLocation")
        if loc is not None and len(loc) == 3:
            xyz[i] = loc
    return Annotations(side, superclass, hex1, hex2, xyz)


def load(path: Path, body_ids: np.ndarray) -> Annotations:
    table = feather.read_table(str(path), columns=COLUMNS)
    return from_rows(table.to_pylist(), body_ids)
