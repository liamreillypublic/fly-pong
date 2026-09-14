"""Build static/atlas.bin: 2D soma positions and class codes in graph order.

Layout: uint32 LE header length; UTF-8 JSON header padded with spaces to a
multiple of 4 bytes; x as uint16[n]; y as uint16[n]; class code as uint8[n].
Missing positions are 65535 on both axes.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

from .annotations import Annotations

MISSING = 65535
CLASS_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("optic lobe", ("ol_intrinsic", "ol_sensory", "visual_centrifugal")),
    ("central brain", ("cb_intrinsic", "cb_endocrine", "cb_efferent", "cb_motor")),
    ("visual projection", ("visual_projection", "visual_projection_tbc")),
    ("descending", ("descending_neuron", "descending_neuron_tbc", "sensory_descending", "efferent_descending")),
    ("ascending", ("ascending_neuron", "sensory_ascending", "sensory_ascending_tbc", "efferent_ascending")),
    ("sensory", ("cb_sensory", "cb_sensory_tbc", "vnc_sensory", "vnc_sensory_tbc")),
    ("motor", ("vnc_motor", "vnc_efferent")),
    ("nerve cord", ("vnc_intrinsic", "vnc_tbc", "vnc_endocrine")),
    ("other", ()),
]
CLASS_NAMES = [name for name, _ in CLASS_GROUPS]
_LOOKUP = {sc: code for code, (_, members) in enumerate(CLASS_GROUPS) for sc in members}
OTHER = len(CLASS_GROUPS) - 1


def class_code(superclass: str) -> int:
    return _LOOKUP.get(superclass, OTHER)


def project(xyz: np.ndarray) -> np.ndarray:
    """PCA-project (n, 3) coordinates onto two axes. Both axes share one scale
    (the larger span maps to 0..65534). NaN rows become MISSING."""
    xyz = np.asarray(xyz, dtype=np.float64)
    out = np.full((len(xyz), 2), MISSING, dtype=np.uint16)
    valid = np.isfinite(xyz).all(axis=1)
    if valid.sum() < 2:
        return out
    pts = xyz[valid]
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    span = float(max((hi - lo).max(), 1e-9))
    scaled = (proj - lo) / span * 65534.0
    out[valid] = np.round(scaled).astype(np.uint16)
    return out


def write_atlas(path: Path, xy: np.ndarray, codes: np.ndarray, class_names: list[str]) -> dict:
    n = len(xy)
    header = {"neurons": n, "classes": list(class_names), "missing": MISSING,
              "layout": "u32 header_len; json; x:u16[n]; y:u16[n]; class:u8[n]"}
    hbytes = json.dumps(header).encode("utf-8")
    hbytes += b" " * (-len(hbytes) % 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(hbytes)))
        f.write(hbytes)
        f.write(np.ascontiguousarray(xy[:, 0], dtype="<u2").tobytes())
        f.write(np.ascontiguousarray(xy[:, 1], dtype="<u2").tobytes())
        f.write(np.ascontiguousarray(codes, dtype="u1").tobytes())
    return header


def read_atlas(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    data = Path(path).read_bytes()
    (hlen,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + hlen].decode("utf-8"))
    n = int(header["neurons"])
    offset = 4 + hlen
    x = np.frombuffer(data, dtype="<u2", count=n, offset=offset)
    offset += 2 * n
    y = np.frombuffer(data, dtype="<u2", count=n, offset=offset)
    offset += 2 * n
    codes = np.frombuffer(data, dtype="u1", count=n, offset=offset)
    return header, np.stack([x, y], axis=1).astype(np.uint16), codes.copy()


def build(ann: Annotations, path: Path) -> dict:
    xy = project(ann.soma_xyz)
    codes = np.array([class_code(s) for s in ann.superclass], dtype=np.uint8)
    return write_atlas(path, xy, codes, CLASS_NAMES)


def main(argv=None) -> int:
    from . import config
    from .annotations import load

    graph, annotations = config.graph_path(), config.annotations_path()
    for p in (graph, annotations):
        if not p.exists():
            print(f"Missing {p}. Build the upstream data first (see README).", file=sys.stderr)
            return 1
    with np.load(graph, allow_pickle=False) as g:
        body_ids = g["body_ids"]
    header = build(load(annotations, body_ids), config.ATLAS_PATH)
    print(json.dumps(header))
    print(f"Wrote {config.ATLAS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
