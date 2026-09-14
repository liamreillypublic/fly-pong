"""Project-owned connectome graph: full transmitter table, CSR order by source.

Same neuron selection and ordering as the upstream simulator (annotated
superclass, glia excluded, sorted by body ID) so body_ids match and the
atlas and dopamine files stay valid. Unlike upstream, histamine is treated
as the fast inhibitory transmitter it is, and weights are absolute:
0.275 mV of postsynaptic potential per synapse (Shiu et al. 2024).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

SIGNS = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
PSP_MV = 0.275
FORMAT_VERSION = 2


def _sha256(path: Path) -> str:
    from malecns.data import sha256
    return sha256(path)


def select_neurons(annotations_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Body IDs (sorted, int64) and cell types with upstream's selection rule."""
    table = feather.read_table(str(annotations_path), columns=["bodyId", "superclass", "type"])
    rows = [r for r in table.to_pylist() if r["superclass"] and "glia" not in r["superclass"].lower()]
    rows.sort(key=lambda r: r["bodyId"])
    ids = np.array([r["bodyId"] for r in rows], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate neuron body IDs")
    types = np.array([r["type"] or "" for r in rows])
    return ids, types


def neuron_signs(transmitters_path: Path, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron output sign and transmitter label, aligned to body_ids."""
    table = feather.read_table(str(transmitters_path), columns=["body", "consensus_nt"])
    nt = {int(r["body"]): (r["consensus_nt"] or "unknown").lower() for r in table.to_pylist()}
    labels = np.array([nt.get(int(b), "unknown") for b in body_ids])
    signs = np.array([SIGNS.get(label, 0.0) for label in labels], dtype=np.float32)
    return signs, labels


def scan_edges(raw_edges_path: Path, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Edges with both ends among body_ids, as (src index, dst index, count, raw rows scanned)."""
    body_ids = np.asarray(body_ids, dtype=np.int64)
    srcs, dsts, cnts, raw = [], [], [], 0
    with pa.memory_map(str(raw_edges_path), "r") as stream:
        reader = ipc.open_file(stream)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            pre = batch.column("body_pre").to_numpy()
            post = batch.column("body_post").to_numpy()
            cnt = batch.column("weight").to_numpy()
            raw += len(pre)
            si = np.searchsorted(body_ids, pre)
            di = np.searchsorted(body_ids, post)
            ok = (si < len(body_ids)) & (di < len(body_ids))
            ok[ok] &= (body_ids[si[ok]] == pre[ok]) & (body_ids[di[ok]] == post[ok])
            if ok.any():
                if np.any(cnt[ok] <= 0):
                    raise ValueError("Nonpositive synapse count")
                srcs.append(si[ok]); dsts.append(di[ok]); cnts.append(cnt[ok])
    if not srcs:
        raise ValueError("Selected neurons have no connections")
    return (np.concatenate(srcs).astype(np.int64), np.concatenate(dsts).astype(np.int64),
            np.concatenate(cnts).astype(np.int64), raw)


def to_csr(n: int, src: np.ndarray, dst: np.ndarray, count: np.ndarray, signs: np.ndarray) -> dict:
    order = np.lexsort((dst, src))
    src, dst, count = src[order], dst[order], count[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    weight = (PSP_MV * count * signs[src]).astype(np.float32)
    return {"indptr": indptr, "source": src.astype(np.int32), "target": dst.astype(np.int32),
            "count": count.astype(np.int32), "weight": weight}


def build(annotations_path: Path, transmitters_path: Path, raw_edges_path: Path, out_path: Path) -> dict:
    body_ids, types = select_neurons(annotations_path)
    signs, labels = neuron_signs(transmitters_path, body_ids)
    src, dst, count, raw = scan_edges(raw_edges_path, body_ids)
    csr = to_csr(len(body_ids), src, dst, count, signs)
    metadata = {
        "dataset": "male-cns:v1.0", "format_version": FORMAT_VERSION, "layout": "csr-by-source",
        "neurons": int(len(body_ids)), "edges": int(len(src)), "synaptic_contacts": int(count.sum()),
        "raw_segment_edges": int(raw), "psp_mv": PSP_MV, "signs": SIGNS,
        "zero_sign_neurons": int((signs == 0).sum()),
        "zero_weight_edges": int((csr["weight"] == 0).sum()),
        "histamine_neurons": int((labels == "histamine").sum()),
        "source_sha256": {p.name: _sha256(p) for p in (annotations_path, transmitters_path, raw_edges_path)},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, body_ids=body_ids, cell_types=types, neurotransmitters=labels, signs=signs,
                 metadata=json.dumps(metadata), **csr)
    tmp.replace(out_path)
    return metadata


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as g:
        data = {k: g[k] for k in g.files}
    data["metadata"] = json.loads(str(data["metadata"]))
    if data["metadata"].get("layout") != "csr-by-source":
        raise ValueError(f"{path} is not a project CSR graph; run: uv run python -m flypong.graph")
    return data


def main(argv=None) -> int:
    from . import config
    paths = (config.annotations_path(), config.transmitters_path(), config.raw_edges_path())
    for p in paths:
        if not p.exists():
            print(f"Missing {p}; run the upstream download (see README).", file=sys.stderr)
            return 1
    print("Building the project graph (about a minute)...", flush=True)
    meta = build(*paths, config.PROJECT_GRAPH_PATH)
    print(json.dumps({k: v for k, v in meta.items() if k != "source_sha256"}, indent=2))
    print(f"Wrote {config.PROJECT_GRAPH_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
