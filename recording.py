import csv
import json
import os
from datetime import datetime, timezone
import numpy as np

_META, _LOG, _ARR = "__meta.json", "__log.csv", "__arrays.npz"

def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}.", v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix[:-1]] = json.dumps(list(obj))
    else:
        out[prefix[:-1]] = obj


def save_run(outdir, name, config, log, *, arrays=None, config_in_rows=True):
    os.makedirs(outdir, exist_ok=True)
    cfg_cols = {}
    if config_in_rows:
        _flatten("cfg.", dict(config), cfg_cols)
    seen, lead = list(cfg_cols.keys()), ["iter", "beta", "lam"]
    keys = set()
    for rec in log:
        keys.update(rec.keys())
    rest = sorted(k for k in keys if k not in lead)
    fieldnames = seen + [k for k in lead if k in keys] + rest

    log_path = os.path.join(outdir, name + _LOG)
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        for rec in log:
            row = dict(cfg_cols)
            row.update({k: _scalarize(v) for k, v in rec.items()})
            w.writerow(row)

    meta = {"name": name,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_records": len(log),
            "schema": fieldnames,
            "config": _jsonable(config)}
    meta_path = os.path.join(outdir, name + _META)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    paths = {"meta": meta_path, "log": log_path}
    if arrays:
        arr_path = os.path.join(outdir, name + _ARR)
        np.savez_compressed(arr_path, **{k: np.asarray(v) for k, v in arrays.items()})
        paths["arrays"] = arr_path
    return paths


def load_run(outdir, name):
    with open(os.path.join(outdir, name + _META)) as f:
        meta = json.load(f)
    log = []
    with open(os.path.join(outdir, name + _LOG), newline="") as f:
        for row in csv.DictReader(f):
            log.append({k: _coerce(v) for k, v in row.items()})
    arrays = None
    arr_path = os.path.join(outdir, name + _ARR)
    if os.path.exists(arr_path):
        with np.load(arr_path) as z:
            arrays = {k: z[k] for k in z.files}
    return meta, log, arrays


def _scalarize(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(v).tolist())
    return v


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _coerce(v):
    if v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except (ValueError, TypeError):
        return v