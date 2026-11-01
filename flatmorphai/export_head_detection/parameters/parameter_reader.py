# -*- coding: utf-8 -*-
"""Reader for the parameter files of the export (folder parameters/, separator ";").

Every file has four columns: parameter ; default_value ; corrected_value ; description.
Single rule: if corrected_value is not empty it is used, otherwise default_value.
Values are converted to int, float, bool, tuple (e.g. "0.00,0.30") or string according to their form.
A user interface (PySide or other) can therefore edit the corrected_value column without touching the code.
"""
import os, csv
EXPORT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(EXPORT_ROOT, "parameters")
COLUMNS = ["parameter", "default_value", "corrected_value", "description"]

def _convert(s):
    s = str(s).strip()
    if s == "": return None
    if s.lower() in ("true", "yes"): return True
    if s.lower() in ("false", "no"): return False
    if "," in s:
        try: return tuple(_convert(x) for x in s.split(","))
        except Exception: return s
    try:
        return int(s) if s.lstrip("-").isdigit() else float(s)
    except ValueError:
        return s

def read(name):
    """Return {parameter: effective value} for parameters/<name>.csv (corrected_value first when not empty)."""
    path = os.path.join(FOLDER, name if name.endswith(".csv") else name + ".csv")
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            corr = row.get("corrected_value", "")
            raw = corr if corr is not None and corr.strip() != "" else row["default_value"]
            out[row["parameter"].strip()] = _convert(raw)
    return out

def read_table(name):
    """Return the raw rows (dict) of the file, for display or editing."""
    path = os.path.join(FOLDER, name if name.endswith(".csv") else name + ".csv")
    with open(path, encoding="utf-8") as f: return list(csv.DictReader(f, delimiter=";"))

def write_correction(name, parameter, value):
    """Write a corrected value (empty string or None to return to the default) and rewrite the file."""
    path = os.path.join(FOLDER, name if name.endswith(".csv") else name + ".csv")
    rows = read_table(name); found = False
    for r in rows:
        if r["parameter"].strip() == parameter: r["corrected_value"] = "" if value is None else str(value); found = True
    if not found: raise KeyError("unknown parameter: %s in %s" % (parameter, name))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";", lineterminator="\n"); w.writeheader(); w.writerows(rows)  # LF, as the shipped files: a run must not show up as a modification

def model_path(role):
    """Absolute path of the checkpoint for the given role (file models.csv), corrected value first."""
    m = read("models")
    if role not in m: raise KeyError("unknown model role: %s" % role)
    p = str(m[role])
    return p if os.path.isabs(p) else os.path.join(EXPORT_ROOT, p)
