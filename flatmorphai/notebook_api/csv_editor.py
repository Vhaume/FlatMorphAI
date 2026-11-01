"""
Notebook 5 port: in-place CSV cell modifications used by the Step 5
Stereotype Editor.

The Step 5 editor reads and writes CSVs with ``sep=';'`` and
``encoding='latin1'`` (spec section 6.6). The function below mirrors
the notebook behaviour byte-for-byte, including the use of
:mod:`fileinput` for true in-place rewriting.
"""

from __future__ import annotations

import csv
import fileinput
import io
import os
from typing import Any


def _modify_text_remove_split_elements(input_text: str, positions_to_remove=(-5, -6)) -> str:
    """Strip the ``planaria_N_`` infix from a frame filename."""
    parts = input_text.split("_")
    normalized = [p for idx, p in enumerate(parts) if idx - len(parts) not in positions_to_remove]
    return "_".join(normalized)


def modify_column_of_row_in_csv(
    filename: str,
    row_index: int,
    column_name: str,
    new_value: Any,
    separator: str = ";",
    encoding: str = "latin1",
) -> None:
    """Rewrite the value at ``(row_index, column_name)`` of ``filename`` in place.

    Uses :mod:`fileinput` to preserve the original CSV dialect without
    loading the entire file into memory. Raises :class:`PermissionError`
    when the file is locked - callers must show a critical dialog and
    restore the previous value.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    # Resolve the target column index.
    with open(filename, "r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=separator)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"Empty CSV: {filename}")
    try:
        column_index = header.index(column_name)
    except ValueError as exc:
        raise KeyError(f"Column '{column_name}' not in {filename}") from exc

    target_line = row_index + 2  # header is line 1, first data row is line 2
    current_line = 0

    with fileinput.input(files=(filename,), inplace=True, encoding=encoding) as handle:
        for line in handle:
            current_line += 1
            if current_line == target_line:
                stripped = line.rstrip("\r\n")
                newline_suffix = line[len(stripped):]
                fields = next(csv.reader([stripped], delimiter=separator))
                while len(fields) <= column_index:
                    fields.append("")
                fields[column_index] = "" if new_value is None else str(new_value)
                sink = io.StringIO()
                writer = csv.writer(sink, delimiter=separator, lineterminator="")
                writer.writerow(fields)
                print(sink.getvalue() + newline_suffix, end="")
            else:
                print(line, end="")
