#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Read ``timings.tsv``, shared by build_weights.py and schedule_report.py.

run_queue.sh appends one row per work item as it finishes. Columns:

  label       suite label, e.g. "torch"
  tag         work item within the suite, or "whole" for an opaque suite
  gpus        comma-separated HIP device ids the item ran on
  secs        wall-clock duration
  rc          exit code of the suite invocation
  start_off   seconds from the queue start to the item's start
  end_off     seconds from the queue start to the item's end
  est         weight used to schedule it (the default weight if unknown)
  incomplete  1 if pytest died mid-run

Added on read:

  width       number of GPUs the item held
  gpu_secs    ``secs * width``
  measured    whether ``secs`` is a full measurement (see ``read_timings``)
"""

import pandas as pd

COLUMNS = ["label", "tag", "gpus", "secs", "rc", "start_off", "end_off", "est", "incomplete"]
TEXT_COLUMNS = ["label", "tag", "gpus"]
NUMERIC_COLUMNS = ["secs", "rc", "start_off", "end_off", "est", "incomplete"]

# rc of an item killed from outside: 124 is a `timeout` expiry, 137 a SIGKILL or
# an OOM-kill
KILLED_RCS = frozenset((124, 137))


def read_timings(path):
    """``timings.tsv`` as a DataFrame, one row per dispatched work item."""
    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=COLUMNS,
            dtype={c: str for c in TEXT_COLUMNS},
            keep_default_na=False,
            na_values=[],
            on_bad_lines="skip",
        )
    except (OSError, pd.errors.EmptyDataError):
        frame = pd.DataFrame({c: pd.Series(dtype=str) for c in COLUMNS})

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Drop short or garbled rows
    frame = frame.dropna(subset=NUMERIC_COLUMNS)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].astype(int)

    frame["name"] = frame["label"] + "/" + frame["tag"]
    frame["width"] = frame["gpus"].str.count(",") + 1
    frame["gpu_secs"] = frame["secs"] * frame["width"]
    # A row is a measurement unless the item was stopped: killed from outside or
    # cut off mid-run. A failing item still counts; it took as long as it took.
    frame["measured"] = ~frame["rc"].isin(KILLED_RCS) & (frame["incomplete"] != 1)
    return frame.reset_index(drop=True)
