#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""The ``timings.tsv`` contract, shared by the two tools that read it.

``run_queue_sgpu.sh`` appends one row per work item as it dispatches. Two tools
consume that file afterwards -- ``build_weights.py`` to learn the next run's
schedule and ``schedule_report.py`` to report on this one

Columns, in order:

  label       suite label, e.g. "torch"
  tag         work item within the suite, or "whole" for an opaque suite
  gpu         HIP device id it ran on, as written (ids need not start at zero)
  secs        wall-clock duration
  rc          exit code of the suite invocation
  start_off   seconds from the start of the queue to this item's start
  end_off     seconds from the start of the queue to this item's finish
  est         the weight used to schedule it, or the caller's default weight
  incomplete  1 if the item was cut off mid-way

And one column derived on read:

  measured    whether ``secs`` is what the item costs, or merely where it was
              stopped. See ``read_timings``.
"""

import pandas as pd

COLUMNS = ["label", "tag", "gpu", "secs", "rc", "start_off", "end_off", "est", "incomplete"]
TEXT_COLUMNS = ["label", "tag", "gpu"]
NUMERIC_COLUMNS = ["secs", "rc", "start_off", "end_off", "est", "incomplete"]

# rc values that mean the item was killed rather than measured. 124 is a GNU
# timeout expiry (the job's own step timeout, for instance), 137 a SIGKILL or an
# OOM-kill.
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
    # A short row leaves NaN in the numeric columns; a torn one leaves garbage
    # that to_numeric just turned into NaN. Either way there is no measurement.
    frame = frame.dropna(subset=NUMERIC_COLUMNS)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].astype(int)

    frame["name"] = frame["label"] + "/" + frame["tag"]
    # Whether the row is a measurement of the item or just a record of where it
    # stopped. Derived here, once, so the tool that learns from these rows and
    # the tool that reports on them cannot come to different answers.
    #
    # Note what is *not* in the rule: rc. A suite that fails every test in it
    # still took exactly as long as it took, so a failure is as good a
    # measurement as a pass. Only being stopped spoils the number, and there are
    # two ways for that to happen:
    #
    #   rc in KILLED_RCS   stopped from outside -- 124 from a ``timeout`` expiry,
    #                      137 from a SIGKILL or an OOM-kill. The duration is a
    #                      ceiling someone else imposed.
    #   incomplete == 1    pytest never reached its end-of-session write, which
    #                      rc cannot say on its own: a --timeout-method=thread
    #                      expiry exits 1, indistinguishable from an ordinary
    #                      test failure. The scheduler raises the flag from the
    #                      result sidecar the dying process left behind.
    frame["measured"] = ~frame["rc"].isin(KILLED_RCS) & (frame["incomplete"] != 1)
    return frame.reset_index(drop=True)
