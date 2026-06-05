#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""benchstats parser for Transformer Engine microbenchmark samples CSVs.

Reads the per-sample CSV produced by ``run_benchmarks(...)`` with the
``--csv-samples`` flag and turns it into the ``{benchmark_name: {metric: ndarray}}``
structure consumed by ``benchstats.compare.compareStats``.

Columns of the samples CSV: the benchmark parameter columns, plus ``label``,
``sample_idx``, ``time_ms``, ``throughput`` and ``unit``. A benchmark name is
built by joining every parameter column and ``label``, so each unique
(parameters, label) combination becomes one benchmark.

Two metrics are exposed:

- ``time_s`` (seconds, lower is better) -- always present; intended as the
  *main* metric. Exposed in seconds because benchstats' renderer auto-scales
  time values (to ms/us/ns) assuming a seconds base unit.
- the throughput metric, keyed by its unit (e.g. ``TFLOPS`` or ``GB/s``; higher
  is better) -- present when the CSV carries throughput values.

``benchstats``' renderer requires every benchmark to expose the same metric set.
Records without throughput (the samples-only ``Forward+Backward`` composites) are
therefore dropped from the comparison when throughput is available for the other
benchmarks; their raw samples remain in the CSV for other downstream analysis.

The class name matches the file name (``parser_TEsamples``) so it can also be
loaded by the ``benchstats`` CLI via ``--files_parser`` / ``--file1_parser``.
"""

import re

import numpy as np
import pandas as pd

from benchstats.common import ParserBase, LoggingConsole

_TIME_COL = "time_ms"          # column name in the samples CSV (milliseconds)
_TIME_KEY = "time_s"           # metric key exposed to benchstats (seconds)
_THR_COL = "throughput"
_UNIT_COL = "unit"
_GENERIC_THR = "throughput"
_NON_NAME_COLS = ("sample_idx", _TIME_COL, _THR_COL, _UNIT_COL)
_NAME_DELIM = " | "


class parser_TEsamples(ParserBase):
    def __init__(self, csv_file_path, filter, metrics=None, debug_log=True) -> None:
        assert isinstance(csv_file_path, str)
        assert filter is None or isinstance(filter, (str, re.Pattern))
        assert metrics is None or (
            isinstance(metrics, (list, tuple)) and all(isinstance(m, str) for m in metrics)
        )

        if debug_log is None or (isinstance(debug_log, bool) and not debug_log):
            self.debug_log = False
        elif isinstance(debug_log, bool) and debug_log:
            self.debug_log = True
            self.logger = LoggingConsole(log_level=LoggingConsole.LogLevel.Debug)
        else:
            self.debug_log = True
            self.logger = debug_log

        self.file = csv_file_path
        self.filter = (
            filter if filter is None or isinstance(filter, re.Pattern) else re.compile(filter)
        )
        self._requested_metrics = list(metrics) if metrics is not None else None
        self._stats = self._build()

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return self._stats

    def _log(self, level, msg):
        if self.debug_log:
            getattr(self.logger, level)(f"parser_TEsamples: {msg}")

    def _build(self) -> dict[str, dict[str, np.ndarray]]:
        df = pd.read_csv(self.file)

        if _TIME_COL not in df.columns or "sample_idx" not in df.columns:
            raise ValueError(
                f"'{self.file}' is missing 'time_ms'/'sample_idx' columns. "
                "Was it written with --csv-samples?"
            )

        name_cols = [c for c in df.columns if c not in _NON_NAME_COLS]
        if not name_cols:
            raise ValueError(f"No benchmark-name columns found in '{self.file}'.")

        df[_TIME_COL] = pd.to_numeric(df[_TIME_COL], errors="coerce")
        has_thr_col = _THR_COL in df.columns
        if has_thr_col:
            df[_THR_COL] = pd.to_numeric(df[_THR_COL], errors="coerce")

        # First pass: collect per-benchmark time samples, throughput samples and unit.
        per_bm = {}  # name -> {"time": ndarray, "thr": ndarray|None, "unit": str|None}
        for key_vals, group in df.groupby(name_cols, sort=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            bm_name = _NAME_DELIM.join(str(v) for v in key_vals)

            if self.filter is not None and self.filter.search(bm_name) is None:
                continue

            time_ms = group[_TIME_COL].to_numpy(dtype=np.float64)
            time_ms = time_ms[np.isfinite(time_ms)]
            if time_ms.size == 0:
                self._log("warning", f"benchmark '{bm_name}' has no finite time samples; skipping.")
                continue
            # benchstats' renderer auto-scales assuming seconds, so expose seconds.
            time_s = time_ms / 1e3

            thr_s, unit = None, None
            if has_thr_col:
                thr_s = group[_THR_COL].to_numpy(dtype=np.float64)
                thr_s = thr_s[np.isfinite(thr_s) & (thr_s > 0)]
                if thr_s.size == 0:
                    thr_s = None
                else:
                    units = [u for u in group[_UNIT_COL].astype(str).unique()
                             if u and u.lower() != "nan"] if _UNIT_COL in df.columns else []
                    unit = units[0] if len(units) == 1 else (_GENERIC_THR if units else None)

            if self.debug_log and time_s.size < 10:
                self._log(
                    "warning",
                    f"benchmark '{bm_name}' has only {time_s.size} samples (>= 10 recommended); "
                    "re-run with a larger --min-samples.",
                )
            per_bm[bm_name] = {"time": time_s, "thr": thr_s, "unit": unit}

        if not per_bm:
            self._log("warning", f"no benchmarks read from '{self.file}'.")
            return {}

        # Decide on a uniform metric set across all benchmarks.
        with_thr = {n: d for n, d in per_bm.items() if d["thr"] is not None}
        thr_key = self._resolve_throughput_key(with_thr)

        if thr_key is not None and 0 < len(with_thr) < len(per_bm):
            dropped = sorted(set(per_bm) - set(with_thr))
            self._log(
                "warning",
                f"excluding {len(dropped)} benchmark(s) without throughput from the comparison so "
                f"throughput can be shown uniformly: {', '.join(dropped)}",
            )
            per_bm = with_thr

        # Build the result, honoring an explicit metric request if given.
        stats = {}
        for bm_name, d in per_bm.items():
            entry = {}
            if self._metric_requested(_TIME_KEY, thr_key):
                entry[_TIME_KEY] = d["time"]
            if thr_key is not None and d["thr"] is not None and self._metric_requested(thr_key, thr_key):
                entry[thr_key] = d["thr"]
            if entry:
                stats[bm_name] = entry
        return stats

    def _resolve_throughput_key(self, with_thr):
        """Return a single throughput metric key shared by all throughput-bearing benchmarks."""
        if not with_thr:
            return None
        units = {d["unit"] for d in with_thr.values()}
        if len(units) == 1:
            return next(iter(units)) or _GENERIC_THR
        return _GENERIC_THR  # mixed units in one file (atypical) -> generic header

    def _metric_requested(self, key, thr_key):
        """Honor an explicit metrics= request (benchstats CLI), else expose everything."""
        if self._requested_metrics is None:
            return True
        req = self._requested_metrics
        if key == _TIME_KEY:
            return any(t in req for t in (_TIME_KEY, _TIME_COL, "time"))
        # throughput: match the unit key, the generic name, or literal 'throughput'
        return key in req or _GENERIC_THR in req or _THR_COL in req
