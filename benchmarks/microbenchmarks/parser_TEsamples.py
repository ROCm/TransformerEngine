#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""benchstats parser for Transformer Engine microbenchmark samples CSVs.

Reads the per-sample CSV produced by ``run_benchmarks(... )`` with the
``--csv-samples`` flag (columns: the benchmark parameter columns plus
``label``, ``sample_idx``, ``time_ms``) and turns it into the
``{benchmark_name: {metric: ndarray}}`` structure consumed by
``benchstats.compare.compareStats``.

A benchmark name is built by joining every column except ``sample_idx`` and the
metric column, so each unique (parameters, label) combination becomes one
benchmark. The single available metric is ``time_ms`` (lower is better).

The class name matches the file name (``parser_TEsamples``) so it can also be
loaded by the ``benchstats`` CLI via ``--files_parser`` / ``--file1_parser``.
"""

import re

import numpy as np
import pandas as pd

from benchstats.common import ParserBase, LoggingConsole

_METRIC = "time_ms"
_NON_NAME_COLS = ("sample_idx", _METRIC)
_NAME_DELIM = " | "


class parser_TEsamples(ParserBase):
    def __init__(self, csv_file_path, filter, metrics, debug_log=True) -> None:
        assert isinstance(csv_file_path, str)
        assert filter is None or isinstance(filter, (str, re.Pattern))
        assert isinstance(metrics, (list, tuple)) and len(metrics) > 0
        assert all(isinstance(m, str) for m in metrics)

        if debug_log is None or (isinstance(debug_log, bool) and not debug_log):
            self.debug_log = False
        elif isinstance(debug_log, bool) and debug_log:
            self.debug_log = True
            self.logger = LoggingConsole(log_level=LoggingConsole.LogLevel.Debug)
        else:
            self.debug_log = True
            self.logger = debug_log

        unsupported = [m for m in metrics if m != _METRIC]
        if unsupported:
            raise ValueError(
                f"parser_TEsamples only supports the '{_METRIC}' metric, got: {unsupported}. "
                "The samples CSV produced by --csv-samples carries per-run times only."
            )

        self.file = csv_file_path
        self.filter = filter if filter is None or isinstance(filter, re.Pattern) else re.compile(filter)
        self._stats = self._build()

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return self._stats

    def _build(self) -> dict[str, dict[str, np.ndarray]]:
        df = pd.read_csv(self.file)

        for col in _NON_NAME_COLS:
            if col not in df.columns:
                raise ValueError(
                    f"'{col}' column not found in '{self.file}'. Was the CSV written with "
                    "--csv-samples?"
                )

        name_cols = [c for c in df.columns if c not in _NON_NAME_COLS]
        if not name_cols:
            raise ValueError(f"No benchmark-name columns found in '{self.file}'.")

        df[_METRIC] = pd.to_numeric(df[_METRIC], errors="coerce")

        stats: dict[str, dict[str, np.ndarray]] = {}
        for key_vals, group in df.groupby(name_cols, sort=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            bm_name = _NAME_DELIM.join(str(v) for v in key_vals)

            if self.filter is not None and self.filter.search(bm_name) is None:
                continue

            samples = group[_METRIC].to_numpy(dtype=np.float64)
            samples = samples[np.isfinite(samples)]
            if samples.size == 0:
                if self.debug_log:
                    self.logger.warning(
                        f"parser_TEsamples: benchmark '{bm_name}' has no finite samples; skipping."
                    )
                continue
            if self.debug_log and samples.size < 10:
                self.logger.warning(
                    f"parser_TEsamples: benchmark '{bm_name}' has only {samples.size} samples "
                    "(>= 10 recommended). Re-run the benchmark with a larger --repetitions."
                )
            stats[bm_name] = {_METRIC: samples}

        if not stats and self.debug_log:
            self.logger.warning(f"parser_TEsamples: no benchmarks read from '{self.file}'.")
        return stats
