#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""benchstats parser for ASV-format result JSON files written by ``driver.py``.

Reads one ASV result file (``<results_dir>/<machine>/<hash>-<env>.json``) and
turns it into the ``{benchmark_name: {metric: ndarray}}`` structure consumed by
``benchstats.compare.compareStats``.

An ASV result file stores, per benchmark, a row whose columns are named by the
file's ``result_columns`` list. The driver records raw per-call timing samples
in the ``samples`` column (a list of sample-lists, one per parameter
combination, in ``itertools.product`` order over ``params``). This parser flattens
that into one benchstats "benchmark" per (benchmark, parameter-combination):

- the benchmark name is ``<suite>.<Class>.<time_method> | name=val, ...`` where
  the parameter names come from ``benchmarks.json`` (falling back to positional
  ``p0, p1, ...`` when the index is unavailable).
- a single metric ``time_s`` (seconds, lower is better) holds the raw samples.
  Samples are already stored in seconds; benchstats' renderer auto-scales them
  (to ms/us/ns) assuming a seconds base unit.

Throughput is intentionally not exposed as a separate metric: the ASV result
file carries no per-sample work, and because work is constant per parameter
combination a rank-based test on throughput is identical to the test on time.
The driver already prints throughput columns during a run.

The class name matches the file name (``parser_TEasv``) so it can also be loaded
by the ``benchstats`` CLI via ``--files_parser`` / ``--file1_parser``.
"""

import itertools
import json
import os
import re

import numpy as np

from benchstats.common import ParserBase, LoggingConsole

_TIME_KEY = "time_s"           # metric key exposed to benchstats (seconds)
_NAME_DELIM = " | "


class parser_TEasv(ParserBase):
    def __init__(self, json_file_path, filter, metrics=None, debug_log=True) -> None:
        assert isinstance(json_file_path, str)
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

        self.file = json_file_path
        self.filter = (
            filter if filter is None or isinstance(filter, re.Pattern) else re.compile(filter)
        )
        self._requested_metrics = list(metrics) if metrics is not None else None
        self._stats = self._build()

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return self._stats

    def _log(self, level, msg):
        if self.debug_log:
            getattr(self.logger, level)(f"parser_TEasv: {msg}")

    def _load_param_names(self) -> dict:
        """Map ``bench_key -> [param_names]`` from the sibling ``benchmarks.json``.

        Layout: ``<results_dir>/<machine>/<file>.json`` and
        ``<results_dir>/benchmarks.json``. The names are only used for readable
        labels, so a missing/unreadable index degrades gracefully to ``{}``.
        """
        results_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.file)))
        index_path = os.path.join(results_dir, "benchmarks.json")
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (OSError, ValueError):
            self._log("warning", f"could not read '{index_path}'; using positional param names.")
            return {}
        return {
            key: meta["param_names"]
            for key, meta in index.items()
            if isinstance(meta, dict) and "param_names" in meta
        }

    def _build(self) -> dict[str, dict[str, np.ndarray]]:
        with open(self.file) as f:
            data = json.load(f)

        columns = data.get("result_columns")
        results = data.get("results", {})
        if not isinstance(columns, list) or "samples" not in columns:
            raise ValueError(
                f"'{self.file}' has no 'samples' column. Re-run the benchmarks with a "
                "driver.py that records raw samples."
            )
        i_params = columns.index("params")
        i_samples = columns.index("samples")

        names_map = self._load_param_names()
        want_time = self._metric_requested(_TIME_KEY)

        stats = {}
        for bench_key, row in results.items():
            if not row or len(row) <= i_samples:
                continue
            params = row[i_params] or []
            sample_lists = row[i_samples]
            if sample_lists is None:
                self._log("warning", f"benchmark '{bench_key}' has no samples; skipping.")
                continue

            combos = list(itertools.product(*params)) if params else [()]
            param_names = names_map.get(bench_key)

            for combo, samples in itertools.zip_longest(combos, sample_lists):
                if samples is None:
                    continue
                time_s = np.asarray(samples, dtype=np.float64)
                time_s = time_s[np.isfinite(time_s)]
                if time_s.size == 0:
                    continue

                label = self._format_combo(param_names, combo)
                bm_name = bench_key + (_NAME_DELIM + label if label else "")
                if self.filter is not None and self.filter.search(bm_name) is None:
                    continue

                if self.debug_log and time_s.size < 10:
                    self._log(
                        "warning",
                        f"benchmark '{bm_name}' has only {time_s.size} samples "
                        "(>= 10 recommended); re-run with a larger -n/--iters.",
                    )
                if want_time:
                    stats[bm_name] = {_TIME_KEY: time_s}

        if not stats:
            self._log("warning", f"no benchmarks read from '{self.file}'.")
        return stats

    @staticmethod
    def _format_combo(param_names, combo):
        """Build a readable ``name=val, ...`` label for one parameter combination."""
        if combo is None:
            return ""
        values = [str(v) for v in combo]
        if param_names and len(param_names) == len(values):
            return ", ".join(f"{n}={v}" for n, v in zip(param_names, values))
        return ", ".join(values)

    def _metric_requested(self, key):
        """Honor an explicit metrics= request (benchstats CLI), else expose everything."""
        if self._requested_metrics is None:
            return True
        if key == _TIME_KEY:
            return any(t in self._requested_metrics for t in (_TIME_KEY, "time_ms", "time"))
        return key in self._requested_metrics
