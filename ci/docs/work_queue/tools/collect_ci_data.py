#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Collect the GitHub Actions data behind the work queue design doc.

Fetches the runs of the ROCm CI workflows since --since, the jobs of every
attempt, the log of every sGPU job and the changed files of every pull request
that ran GPU tests, and writes the CSV files make_figures.py plots. Downloads
are cached under --cache, so a second run fetches only what is new.

Usage:
  GH_TOKEN=<token> python3 collect_ci_data.py [--since 2026-08-01] [--cache DIR] [--out DIR]

Without GH_TOKEN the token comes from `gh auth token`.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import requests

REPO = "ROCm/TransformerEngine"
API = f"https://api.github.com/repos/{REPO}"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

# Branch runs (pushes, manual runs) and pull request runs
WORKFLOWS = {"rocm-ci.yml": "branch", "rocm-ci-dispatch.yml": "pr"}

# In the queue's sGPU test step name; the old runner's is "Run sGPU tests in parallel"
QUEUE_STEP = "as one queue"

# Queue progress line: [time] gpu<ids> t+<start>s <secs>s rc=<rc> [est=<w>] <label> <tag>
QUEUE_ITEM = re.compile(
    r"\] gpu(\S+)\s+t\+(\d+)s\s+(\d+)s\s+rc=(\d+)\s+(?:est=(\S+)\s+)?(\S+)\s+(\S+)\s*$"
)
QUEUE_GPUS = re.compile(r"== GPUs: (\d+) visible")
# Old static runner: one line per suite at launch and at exit
STATIC_START = re.compile(r"^\ufeff?(\S+)Z Started '(\w+)'")
STATIC_END = re.compile(r"^\ufeff?(\S+)Z \[[^]]+\] '(\w+)' \(pid \d+\) finished with rc=(\d+)")


def token():
    """GH_TOKEN, else the gh CLI's token."""
    if os.environ.get("GH_TOKEN"):
        return os.environ["GH_TOKEN"]
    return subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=True
    ).stdout.strip()


SESSION = requests.Session()


def get(path, params=None, raw=False):
    """GET an API path, retrying on rate limits and transient errors."""
    url = path if path.startswith("http") else f"{API}/{path}"
    for attempt in range(6):
        resp = SESSION.get(url, params=params, timeout=120)
        if resp.status_code == 200:
            return resp.content if raw else resp.json()
        if resp.status_code in (404, 410):
            return None
        time.sleep(10 * (attempt + 1))
    resp.raise_for_status()
    return None


def cached_json(path, fetch, fresh=lambda data: True):
    """Load ``path`` if it exists and is fresh, else fetch and store it."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if fresh(data):
            return data
    data = fetch()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return data


def list_runs(since):
    """Every run of WORKFLOWS created on or after ``since``."""
    runs = []
    for workflow, kind in WORKFLOWS.items():
        page = 1
        while True:
            data = get(
                f"actions/workflows/{workflow}/runs",
                {"per_page": 100, "page": page, "created": f">={since}"},
            )
            for run in data["workflow_runs"]:
                run["_kind"] = kind
            runs += data["workflow_runs"]
            if len(data["workflow_runs"]) < 100:
                break
            page += 1
    return runs


def run_jobs(cache, run):
    """The jobs of every attempt of ``run``, cached once the run is complete."""

    def fetch():
        attempts = {}
        for attempt in range(1, run["run_attempt"] + 1):
            jobs, page = [], 1
            while True:
                data = get(
                    f"actions/runs/{run['id']}/attempts/{attempt}/jobs",
                    {"per_page": 100, "page": page},
                )
                jobs += data["jobs"] if data else []
                if not data or len(data["jobs"]) < 100:
                    break
                page += 1
            attempts[str(attempt)] = jobs
        return {"run_attempt": run["run_attempt"], "attempts": attempts}

    return cached_json(
        f"{cache}/raw/jobs_{run['id']}.json",
        fetch,
        lambda data: run["status"] == "completed" and data["run_attempt"] == run["run_attempt"],
    )


def seconds(start, end):
    """Seconds between two API timestamps, or None."""
    if not start or not end:
        return None
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (parse(end) - parse(start)).total_seconds()


def job_type(name):
    """Classify a job by its name."""
    for prefix, jtype in (
        ("sGPU", "sgpu"),
        ("mGPU Tests", "mgpu-queue"),
        ("mGPU Torch", "mgpu-torch"),
        ("mGPU JAX", "mgpu-jax"),
        ("build", "build"),
        ("Select Test Suites", "scope"),
    ):
        if name.startswith(prefix):
            return jtype
    return None


def flatten(run, jobs_by_attempt):
    """One row per job execution. A job "Re-run failed jobs" copied into a later
    attempt keeps its timestamps, so (name, started, completed) drops the copies."""
    rows, seen = [], set()
    for jobs in jobs_by_attempt.values():
        for job in jobs:
            match = re.match(r"CI Level (\d) / (.*)", job["name"])
            if match:
                level, name = int(match.group(1)), match.group(2)
            else:
                name = job["name"]
                title = re.match(r"CI Level (\d)", run.get("display_title") or "")
                level = int(title.group(1)) if run["_kind"] == "branch" and title else None
            jtype = job_type(name)
            key = (name, job["started_at"], job["completed_at"])
            if jtype is None or key in seen:
                continue
            seen.add(key)
            arch = re.search(r"\((mi\d\dx)\)", name)
            row = dict(
                run_id=run["id"],
                kind=run["_kind"],
                event=run["event"],
                branch=run["head_branch"],
                run_created=run["created_at"],
                job_id=job["id"],
                attempt=job.get("run_attempt"),
                name=name,
                jtype=jtype,
                arch=arch.group(1) if arch else None,
                level=level,
                conclusion=job["conclusion"],
                started=job["started_at"],
                completed=job["completed_at"],
                dur_s=seconds(job["started_at"], job["completed_at"]),
                pool=re.sub(r"-[a-z0-9]{5}-runner-[a-z0-9]{5}$", "", job.get("runner_name") or ""),
                era=None,
            )
            for step in job.get("steps") or []:
                ran = step.get("conclusion") not in (None, "skipped")
                dur = seconds(step["started_at"], step["completed_at"]) if ran else None
                if step["name"] == "Pull Docker Image":
                    row["pull_s"] = dur
                elif step["name"] == "Install packages":
                    row["install_s"] = dur
                elif step["name"].startswith(("Run sGPU tests", "Run mGPU tests")):
                    row["test_s"] = dur
                    row["test_conclusion"] = step.get("conclusion")
                    if jtype == "sgpu":
                        row["era"] = "queue" if QUEUE_STEP in step["name"] else "static"
            rows.append(row)
    return rows


def job_log(cache, job_id):
    """A job's log text, cached; empty if GitHub no longer has it."""
    path = f"{cache}/logs/job_{job_id}.log"
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        data = get(f"actions/jobs/{job_id}/logs", raw=True)
        if data is None:
            return ""
        with open(path, "wb") as handle:
            handle.write(data)
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def log_time(stamp):
    """A job-log timestamp (up to 7 fractional digits, no zone) as a datetime."""
    whole, _, fraction = stamp.partition(".")
    return datetime.fromisoformat(f"{whole}.{(fraction + '000000')[:6]}")


def static_suites(text):
    """Seconds and exit code of each suite in an old static-runner log."""
    start, out = None, {}
    for line in text.splitlines():
        match = STATIC_START.match(line)
        if match and start is None:
            start = log_time(match.group(1))
        match = STATIC_END.match(line)
        if match and start is not None:
            end = log_time(match.group(1))
            out[f"{match.group(2)}_s"] = (end - start).total_seconds()
            out[f"{match.group(2)}_rc"] = int(match.group(3))
    return out


def queue_items(text):
    """The GPU count and the per-item progress lines of a queue log."""
    gpus, items = None, []
    for line in text.splitlines():
        match = QUEUE_GPUS.search(line)
        if match:
            gpus = int(match.group(1))
        match = QUEUE_ITEM.search(line.rstrip())
        if match:
            gpu, start, secs, rc, est, label, tag = match.groups()
            est = None if est in (None, "unknown") else float(est.rstrip("s"))
            items.append(
                dict(
                    gpu=gpu,
                    start=int(start),
                    secs=int(secs),
                    rc=int(rc),
                    est=est,
                    label=label,
                    tag=tag,
                )
            )
    return gpus, items


def pr_number(run, by_branch):
    """The pull request a PR-workflow run belongs to."""
    if run.get("pull_requests"):
        return run["pull_requests"][0]["number"]
    branch = run["head_branch"]
    if branch not in by_branch:
        owner = run["head_repository"]["owner"]["login"]
        found = get("pulls", {"head": f"{owner}:{branch}", "state": "all"}) or []
        by_branch[branch] = found[0]["number"] if found else None
    return by_branch[branch]


def pr_files(cache, number):
    """Changed paths of a pull request, old paths of renames included."""

    def fetch():
        files, page = [], 1
        while page <= 30:
            data = get(f"pulls/{number}/files", {"per_page": 100, "page": page}) or []
            files += data
            if len(data) < 100:
                break
            page += 1
        return {
            "files": [
                {"filename": f["filename"], "previous_filename": f.get("previous_filename")}
                for f in files
            ]
        }

    data = cached_json(f"{cache}/prfiles/{number}.json", fetch)
    paths = []
    for entry in data["files"]:
        paths.append(entry["filename"])
        if entry.get("previous_filename"):
            paths.append(entry["previous_filename"])
    return paths


def main():
    """Fetch, parse and write data/*.csv."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--since", default="2026-08-01", help="first run creation date (YYYY-MM-DD)"
    )
    parser.add_argument("--cache", default=os.path.expanduser("~/.cache/te_ci_work_queue_doc"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "data"))
    args = parser.parse_args()

    SESSION.headers.update(
        {"Authorization": f"Bearer {token()}", "Accept": "application/vnd.github+json"}
    )
    for sub in ("raw", "logs", "prfiles"):
        os.makedirs(f"{args.cache}/{sub}", exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    runs = list_runs(args.since)
    print(f"{len(runs)} runs since {args.since}", file=sys.stderr)
    with ThreadPoolExecutor(8) as pool:
        jobs = list(pool.map(lambda run: run_jobs(args.cache, run), runs))
    rows = []
    for run, data in zip(runs, jobs):
        rows += flatten(run, data["attempts"])
    frame = pd.DataFrame(rows)
    frame.to_csv(f"{args.out}/jobs.csv", index=False)
    print(f"{len(frame)} jobs", file=sys.stderr)

    sgpu = frame[(frame.jtype == "sgpu") & frame.era.notna()]
    with ThreadPoolExecutor(8) as pool:
        logs = dict(zip(sgpu.job_id, pool.map(lambda jid: job_log(args.cache, jid), sgpu.job_id)))

    static_rows, item_rows = [], []
    for job in sgpu.itertuples():
        if job.era == "static":
            suites = static_suites(logs[job.job_id])
            if suites:
                static_rows.append(dict(job_id=job.job_id, **suites))
        else:
            gpus, items = queue_items(logs[job.job_id])
            for item in items:
                item_rows.append(dict(job_id=job.job_id, gpus=gpus, **item))
    pd.DataFrame(static_rows).to_csv(f"{args.out}/static_suites.csv", index=False)
    pd.DataFrame(item_rows).to_csv(f"{args.out}/queue_items.csv", index=False)
    print(
        f"{len(static_rows)} static jobs with suite timings, {len(item_rows)} queue items",
        file=sys.stderr,
    )

    # Which suites each PR run would select under the current rules
    sys.path.insert(0, os.path.join(REPO_ROOT, ".github", "scripts"))
    import change_scope  # pylint: disable=import-outside-toplevel

    by_id = {run["id"]: run for run in runs}
    gpu_runs = frame[
        (frame.kind == "pr") & frame.jtype.isin(["sgpu", "mgpu-torch", "mgpu-jax", "mgpu-queue"])
    ]
    by_branch, scope_rows = {}, []
    for run_id in sorted(gpu_runs.run_id.unique()):
        number = pr_number(by_id[run_id], by_branch)
        if number is None:
            continue
        needed, _ = change_scope.scope(pr_files(args.cache, number))
        scope_rows.append(
            dict(
                run_id=run_id,
                pr=number,
                sgpu=change_scope.ordered(needed, change_scope.SGPU_SUITES),
                mgpu=change_scope.ordered(needed, change_scope.MGPU_SUITES),
                everything=needed == change_scope.EVERYTHING,
            )
        )
    pd.DataFrame(scope_rows).to_csv(f"{args.out}/pr_scope.csv", index=False)
    print(f"{len(scope_rows)} PR runs classified", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
