# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os, sys
import copy
import pytest
import tempfile
import shutil
import subprocess
import csv
import warnings

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch.cpp_extensions import general_gemm


storage_fname = "te_algo"


def dump_storage(fname):
    print("========")
    with open(fname, "r") as ifile:
        for row in ifile:
            print(row)
    print("========")


def analyse_storage(fname):
    with open(fname, "r") as ifile:
        reader = csv.DictReader(ifile)
        next(reader)
        head = reader.fieldnames
    assert ("m" in head and "algo_id" in head and  "ws_min" in head and "ws_max" in head
            ), "Invalid CSV format"
    return head

def read_storage(fname):
    data = []
    with open(fname, "r") as ifile:
        reader = csv.DictReader(ifile)
        for row in reader:
            data.append(row)
    return data


def write_storage(fname, head, data):
    with open(fname, "w") as ofile:
        writer = csv.DictWriter(ofile, fieldnames = head, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


@pytest.mark.skipif(not IS_HIP_EXTENSION, reason="Autotune requires ROCm TE")
def test_gemm_autotune():
    storage_dir = tempfile.mkdtemp();
    fname = storage_dir+"/"+storage_fname
    script = os.path.abspath(__file__)
    try:
        os.environ["TE_HIPBLASLT_ALGO_LOAD"] = fname
        os.environ["TE_HIPBLASLT_ALGO_SAVE"] = fname
        run_args = ["python", script, "--run"]

        #Initial algo creation
        subprocess.run(run_args)
        head = analyse_storage(fname)
        algos = read_storage(fname)
        assert len(algos)==1, "Expected 1 cached record"
        algo0 = copy.copy(algos[0])

        ofile = fname+".1"
        os.environ["TE_HIPBLASLT_ALGO_SAVE"] = ofile

        #Stored algo is resolved once and reused
        out = subprocess.run(run_args + ["2"], capture_output=True, text=True,
                             env=dict(os.environ, TE_HIPBLASLT_LOG_TUNING="1")).stdout
        assert "Cannot get corresponding solution" not in out, "Stored algo was rejected"
        assert out.count("Use hipBLASLt algo") == 1, "Stored algo was not cached"

        #Unused cache entries
        algos[0]["m"] = "999"+algos[0]["m"] # fake record for different shape
        write_storage(fname, head, algos)
        subprocess.run(run_args)
        algos = read_storage(ofile)
        assert len(algos)==2, "Expected 2 cached records"
        assert algo0 == algos[1], "Invalid algo"

        #Adjust workspace size
        ws_max = int(algo0["ws_max"])
        if (ws_max > 0):
            algos=[copy.copy(algo0)]
            algos[0]["ws_max"] = str(ws_max - 1) # decrease WS range should restore size
            ws_min = int(algos[0]["ws_min"])
            if (ws_max - ws_min > 1):
                ws_min = ws_min + 1
                algos[0]["ws_min"] = str(ws_min)
            write_storage(fname, head, algos)
            subprocess.run(run_args)
            algos = read_storage(ofile)
            assert len(algos)==1, "Expected 1 cached record"
            assert (str(ws_min), str(ws_max)) == (algos[0]["ws_min"], algos[0]["ws_max"]), "Invalid WS size"
        else:
            warnings.warn("Cached algo Workspace size is 0")
            
    finally:
        shutil.rmtree(storage_dir)
        pass


@pytest.mark.skipif(not IS_HIP_EXTENSION, reason="Autotune requires ROCm TE")
def test_gemm_autotune_accumulate():
    storage_dir = tempfile.mkdtemp()
    fname = storage_dir+"/"+storage_fname
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("TE_HIPBLASLT_")}
        env.update(TE_HIPBLASLT_TUNING_RUN_COUNT="5", TE_HIPBLASLT_LOG_TUNING="1",
                   TE_HIPBLASLT_ALGO_SAVE=fname)
        res = subprocess.run(["python", os.path.abspath(__file__), "--accumulate"], env=env,
                             capture_output=True, text=True)
        assert res.returncode == 0, "Tuning changed the result of an accumulating GEMM\n" + res.stderr
        out = res.stdout
        assert out.count("Skip hipBLASLt algo tuning") == 1, "Expected one warning for the shape"
        # Once for the untuned pick, once for the tuned one
        assert out.count("Use hipBLASLt algo") == 2, "Untuned algo was not cached"
        assert out.count("Perform hipBLASLt algo selection") == 1, "Non-accumulating GEMM was not tuned"
        assert len(read_storage(fname)) == 1, "Expected 1 cached record, only for the tuned algo"
    finally:
        shutil.rmtree(storage_dir)


def run_gemm(calls=1):
    N = 32
    datatype = torch.float16    
    inp = torch.randn((N, N), device="cuda", dtype=datatype)
    for _ in range(calls):
        _, _, _, _ = general_gemm(A=inp, B=inp, out_dtype=datatype)


def run_gemm_accumulate():
    N = 32
    datatype = torch.float16
    inp = torch.randn((N, N), device="cuda", dtype=datatype)
    out = torch.randn((N, N), device="cuda", dtype=datatype)
    # Untuned accumulating calls, a tuned overwriting call, then an accumulating call using the tuned algo
    for accumulate in (True, True, False, True):
        ref = inp.float() @ inp.float().t()
        if accumulate:
            ref += out.float()
        general_gemm(A=inp, B=inp, out_dtype=datatype, out=out, accumulate=accumulate)
        torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run_gemm(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif sys.argv[1] == "--accumulate":
        run_gemm_accumulate()



