# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

from transformer_engine.pytorch.cpp_extensions import gemm
from transformer_engine.pytorch.module.base import get_workspace


def use_hipblaslt():
    return (os.getenv("NVTE_USE_HIPBLASLT") is not None
            or os.getenv("NVTE_USE_ROCBLAS") is None )


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
            and "aidx" in head), "Invalid CSV format"
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


@pytest.mark.skipif(not use_hipblaslt(), reason="Autotune requires hipBLASLt")
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

        #Modify algo index
        algo_index = int(algo0["aidx"])
        algos=[copy.copy(algo0)]
        algos[0]["aidx"] = str(algo_index + 1);
        write_storage(fname, head, algos)
        subprocess.run(run_args)
        algos = read_storage(ofile)
        assert len(algos)==1, "Expected 1 cached record"
        assert (algo0["aidx"], algo0["algo_id"]) == (algos[0]["aidx"], algos[0]["algo_id"]), "Invalid algo IDX"

        # Configure autotune range so current cached algo is out of it 
        # and cache new value
        os.environ["TE_HIPBLASLT_ALGO_LOAD"] = ""
        os.environ["TE_HIPBLASLT_ALGO_SAVE"] = fname
        os.environ["TE_HIPBLASLT_ALGO_SELECTION"] = str(algo_index + 1)
        subprocess.run(run_args)
        algos = read_storage(fname)
        assert len(algos)==1, "Expected 1 cached record"
        algo1 = copy.copy(algos[0])
        assert algo0["algo_id"] != algo1["algo_id"], "Unexpected algo ID"

        #Restore autotune range begining, the new algo should still be used
        os.environ["TE_HIPBLASLT_ALGO_LOAD"] = fname
        del os.environ["TE_HIPBLASLT_ALGO_SELECTION"]
        subprocess.run(run_args)
        algos = read_storage(fname)
        assert len(algos)==1, "Expected 1 cached record"
        assert algo1 == algos[0], "Invalid algo ID"

    finally:
        shutil.rmtree(storage_dir)
        pass


def run_gemm():
    N = 32
    datatype = torch.float16    
    inp = torch.randn((N, N), device="cuda", dtype=datatype)
    _, _, _ = gemm(A=inp, B=inp, dtype=datatype, workspace=get_workspace())


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run_gemm()



