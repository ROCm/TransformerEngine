# Runbook: diagnostics snapshot triage

`python -m transformer_engine.te_rocm_diagnostics` prints the bug-report snapshot. Read it
top-down:

- **transformer_engine_file**: WHICH tree answered — repo checkout, overlay, or site-packages.
  Most "impossible" bugs are the wrong tree (cwd shadows PYTHONPATH; run from a neutral cwd).
- **core_abi_version**: must match the loader's expectation (mismatch refuses at import with
  TE_ROCM_CORE_ABI; means mixed core/python versions).
- **seam.alias_identity**: True means `transformer_engine_torch` IS the compiled ROCm module;
  False means something re-imported over the alias — find the import-order violator.
- **registry**: which implementation each op family selected and every rejection reason. An
  env flag that "does nothing" usually shows here as a logged rejection (predicate refused) or
  as policy frozen before the flag was set (flags are read once, at first selection).
- **overlay**: bundle hash + patch ids — quote these in any report; they identify the exact
  patched surface. No overlay section = installed wheel or raw checkout.
- **env**: the NVTE_* flags actually visible to the process.
