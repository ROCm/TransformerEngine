## JAX Fused-Attention Benchmarking
The benchmarking process is split into two stages: *generating* the timing data, and *visualizing* the timing data. The following steps assume you are located in `TransformerEngine/benchmarks/attention` (i.e. where this README is located). First, ensure that you install requirements via `pip install -r requirements.txt`.

Note: Only forward timings are supported at this point.

### Generate Timing Data
Run the following command to generate timing data. Please use the `-h` flag for details on the available arguments. The output csv, which will later be parsed to generate the interactive visualizations, is generated in the same directory as the script, since that is where the visualization stage expects it.

```bash
XLA_FLAGS="--xla_gpu_graph_level=0" python benchmark_attention_jax.py --fwd-v3 --bwd-v3 -v
```

The `XLA_FLAGS` environment variable is necessary in order to ensure that the timings can be dumped at the C++ backend level.

Note that you can also specify a target HIP device via `HIP_VISIBLE_DEVICES=<device index>` which may be useful in isolating the benchmarks to an unused GPU on a shared machine.

### Generating Interactive Visualization
Simply run `panel serve panel_app.py`. This will launch a web-service on your localhost which displays an interactive visualization app. If launching on a remote server, VS code users will find that their IDE automatically port-forwards the correct ports, and thus they may directly open the link that is printed after running the command. Other users must ensure that their `ssh` into the remote server includes an appropriate port-forwarding (the default port is `5006`).