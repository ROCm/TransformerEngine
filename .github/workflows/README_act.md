# Running GitHub CI locally with `act`

The `act` tool (https://github.com/nektos/act) allows you to run GitHub Actions
workflows locally using Docker. This can be useful for debugging CI logic,
reproducing failures, or iterating on workflow changes without pushing commits
to GitHub.

## Installation

There is currently no official Ubuntu package for `act`. Installation options
include:

- Pre-built binaries, available at  
  https://nektosact.com/installation/index.html#pre-built-artifacts
- The conda-forge package:  
  https://github.com/conda-forge/act-feedstock

Make sure that Docker is installed and that your user has permission to access
the Docker daemon.

## Running the TransformerEngine CI workflow

To run the CI job locally, use:

```console
$ act -j build_and_test --matrix runner:linux-mi325-8 -P linux-mi325-8=-self-hosted
```

## Caching and Cleanup

`act` caches intermediate data (such as runner state and action checkouts) in
`~/.cache/act`. This cache can grow significantly over time and is safe to
delete if disk space becomes an issue.

To inspect the current size of the cache:

```console
$ du -sh ~/.cache/act/
```

The workflow launches Docker containers internally. If an `act` run
is interrupted, containers may be left running and consume disk space. Use
`docker ps` and `docker rm -f` to clean up if needed.
