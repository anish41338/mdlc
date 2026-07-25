# Kaggle runbook (copy-paste cells)

Notebook settings: **Accelerator = GPU T4 x1** (x2 wastes quota; we use one),
Internet ON, Persistence off.

## Cell 1 — clone + full run

```bash
%%bash
git clone https://github.com/<OWNER>/mdlc.git
cd mdlc
bash tools/kaggle_run.sh
```

If the repo is private, use a fine-grained PAT:
`git clone https://<PAT>@github.com/<OWNER>/mdlc.git`

## Cell 2 — package results for download

```bash
%%bash
cd mdlc
RUN=$(ls -dt artifacts/gpu_run_* | head -1)
zip -r /kaggle/working/mdlc_gpu_results.zip \
  "$RUN" artifacts/tune_cache.json docs/BENCHMARKS.md
echo "download mdlc_gpu_results.zip from the output panel"
```

## Cell 3 (alternative) — commit results straight back

```bash
%%bash
cd mdlc
git config user.email "you@example.com" && git config user.name "gpu-runner"
git checkout -b gpu-results-$(date +%Y%m%d)
git add artifacts/gpu_run_* artifacts/tune_cache.json docs/BENCHMARKS.md
git commit -m "GPU run: T4 tune cache + benchmark artifacts"
git push origin HEAD    # needs the PAT clone from Cell 1
```

## If something fails

- **`pytest -m gpu` failures**: copy the full `artifacts/gpu_run_*/pytest_gpu.txt`
  back — the fix loop is on the CPU side.
- **NVRTC "unsupported gpu architecture"**: confirm `ctx.arch` printed `sm_75`;
  the compile flag is `--gpu-architecture=compute_75` derived from it.
- **OOM during benchmark**: T4 has 16 GB; if torch+ORT+mdlc coexist badly,
  rerun the suite with one system commented out and note it in the artifact.
- Bring back *whatever* was produced — partial artifacts still move the loop.
