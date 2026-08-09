# Windows Run Guide

Open PowerShell inside the extracted `AeroECAD-Agent-v0.2` folder.

## 1. Install the updated local package

```powershell
uv sync
```

## 2. Confirm tests and Ollama access

```powershell
uv run python -m unittest discover -s tests -v
uv run aeroecad ollama-check --models qwen2.5:7b llama3.1:8b mistral:7b
```

Expected: nine tests pass and the Ollama check returns `"status": "ready"`.

If the Ollama check cannot connect, open another PowerShell window, run `ollama serve`, leave it open, and retry. Do not run `ollama pull`; the required models are already installed.

## 3. Run one-model smoke test

```powershell
uv run aeroecad llm-experiment --models qwen2.5:7b --modes llm_only retrieval_grounded hybrid_explainer --profile smoke
```

This performs 21 calls: seven balanced scenarios multiplied by three review modes. Progress is printed as `[completed/total] model | mode | scenario`.

## 4. Run the publication pilot

```powershell
uv run aeroecad llm-experiment --models qwen2.5:7b llama3.1:8b mistral:7b --modes llm_only retrieval_grounded hybrid_explainer --profile pilot
```

This performs 360 calls over 40 balanced scenarios per model/mode. Runtime depends on the hardware and should be estimated from the smoke test's median latency. You may stop with `Ctrl+C`; rerunning the same command continues from the saved responses.

## 5. Share the results

After the pilot finishes, send these two files for analysis and abstract drafting:

- `results\llm\llm_preliminary_results.md`
- `results\llm\llm_benchmark_summary.json`

Keep `results\llm\ollama_responses.jsonl` as the raw reproducibility record; it may be much larger.
