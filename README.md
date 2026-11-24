# CS-370 OEL: Autonomous Delivery Agent
# AI-OEL
Members: see members.py
Setup: pip install -r requirements.txt
Run everything: python run_all.py
UI: streamlit run ui/app.py

## Use a real LLM backend

The dispatcher uses OpenAI's `gpt-6-luna` model when `OPENAI_API_KEY` is configured; it is OpenAI's
most efficient model for focused, high-volume tasks. It costs $0.10 per 1M input tokens and $0.50 per
1M output tokens at standard rates. The offline default remains a mock when no key is configured, and
mock results must not be reported as real LLM results. See [OpenAI model details](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [current pricing](https://developers.openai.com/api/docs/pricing).

Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` in `.env`. The file is ignored by Git.
Run `python -m src.llm_agent --ask "Deliver stop 3 first"` for one live request. The Streamlit UI
also has an **LLM dispatcher** tab that uses this configuration.

The dispatcher also accepts OpenAI-compatible endpoints, including Ollama Cloud, local vLLM, and
Mistral. Set credentials in environment variables; never commit API keys.

For Ollama Cloud in PowerShell:

```powershell
$env:OEL_LLM_BACKEND = "openai"
$env:OEL_LLM_BASE_URL = "https://ollama.com/v1"
$env:OEL_LLM_MODEL = "deepseek-v4.1-flash"
$env:OLLAMA_API_KEY = "<your rotated key>"
python -m src.llm_agent --ask "Deliver stop 3 first"
```

For a vLLM server, start the model server in Linux/WSL with `vllm serve <model> --host 0.0.0.0 --port 8000`.
Then set `OEL_LLM_BASE_URL` to `http://localhost:8000/v1`, `OEL_LLM_MODEL` to the served model name,
and `VLLM_API_KEY` if the server was started with `--api-key`. Mistral's compatible endpoint is
`https://api.mistral.ai/v1`, with the key in `MISTRAL_API_KEY`.

`python -m src.llm_agent --ask ...` makes a small one-request check. The full `python run_all.py` run
repeats 20 requests in raw and validated modes and can consume provider quota or credits.
