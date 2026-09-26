# Magicpin AI Challenge — Grounded Vera Bot

This implementation is deliberately built around the rejection criteria in the challenge:

- no fabricated facts/prices/dates/percentages
- no generic one-size-fits-all template for known triggers
- deterministic endpoint behavior
- versioned adaptive context storage
- exact required action fields
- URL removal/guard in message bodies
- duplicate suppression
- consent checks for customer-facing outreach
- auto-reply detection and escalation from send → wait → end
- explicit intent-transition handling
- graceful hostile/off-topic handling
- optional LLM composition with strict validation and deterministic fallback

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

## Optional LLM

The bot works without an LLM. For higher message quality, set an OpenAI-compatible chat-completions endpoint through environment variables:

```bash
export LLM_API_KEY='...'
export LLM_BASE_URL='https://YOUR_PROVIDER/v1'
export LLM_MODEL='YOUR_MODEL'
```

The LLM result is accepted only if it passes the grounded-message validator. Otherwise the deterministic composer is used.

## Verify

```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/metadata
```

Then use the supplied `judge_simulator.py` after configuring its own LLM provider/key.

## Important deployment rule

Do not restart the process during the judge run. The challenge allows in-memory state and expects the context to persist for the full test.

For a public deployment, bind to `0.0.0.0` and use the platform-provided `PORT`.
