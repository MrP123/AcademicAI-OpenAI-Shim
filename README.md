# OpenAI-compatible Shim for AcademicAI API
This project provides a FastAPI-based shim that translates OpenAI-compatible requests into AcademicAI API requests for the use with the [LiteLLM](https://github.com/BerriAI/litellm) proxy.
It is mainly intended for playing around with Claude Code without needing an Anthropic account by just pointing the the `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL` environment variables to the LiteLLM proxy. 

## Installation
Get an AcademicAI API Client ID and Secret and put it in the `.env` file. A template is provided in `template.env`.

Install the dependencies with `uv`
```powershell
uv sync
```
and also install the LiteLLM proxy as a `uv` tool using
```powershell
uv tool install 'litellm[proxy]'
```

To run the shim just run
```powershell
uv run ./app.py
```
and afterwards start the LiteLLM proxy with
```powershell
litellm --config config.yaml --host 0.0.0.0 --port 4000 --debug
```
where the debug flag is optional but recommended for development.

### Run Claude Code
Install Claude Code as described in the [Claude Code README](https://github.com/anthropics/claude-code#get-started) and then run it with the following environment variables set:

```powershell
$env:ANTHROPIC_BASE_URL="http://0.0.0.0:4000"     # This is the LiteLLM proxy URL incl. port -- make sure to use the same port as in the previous command
$env:ANTHROPIC_AUTH_TOKEN="dummy_token"           # As the LiteLLM proxy does not require authentication, you can just set a dummy token here
$env:ANTHROPIC_MODEL="aai_claude-opus-4-6"        # main model --> must be one from the configured ones in the LiteLLM proxy config.yaml
$env:ANTHROPIC_SMALL_FAST_MODEL="aai_gpt-5-nano"  # background tasks
$env:ANTHROPIC_API_KEY=""                         # keep empty so Claude Code does not fall back to Anthropic or forces you to log in
claude
```

As the AcademicAI API does not support tool calling, the following tools are provided manually/locally in this shim to get a basic Claude Code workflow going. This is not ideal, but in practice not as limiting as one might suspect.
The provided tools are:
- `Write`: Writes a text file with the given content
- `Read`: Reads a (text) file and returns its content
- `Edit`: Performs exact string replacement in a text file
- `Grep`: Searches for a regex pattern in a text file and returns the matching lines
It uses a special text format with a fenced JSON block to communicate with the tools