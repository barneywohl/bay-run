# Bay Run

Bay Run is a free-at-launch, OpenAI-compatible REST and MCP service for small-model inference at the canonical origin: <https://run.huggingbay.xyz>.

## Free launch: 30 seconds

The launch tier is free. Mint a demo bearer with a form-encoded `client_credentials` request, then call the API. The published guardrails are 60 requests/minute and 1,500/day; demo bearers expire after 24 hours.

Start with the live, copy-paste quickstart: **<https://run.huggingbay.xyz/quickstart>**.

The repository’s standalone demo is a small classify → signed receipt → verify flow:

```bash
python examples/bakeoff.py
```

```python
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://run.huggingbay.xyz"

def post(path, body, headers):
    request = Request(BASE + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urlopen(request))

token_request = Request(
    BASE + "/oauth/token",
    data=urlencode({"grant_type": "client_credentials"}).encode(),
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)
token = json.load(urlopen(token_request))["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
result = post("/v1/classify", {"input": "The setup was quick and clear.", "model": "auto"}, headers)
receipt = result["provenance_receipt"]
verified = post("/v1/provenance/verify", {"receipt": receipt}, headers)
assert verified.get("valid") is True or verified.get("receipt_valid") is True
print(result["labels"][0])
print("receipt verified")
```

This demo uses the real `/v1/classify` and provenance-verification surfaces; it does not invent a bake-off endpoint.

## REST and SDKs

The REST surface includes OpenAI-compatible `chat/completions` (including SSE streaming and tools), embeddings, reranking, classification, summarization, RAG, calculation, JSON validation, and memory. The thin SDKs live beside this checkout during build/test:

```bash
pip install -e ../bayrun-sdk/python
npm install ../bayrun-sdk/js
```

```python
from bayrun import Client

c = Client()
answer = c.classify("A helpful result.")
stream = c.chat([{"role": "user", "content": "Say hello."}], stream=True)
```

```js
import { Client } from "bayrun";

const c = new Client();
const answer = await c.classify("A helpful result.");
const stream = await c.chat([{ role: "user", content: "Say hello." }], { stream: true });
```

Adapter snippets for OpenAI Python, Vercel AI SDK, LangChain, and LlamaIndex are in [`docs/integrations.md`](docs/integrations.md).

## MCP: exactly three tools

Point an MCP client at **<https://run.huggingbay.xyz/mcp/>**. The live server exposes exactly:

- `get_task_quote`
- `run_task`
- `verify_result`

Token and request-shape details are kept in the live quickstart so the copy-paste contract stays current.

## Trust and data policy

REST results carry signed provenance receipts, and task execution returns a signed receipt that can be checked with the verification surface. Read the receipt semantics and the live verification request at <https://run.huggingbay.xyz/quickstart>.

Before sending data, read the canonical policy: <https://run.huggingbay.xyz/.well-known/data-policy.json>. The published policy says Bay Run does not train on inputs.

## Hugging Face drop-in mirror catalog

Hugging Bay’s mirror catalog is available at <https://huggingbay.xyz>. For Hugging Face Hub clients that honor `HF_ENDPOINT`, point the client at the mirror origin:

```bash
export HF_ENDPOINT=https://huggingbay.xyz
```

The catalog currently lists 893+ mirrored models. Check the mirror’s current serving status before relying on a specific model or repository path.
