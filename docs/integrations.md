# Bay Run integrations

All four adapters target the canonical OpenAI-compatible base URL `https://run.huggingbay.xyz/v1`.

Each snippet mints the free launch bearer when `BAYRUN_API_KEY` is not already set. Keep the bearer in an environment variable or secret manager; do not paste it into source control.

## OpenAI Python

Install with `pip install openai requests`, then run:

```python
import os
import requests
from openai import OpenAI

token = os.getenv("BAYRUN_API_KEY") or requests.post(
    "https://run.huggingbay.xyz/oauth/token",
    data={"grant_type": "client_credentials"},
).json()["access_token"]
client = OpenAI(base_url="https://run.huggingbay.xyz/v1", api_key=token)
response = client.chat.completions.create(
    model="auto", messages=[{"role": "user", "content": "Reply with one word: hello"}]
)
print(response.choices[0].message.content)
```

## Vercel AI SDK

Install with `npm install ai @ai-sdk/openai-compatible`, then run as an ESM module:

```js
import { generateText } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

const token = process.env.BAYRUN_API_KEY ?? (await fetch("https://run.huggingbay.xyz/oauth/token", {
  method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" },
  body: "grant_type=client_credentials",
})).then((response) => response.json()).then((body) => body.access_token);
const bayrun = createOpenAICompatible({ name: "bayrun", apiKey: token, baseURL: "https://run.huggingbay.xyz/v1" });
const { text } = await generateText({ model: bayrun("auto"), prompt: "Reply with one word: hello" });
console.log(text);
```

## LangChain

Install with `pip install langchain-openai requests`, then run:

```python
import os
import requests
from langchain_openai import ChatOpenAI

token = os.getenv("BAYRUN_API_KEY") or requests.post(
    "https://run.huggingbay.xyz/oauth/token",
    data={"grant_type": "client_credentials"},
).json()["access_token"]
llm = ChatOpenAI(model="auto", api_key=token, base_url="https://run.huggingbay.xyz/v1")
print(llm.invoke("Reply with one word: hello").content)
```

## LlamaIndex

For an arbitrary OpenAI-compatible model id, install `pip install llama-index-llms-openai-like requests`, then run:

```python
import os
import requests
from llama_index.llms.openai_like import OpenAILike

token = os.getenv("BAYRUN_API_KEY") or requests.post(
    "https://run.huggingbay.xyz/oauth/token",
    data={"grant_type": "client_credentials"},
).json()["access_token"]
llm = OpenAILike(model="auto", api_base="https://run.huggingbay.xyz/v1", api_key=token, is_chat_model=True)
print(llm.complete("Reply with one word: hello"))
```

## Verification record

The OpenAI Python base-URL flow was exercised in the same-day Bay Run quickstart transcript. The Vercel AI SDK, LangChain, and LlamaIndex snippets are syntax-reviewed against their official adapter APIs but could not be executed from this checkout because the current shell has no DNS/network access and those optional packages are not installed. They must be run against the live origin before this document is treated as complete or published.
