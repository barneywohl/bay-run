#!/usr/bin/env python3
"""Run one live Bay Run classify -> receipt -> verify flow."""

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
