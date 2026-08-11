import json
import unittest
from pathlib import Path


MANIFEST = json.loads((Path(__file__).parent / "server.json").read_text())
PUBLISHER = MANIFEST["_meta"]["io.modelcontextprotocol.registry/publisher-provided"]


class ServerManifestTests(unittest.TestCase):
    def test_runtime_inventory_and_auth_boundary_are_current(self):
        self.assertEqual(MANIFEST["version"], "1.4.1")
        self.assertEqual(MANIFEST["remotes"][0]["url"], "https://run.huggingbay.xyz/mcp/")
        self.assertFalse(MANIFEST["remotes"][0]["headers"][0]["isRequired"])
        self.assertEqual(len(PUBLISHER["tools"]), 21)
        self.assertEqual(PUBLISHER["tools"][0], "try_bay_run")
        self.assertIn("parse_pdf", PUBLISHER["tools"])

    def test_first_call_is_exact_public_fixed_proof(self):
        first = PUBLISHER["io.github.barneywohl/first-call"]
        self.assertEqual(first["schema"], "bay-run.first-call.v1")
        self.assertEqual(first["kind"], "activation_proof")
        self.assertFalse(first["authRequired"])
        self.assertFalse(first["usageCredit"])
        self.assertEqual(first["request"], {
            "jsonrpc": "2.0",
            "id": "bay-run-first-call",
            "method": "tools/call",
            "params": {"name": "try_bay_run", "arguments": {}},
        })
        self.assertEqual(first["success"]["structuredContent"]["proof"]["result"], 42)

    def test_synthetic_traffic_is_labeled_but_never_hidden(self):
        telemetry = PUBLISHER["telemetry"]
        self.assertEqual(
            telemetry["syntheticTrafficHeader"],
            {"name": "X-Bay-Run-Traffic-Class", "value": "synthetic"},
        )
        self.assertEqual(
            telemetry["optionalClientSessionHeader"], "X-Bay-Run-Client-Session"
        )
        self.assertIn("human_via_agent", telemetry["clientClasses"])
        self.assertIn("all events are retained", telemetry["note"])
        self.assertIn("never infers a human", telemetry["note"])


if __name__ == "__main__":
    unittest.main()
