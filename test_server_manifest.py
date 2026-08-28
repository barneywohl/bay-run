import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "server.json").read_text())
PUBLISHER = MANIFEST["_meta"]["io.modelcontextprotocol.registry/publisher-provided"]
README = (ROOT / "README.md").read_text()
MCP_QUICKSTART = (ROOT / "demo" / "mcp-quickstart.md").read_text()
PUBLIC_MCP_DOCS = [
    README,
    MCP_QUICKSTART,
    (ROOT / "demo" / "README.md").read_text(),
    (ROOT / "flagship" / "README.md").read_text(),
    (ROOT / "demo" / "mcp" / "bay_run_mcp.py").read_text(),
]
PRIMARY_TOOLS = ["coprocessor", "run_pin", "solve_task"]
LEGACY_TOOLS = ["get_task_quote", "run_task", "verify_result"]


class PublicContractTests(unittest.TestCase):
    def test_manifest_matches_live_default_contract(self):
        self.assertEqual(MANIFEST["version"], "2.0.0")
        self.assertLessEqual(len(MANIFEST["description"]), 100)
        self.assertEqual(MANIFEST["remotes"][0]["url"], "https://run.huggingbay.xyz/mcp/")
        self.assertEqual(PUBLISHER["tools"], PRIMARY_TOOLS)
        self.assertTrue(set(LEGACY_TOOLS).isdisjoint(PUBLISHER["tools"]))

    def test_readme_leads_with_coprocessor_on_canonical_mcp(self):
        self.assertIn("https://run.huggingbay.xyz/mcp/", README)
        self.assertIn('"name":"coprocessor"', README)
        self.assertLess(README.index("`coprocessor`"), README.index("`run_pin`"))
        self.assertLess(README.index("`run_pin`"), README.index("`solve_task`"))

    def test_readme_does_not_publish_legacy_or_stale_default_onboarding(self):
        for legacy_tool in LEGACY_TOOLS:
            self.assertNotIn(legacy_tool, README)
        self.assertNotIn("/v1/classify", README)
        self.assertNotIn("classify →", README)
        self.assertNotRegex(README, re.compile(r"\b\d+[+]?(?:\s+mirrored)?\s+models\b", re.IGNORECASE))
        self.assertIn("https://run.huggingbay.xyz/status.json", README)

    def test_companion_mcp_quickstart_uses_the_current_front_door(self):
        self.assertIn("https://run.huggingbay.xyz/mcp/", MCP_QUICKSTART)
        self.assertIn('"name":"coprocessor"', MCP_QUICKSTART)
        self.assertNotIn("bay-run-mvp-889989800693.us-central1.run.app", MCP_QUICKSTART)
        self.assertNotIn("20-tool", MCP_QUICKSTART)
        for legacy_tool in LEGACY_TOOLS:
            self.assertNotIn(legacy_tool, MCP_QUICKSTART)

    def test_public_mcp_docs_do_not_claim_the_old_tool_inventory(self):
        stale_inventory = re.compile(r"\b(?:all\s+)?20[- ]?(?:MCP\s+)?tools?\b", re.IGNORECASE)
        for document in PUBLIC_MCP_DOCS:
            self.assertNotRegex(document, stale_inventory)

    def test_readme_does_not_advertise_unreleased_hf_endpoint(self):
        self.assertNotIn("export HF_ENDPOINT=", README)
        self.assertRegex(README, re.compile(r"not\s+part of the public contract yet"))


if __name__ == "__main__":
    unittest.main()
