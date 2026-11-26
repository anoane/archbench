# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep the public benchmark catalogue bound to the effective full registry."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "update_readme_catalog.py"
SPEC = importlib.util.spec_from_file_location("update_readme_catalog", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load README catalogue generator")
CATALOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CATALOG)


class ReadmeCatalogTests(unittest.TestCase):
    def test_generated_catalog_matches_effective_full_selection(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        expected = CATALOG.replaced_readme(readme, CATALOG.render_block())
        self.assertEqual(readme, expected)

    def test_catalog_is_complete_and_names_every_family(self):
        records, total = CATALOG.inventory()
        self.assertEqual(total, 404129)
        self.assertEqual(tuple(records), CATALOG.FAMILIES)
        rendered = CATALOG.render_block()
        for family in CATALOG.FAMILIES:
            self.assertIn(f"| `{family}` |", rendered)


if __name__ == "__main__":
    unittest.main()
