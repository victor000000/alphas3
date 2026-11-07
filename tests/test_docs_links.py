"""Repository documentation validation uses temporary files, never network access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


class DocsTests(unittest.TestCase):
    def test_missing_heading_and_target_fail(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/check_docs.py'
        spec = importlib.util.spec_from_file_location('docs_checker_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / 'README.md'
            p.write_text('# Overview\n[ok](#overview) [bad](#missing) [file](absent.md)', encoding='utf-8')
            count, errors = module.check(root)
            self.assertEqual(count, 1)
            self.assertEqual(len(errors), 2)
            p.write_text('# Overview\n[ok](#overview) [remote](https://example.invalid)', encoding='utf-8')
            self.assertEqual(module.check(root)[1], [])
