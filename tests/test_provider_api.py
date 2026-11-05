import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.requests = types.ModuleType("requests")
        self.requests.post = Mock()
        path = Path(__file__).resolve().parents[1] / "alphas2/call_api.py"
        spec = importlib.util.spec_from_file_location("provider_api_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"requests": self.requests}):
            spec.loader.exec_module(self.module)
        self.client = self.module.ProviderAPIMixin()

    def test_import_does_not_call_provider(self):
        self.requests.post.assert_not_called()

    def test_missing_credentials_fail_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            for method in (self.client.call_doubao_api, self.client.call_qianwen_api):
                with self.assertRaises(KeyError):
                    method("synthetic prompt")
        self.requests.post.assert_not_called()

    def test_provider_reads_environment_and_returns_response(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "synthetic result"}}]}
        self.requests.post.return_value = response
        with patch.dict(os.environ, {"DOUBAO_API_KEY": "test-only", "DASHSCOPE_API_KEY": "test-only"}):
            for method in (self.client.call_doubao_api, self.client.call_qianwen_api):
                model, content = method("synthetic prompt", max_retries=1)
                self.assertTrue(model)
                self.assertEqual(content, "synthetic result")
                self.assertEqual(self.requests.post.call_args.kwargs["headers"]["Authorization"], "Bearer test-only")

    def test_random_dispatch_preserves_retry_limit(self):
        selected = Mock(return_value=("test-model", "result"))
        with patch.object(self.module.random, "choice", return_value=selected):
            self.assertEqual(self.client.call_random_api("prompt", max_retries=2), ("test-model", "result"))
        selected.assert_called_once_with("prompt", max_retries=2)


if __name__ == "__main__":
    unittest.main()
