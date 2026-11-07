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

    def test_hunyuan_sends_the_model_it_reports(self):
        import json

        request = Mock()
        response = types.SimpleNamespace(Choices=[types.SimpleNamespace(Message=types.SimpleNamespace(Content="result"))])
        sdk_client = Mock()
        sdk_client.ChatCompletions.return_value = response
        names = [
            "tencentcloud", "tencentcloud.common", "tencentcloud.common.profile",
            "tencentcloud.common.profile.client_profile", "tencentcloud.common.profile.http_profile",
            "tencentcloud.common.exception", "tencentcloud.common.exception.tencent_cloud_sdk_exception",
            "tencentcloud.hunyuan", "tencentcloud.hunyuan.v20230901",
        ]
        modules = {name: types.ModuleType(name) for name in names}
        modules["tencentcloud.common"].credential = types.SimpleNamespace(Credential=Mock())
        modules["tencentcloud.common.profile.client_profile"].ClientProfile = Mock
        modules["tencentcloud.common.profile.http_profile"].HttpProfile = Mock
        modules["tencentcloud.common.exception.tencent_cloud_sdk_exception"].TencentCloudSDKException = RuntimeError
        modules["tencentcloud.hunyuan.v20230901"].hunyuan_client = types.SimpleNamespace(
            HunyuanClient=Mock(return_value=sdk_client)
        )
        modules["tencentcloud.hunyuan.v20230901"].models = types.SimpleNamespace(
            ChatCompletionsRequest=Mock(return_value=request)
        )
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
            "TENCENTCLOUD_SECRET_ID": "test-only", "TENCENTCLOUD_SECRET_KEY": "test-only"
        }), patch.object(self.module.random, "choice", return_value="hunyuan-turbos-latest"):
            model, content = self.client.call_hunyuan_api("prompt", max_retries=1)
        sent = json.loads(request.from_json_string.call_args.args[0])
        self.assertEqual(sent["Model"], model)
        self.assertEqual(model, "hunyuan-turbos-latest")
        self.assertEqual(content, "result")


if __name__ == "__main__":
    unittest.main()
