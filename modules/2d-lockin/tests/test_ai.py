from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from stmfft.ai import AIConfig, build_openrouter_payload, extract_response_text, load_key_from_env_file, parse_vision_models


class AITests(unittest.TestCase):
    def test_payload_places_text_before_local_image(self) -> None:
        png = b"\x89PNG\r\n\x1a\nmock"
        payload = build_openrouter_payload(AIConfig(model="vision/model"), {"qx": 1.2}, png, "分析这个峰")
        self.assertEqual(payload["model"], "vision/model")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        encoded = content[1]["image_url"]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), png)

    def test_extract_text(self) -> None:
        response = {"choices": [{"message": {"content": "结果"}}]}
        self.assertEqual(extract_response_text(response), "结果")

    def test_load_key_without_printing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env.local"
            path.write_text("OTHER=x\nOPENROUTER_API_KEY=secret-value\n", encoding="utf-8")
            self.assertEqual(load_key_from_env_file(path), "secret-value")

    def test_filter_vision_models(self) -> None:
        response = {
            "data": [
                {"id": "vision/model", "name": "Vision", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}},
                {"id": "text/model", "name": "Text", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
            ]
        }
        self.assertEqual([item["id"] for item in parse_vision_models(response)], ["vision/model"])


if __name__ == "__main__":
    unittest.main()
