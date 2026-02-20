"""Tests for agent creation and doc fetching."""

import pytest
from unittest.mock import patch, MagicMock

from dialect_mapper.basis.agent import _fetch_docs, OUTPUT_DIR


class TestFetchDocs:
    """Test documentation fetching."""

    def test_unknown_dialect(self):
        result = _fetch_docs("ABS", "unknown_dialect")
        assert "No docs URL" in result

    @patch("dialect_mapper.basis.agent.httpx.get")
    def test_successful_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html>ABS function docs</html>"
        mock_get.return_value = mock_resp
        result = _fetch_docs("ABS", "spark")
        assert "ABS function docs" in result

    @patch("dialect_mapper.basis.agent.httpx.get")
    def test_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp
        result = _fetch_docs("NONEXISTENT", "spark")
        assert "404" in result

    @patch("dialect_mapper.basis.agent.httpx.get", side_effect=Exception("timeout"))
    def test_network_error(self, mock_get):
        result = _fetch_docs("ABS", "spark")
        assert "timeout" in result


class TestConfig:
    """Test agent configuration values."""

    def test_output_dir(self):
        assert OUTPUT_DIR.name == "output"
