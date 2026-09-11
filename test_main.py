import io
import os
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

import main


class DocumentEndpointTests(unittest.TestCase):
    def setUp(self):
        main.get_engine.cache_clear()

    @patch("main.get_engine")
    def test_expanded_archive_limit_rejects_zip_bomb_shape(self, get_engine):
        engine = MagicMock()
        engine.SUPPORTED_EXTENSIONS = {".xlsx"}
        get_engine.return_value = engine
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("xl/worksheets/sheet1.xml", "0" * 4096)

        with patch.dict(os.environ, {"DOCPROC_MAX_EXPANDED_BYTES": "1024"}):
            with self.assertRaises(HTTPException) as raised:
                main._validate_file(payload.getvalue(), "model.xlsx")

        self.assertEqual(raised.exception.status_code, 413)

    @patch("main.get_engine")
    def test_image_upload_is_rejected_when_ocr_is_not_deployed(self, get_engine):
        engine = MagicMock()
        engine.SUPPORTED_EXTENSIONS = {".pdf", ".xlsx"}
        get_engine.return_value = engine

        with self.assertRaises(HTTPException) as raised:
            main._validate_file(b"image", "scan.png")

        self.assertEqual(raised.exception.status_code, 415)


if __name__ == "__main__":
    unittest.main()
