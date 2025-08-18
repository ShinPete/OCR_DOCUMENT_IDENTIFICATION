import os, shutil
from pathlib import Path
import pytesseract

def resolve_tesseract_cmd(cfg: dict | None = None) -> str:
    # 1) env var
    cmd = os.getenv("TESSERACT_CMD")

    # 2) config.yml field
    if not cmd and cfg:
        cmd = (cfg.get("ocr", {}) or {}).get("tesseract_cmd")

    # 3) PATH
    if not cmd:
        cmd = shutil.which("tesseract")

    # 4) common fallbacks
    if not cmd:
        for c in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/opt/homebrew/bin/tesseract",   # Apple Silicon Homebrew
            "/usr/local/bin/tesseract",      # Intel macOS / older Homebrew
            "/usr/bin/tesseract",            # Linux
        ]:
            if Path(c).exists():
                cmd = c; break

    if not cmd:
        raise RuntimeError(
            "Tesseract not found. Set TESSERACT_CMD, put ocr.tesseract_cmd in config.yml, "
            "or add tesseract to PATH."
        )

    pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd
