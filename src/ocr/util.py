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


def clean_text(raw_text):
    # Collapse whitespace
    text = re.sub(r"\s+", " ", raw_text).strip()

    # Remove common header/footer artifacts
    text = re.sub(r"Page\s*\d+|\d+\s*/\s*\d+", "", text, flags=re.IGNORECASE)

    # Strict zero replacement (only in CAPS words)
    text = re.sub(r"\b([A-Z]*?)0([A-Z]*?)\b", lambda m: m.group(0).replace("0", "O"), text)

    # Looser one replacement (fixes Th1s → This)
    text = re.sub(r"\b(\w*?)1(\w*?)\b", lambda m: m.group(0).replace("1", "l"), text)

    # Lowercase for consistency
    text = text.lower()

    # Optional: remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text

def preprocess_image(img):
    img_rgb = img.convert("RGB")
    enhancer = ImageEnhance.Contrast(img_rgb)
    img_contrast = enhancer.enhance(2.0)
    return img_contrast.convert("L").point(lambda x: 0 if x < 128 else 255, "1")


def compare_distributions(full_df, list_of_dfs, column, tolerance=1):
    true_pct = full_df[column].value_counts(normalize=True) * 100
    labels = true_pct.index
    results = {}
    for name, df in list_of_dfs:
        df_pct = df[column].value_counts(normalize=True) * 100
        df_pct = df_pct.reindex(labels, fill_value=0, )
        diff = (np.abs(true_pct - df_pct))
        bad = diff[diff > tolerance]
        ok = (diff <= tolerance).all()
        if ok:
            print(f"✅ Distributions are similar full vs {name}")

        else:
            print(f"🚨 Distributions from {name} are more than {tolerance}% "
                  f"different from full dataset {bad.index} with differences {bad.values}")
        results[name] = {'ok': bool(ok),
                         'max_drift': float(diff.max()),
                         'bad_labels': bad.index.tolist()}
    return results