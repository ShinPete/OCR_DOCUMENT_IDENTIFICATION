import os
import time
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
import pytesseract

# Configure logging once, near the top of your script
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detail
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("ocr_run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(file_types = ['budget', 'email', 'letter', 'invoice']):
    cols = ['filename', 'type', 'text']

    img_dir = Path(r"C:\Users\abajp\PycharmProjects\BofAOCRProject\data\raw\images\Training")
    processed_dir = Path(r"C:\Users\abajp\PycharmProjects\BofAOCRProject\data\processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "output.csv"

    # Load already processed filenames
    processed = set()
    if out_path.exists():
        try:
            processed = set(pd.read_csv(out_path, usecols=['filename'])['filename'])
            logger.info("Loaded %d previously processed filenames from %s", len(processed), out_path)
        except Exception as e:
            logger.exception("Failed to read existing output file. Proceeding with empty processed set. %s", e)

    # Stats
    total_seen = 0
    total_processed = 0
    skipped_not_file = 0
    skipped_no_match = 0
    skipped_already = 0
    errors = 0

    logger.info("Starting OCR pass in %s. Looking for types: %s", img_dir, file_types)

    # Determine if we need to write a header the first time we write
    header_needed = not out_path.exists()

    # Iterate files
    try:
        entries = os.listdir(img_dir)
    except Exception as e:
        logger.exception("Could not list directory %s", img_dir)
        return

    for name in entries:
        total_seen += 1
        path = img_dir / name

        if not path.is_file():
            skipped_not_file += 1
            logger.debug("Skipping non-file: %s", path)
            continue

        lowered = name.lower()
        match = next((w for w in file_types if w in lowered), None)
        if not match:
            skipped_no_match += 1
            logger.debug("No type match for %s", name)
            continue

        if name in processed:
            skipped_already += 1
            logger.debug("Already processed: %s", name)
            continue

        # Process
        t0 = time.time()
        try:
            with Image.open(path) as img:
                pre_img = preprocess_image(img)  # assumes your function exists
                text = clean_text(pytesseract.image_to_string(pre_img))  # assumes your function exists

            new_df = pd.DataFrame([[name, match, text]], columns=cols)
            new_df.to_csv(out_path, mode='a', index=False, header=header_needed)
            header_needed = False  # only write header once
            processed.add(name)
            total_processed += 1

            dt = time.time() - t0
            logger.info("Processed %s as type=%s in %.2fs", name, match, dt)

        except Exception as e:
            errors += 1
            logger.exception("Error processing %s", name)

    # Summary
    logger.info("Done. Seen=%d, processed=%d, skipped_not_file=%d, skipped_no_match=%d, skipped_already=%d, errors=%d",
                total_seen, total_processed, skipped_not_file, skipped_no_match, skipped_already, errors)