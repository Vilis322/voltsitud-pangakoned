from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CLEANED_CALLS = DATA_DIR / "cleaned_bank_calls.csv"

FILTER_COLUMNS = {
    "label": "label_5way",
    "channel": "channel",
    "caller_number": "caller_number",
    "timestamp": "timestamp",
}
