import pandas as pd
from datetime import datetime

def normalize_timestamp(ts_val: str | int | float) -> str:
    """
    Normalizes various timestamp formats to 'YYYY-MM-DD HH:MM:SS'.
    Handles:
    - ISO 8601 (with/without TZ)
    - DD/MM/YYYY HH:MM
    - MM-DD-YY HH:MM:SS
    - Epoch seconds
    - DD.MM.YYYY kell HH:MM (Estonian free-form)
    - YYYY-MM-DD HH:MM:SS (already correct)

    Timezone used: Europe/Tallinn
    """
    tz = 'Europe/Tallinn'

    # Handle numeric (epoch)
    if isinstance(ts_val, (int, float)) or (isinstance(ts_val, str) and ts_val.isdigit()):
        dt = pd.to_datetime(float(ts_val), unit='s', utc=True)
        return dt.tz_convert(tz).strftime('%Y-%m-%d %H:%M:%S')

    ts_str = str(ts_val).strip()

    try:
        if "kell" in ts_str:
            naive_dt = datetime.strptime(ts_str, "%d.%m.%Y kell %H:%M")
            dt = pd.Timestamp(naive_dt)
        elif "/" in ts_str:
            dt = pd.to_datetime(ts_str, dayfirst=True)
        elif "-" in ts_str and len(ts_str.split("-")[0]) < 4:
            dt = pd.to_datetime(ts_str, dayfirst=False)
        else:
            dt = pd.to_datetime(ts_str)

        if dt.tzinfo is None:
            dt = dt.tz_localize(tz)
        else:
            dt = dt.tz_convert(tz)

        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        raise ValueError(f"Could not parse timestamp: {ts_str}") from e

def test_normalize_timestamp():
    test_cases = [
        ("2026-05-25 12:00:00", "2026-05-25 12:00:00"),
        ("25/05/2026 12:00", "2026-05-25 12:00:00"),
        ("05-25-26 12:00:00", "2026-05-25 12:00:00"),
        ("2026-05-25T12:00:00+02:00", "2026-05-25 13:00:00"), 
        ("1779624000", "2026-05-24 15:00:00"), 
        ("25.05.2026 kell 12:00", "2026-05-25 12:00:00"),
    ]

    for input_val, expected in test_cases:
        result = normalize_timestamp(input_val)
        print(f"Testing: {input_val:<30} -> {result:<20} {'[PASS]' if result == expected else '[FAIL]'}")
        assert result == expected, f"Failed: {input_val} -> {result} != {expected}"

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_normalize_timestamp()

