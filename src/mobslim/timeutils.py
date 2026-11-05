"""Time helpers. All simulation times are seconds (float) from midnight."""

HOUR = 3600.0
DAY = 24 * HOUR


def parse_time(string: str) -> float:
    """Parse ``HH:MM`` or ``HH:MM:SS`` (hours may exceed 24) into seconds."""
    parts = string.strip().split(":")
    if len(parts) == 2:
        h, m = parts
        s = "0"
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"Invalid time format: {string!r}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_time(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
