def string_to_seconds(string: str) -> int:
    hms = string.split(":")
    if len(hms) == 2:
        return int(hms[0]) * 3600 + int(hms[1]) * 60
    elif len(hms) == 3:
        return int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    raise ValueError(f"Invalid time format: {string}")
