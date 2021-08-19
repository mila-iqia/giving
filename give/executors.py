import hashlib


def _dethash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), base=16)


def _term_color(key):
    """Generate a terminal color for this group."""
    return 30 + (_dethash(key) % 8)


def display(data):
    if isinstance(data, dict):
        entries = [
            f"\033[1;{_term_color(k)};40m{k}:\033[0m {v}" for k, v in data.items()
        ]
        print("; ".join(entries))
    else:
        print(data)
