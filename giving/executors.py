import hashlib
import os
import pdb
from datetime import datetime

try:
    import breakword as bw
except ImportError:  # pragma: no cover
    bw = None


def _dethash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), base=16)


def _term_color(key):
    """Generate a terminal color for this group."""
    return 30 + (_dethash(key) % 8)


_path_replacements = {
    "PWD": "",
    "CONDA_PREFIX": "$CONDA_PREFIX/",
    "VIRTUAL_ENV": "$VIRTUAL_ENV/",
    "HOME": "~/",
}


def format_libpath(path):
    for var, pfx in sorted(_path_replacements.items(), key=lambda kv: kv[1]):
        val = os.environ.get(var, None)
        if val is not None:
            if not val.endswith("/"):
                val += "/"
            if path.startswith(val):
                return os.path.join(pfx, path[len(val) :])
    else:
        return path


def extract_word(data, method="pop"):
    group = word = None

    if isinstance(data, dict):
        word = getattr(data, method)("$word", None)
        if word is not None:
            group = getattr(data, method)("$group", "")
        return group, word

    elif hasattr(data, "$word"):
        word = getattr(data, "$word", None)
        if word is not None:
            group = getattr(data, "$group", "")

    return group, word


class Displayer:
    def __init__(self, colors=True, time_format="%Y-%m-%d %H:%M:%S"):
        self.colors = colors
        self.time_format = time_format

    def with_color(self, k, color):
        if self.colors:
            return f"\033[{color};40m{k}\033[0m"
        else:
            return k

    def colorize(self, k):
        return self.with_color(k, f"1;{_term_color(k)}")

    def string(self, data):
        s = ""

        if isinstance(data, dict):
            data = dict(data)

        group, word = extract_word(data)
        if word is not None:
            word = f"{group}:{word}" if group else f"{word}"
            gcol = _term_color(group)
            s += self.with_color("«", f"0;{gcol}")
            s += self.with_color(word, f"1;{gcol}")
            s += self.with_color("» ", f"0;{gcol}")

        if isinstance(data, dict):
            time = data.pop("$time", None)
            if time is not None:
                stime = datetime.utcfromtimestamp(time).strftime(self.time_format)
                s += self.with_color(f"[{stime}] ", "1;30")

            line = data.pop("$line", None)
            if line is not None:
                fname = format_libpath(line.filename)
                s += self.colorize(f"({fname}:{line.lineno} {line.name}) ")

            entries = [f"{self.colorize(k + ':')} {v}" for k, v in data.items()]
            s += "; ".join(entries)
            return s

        else:
            return f"{s}{data}"

    def __call__(self, data):
        print(self.string(data))


display = Displayer()


class Breakpoint:  # pragma: no cover
    def __init__(self, use_breakword=False, word=None, skip=[]):
        self.word = word
        self.skip = ["giving.*", "reactivex.*", *skip]
        self.bw = False
        if use_breakword:
            try:
                import breakword as bw
            except ImportError:
                raise ImportError(
                    "Package `breakword` must be installed to use the tag() operator"
                )
            self.bw = bw

    def __call__(self, data):
        if self.bw:
            group, word = extract_word(data, method="get")
            if word is not None:
                results = self.bw._get_watch(group, self.word)
                if word not in results:
                    return

        print("Breaking on", display.string(data))
        try:
            breakpoint(skip=self.skip)
        except TypeError:
            pdb.Pdb(skip=self.skip).set_trace()
