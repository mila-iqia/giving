import os
import re

from giving import give, given
from giving.executors import format_libpath


def fraise(x):
    give(x)
    a = x * x
    give()
    give(x, a, result=a)
    return a


def things(*values):
    for a in values:
        give(a)


def test_print(capsys):
    with given() as gv:
        gv["a"].print()
        things(1, 2, 3)

    captured = capsys.readouterr()
    assert captured.out == "1\n2\n3\n"


def test_print_format(capsys):
    with given() as gv:
        gv.print("a={a}")
        things(1, 2, 3)

    captured = capsys.readouterr()
    assert captured.out == "a=1\na=2\na=3\n"


def test_display(capsys):
    with given() as gv:
        gv.display()
        fraise(4)

    captured = capsys.readouterr()
    assert (
        captured.out
        == """\033[1;34;40mx:\033[0m 4
\033[1;36;40ma:\033[0m 16
\033[1;34;40mx:\033[0m 4; \033[1;36;40ma:\033[0m 16; \033[1;33;40mresult:\033[0m 16
"""
    )
    assert captured.err == ""

    # No dict

    with given() as gv:
        gv["a"].display(colors=False)

        give(a=1234)

    captured = capsys.readouterr()
    assert captured.out == "1234\n"
    assert captured.err == ""


def test_display_line(capsys):
    with given() as gv:
        gv.display(colors=False)

        give.line(abc=123)

    captured = capsys.readouterr()
    assert captured.out == "(tests/test_executors.py:70 test_display_line) abc: 123\n"
    assert captured.err == ""


def test_display_time(capsys):
    with given() as gv:
        gv.display(colors=False)

        give.time(abc=123)

    captured = capsys.readouterr()
    assert re.match(r"\[[0-9 :-]{19}\] abc: 123", captured.out)
    assert captured.err == ""


def test_display_word(capsys):
    with given() as gv:
        gv.tag(group="cerise").display(colors=False)

        give(abc=123)

    captured = capsys.readouterr()
    assert captured.out == "«cerise:quite» abc: 123\n"
    assert captured.err == ""


def test_display_word_object(capsys):
    class XXX:
        def __str__(self):
            return "xxx"

    with given() as gv:
        gv["abc"].tag(group="meringue").display(colors=False)

        give(abc=XXX())

    captured = capsys.readouterr()
    assert captured.out == "«meringue:for» xxx\n"
    assert captured.err == ""


def test_format_libpath():
    assert format_libpath("abc") == "abc"
    assert format_libpath(os.path.join(os.curdir, "abc")) == "./abc"
    assert format_libpath(os.path.expanduser("~/abc")) == "~/abc"
