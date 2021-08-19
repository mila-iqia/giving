from giving import give, given


def fraise(x):
    give(x)
    a = x * x
    give()
    give(x, a, result=a)
    return a


def test_display(capsys):
    with given() as gv:
        gv.display()
        fraise(4)

    captured = capsys.readouterr()
    assert (
        captured.out
        == """\033[1;36;40mx:\033[0m 4
\033[1;31;40ma:\033[0m 16
\033[1;36;40mx:\033[0m 4; \033[1;31;40ma:\033[0m 16; \033[1;32;40mresult:\033[0m 16
"""
    )
    assert captured.err == ""
