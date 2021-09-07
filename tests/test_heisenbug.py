import pytest


@pytest.mark.parametrize("xyz", range(2))
def test_fend_off_heisenbug(xyz):
    # The GC at the end of the pytest process segfaults depending on how many tests
    # there are, for some dumb fucking reason. Needs to be investigated if it happens
    # in the wild, until then, adjust the number until it works. Probably related to
    # threads spawned by rx or whatever, I don't know, I'm not a wizard.
    pass
