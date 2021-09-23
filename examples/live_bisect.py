from giving import give, given


def bisect(arr, key):
    lo = -1
    hi = len(arr)
    give(lo, hi, mid=None)  # emit {"lo": lo, "hi": hi, "mid": None}
    while lo < hi - 1:
        mid = lo + (hi - lo) // 2
        give(mid)  # emit {"mid": mid}
        if arr[mid] > key:
            hi = mid
            give()  # emit {"hi": hi}
        else:
            lo = mid
            give()  # emit {"lo": lo}
    return lo + 1


def main():
    bisect(list(range(1000)), 742)


def rich_display(gv):
    import time

    from rich.live import Live
    from rich.pretty import Pretty
    from rich.table import Table

    def dict_to_table(d):
        table = Table.grid(padding=(0, 3, 0, 0))
        table.add_column("key", style="bold green")
        for k, v in d.items():
            if not k.startswith("$"):
                table.add_row(k, Pretty(v))
        return table

    live = Live(refresh_per_second=4)
    gv.wrap("main", live)

    @gv.kscan().subscribe
    def _(data):
        time.sleep(0.5)  # Wait a bit so that we can see the values change
        live.update(dict_to_table(data))


if __name__ == "__main__":
    with given() as gv:
        rich_display(gv)

        with give.wrap("main"):
            main()
