_LOCAL_RANK = None


def set_local_rank(local_rank) -> None:
    global _LOCAL_RANK
    _LOCAL_RANK = local_rank


def rank0_print(*args):
    if _LOCAL_RANK == 0 or _LOCAL_RANK == "0" or _LOCAL_RANK is None or _LOCAL_RANK == -1:
        print(*args)


__all__ = ["rank0_print", "set_local_rank"]
