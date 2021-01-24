from typing import List, Generator


def chunks(l, n):
    # type: (list[str],int) -> Generator[List[(str,str)]]
    """
    <Description>

    :param list[str] l:
    :param int n:
    :return:
    """
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))
