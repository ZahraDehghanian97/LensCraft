from enum import Enum
from typing import Union


def clip_embedding_name(member: Union[Enum, str]) -> str:
    tail = str(member).split(".")[-1].lower()
    head, *rest = tail.split("_")
    return head + "".join(word.capitalize() for word in rest)
