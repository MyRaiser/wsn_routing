from __future__ import annotations

from typing import Iterable

from router.common import Node
from router.leach import LEACH


class APTEEN(LEACH):
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            **kwargs
    ):
        super().__init__(
            sink, non_sinks, **kwargs)

    def initialize(self):
        pass

    def execute(self):
        pass