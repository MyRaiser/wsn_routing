from abc import ABCMeta, abstractmethod
from typing import Iterable

from router import Node


class ClusterBased(metaclass=ABCMeta):
    @abstractmethod
    def clear_clusters(self):
        pass

    @abstractmethod
    def add_cluster_head(self, head: Node):
        pass

    @abstractmethod
    def add_cluster_member(self, head: Node, node: Node):
        pass

    @abstractmethod
    def get_cluster_heads(self) -> Iterable[Node]:
        pass

    @abstractmethod
    def get_cluster_members(self, head: Node) -> Iterable[Node]:
        pass
