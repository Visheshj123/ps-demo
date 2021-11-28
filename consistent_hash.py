import hashlib
from bisect import bisect, bisect_left, bisect_right
from typing import List, Optional


class ConsistentHashingRing:
    """array based implementation of
    consistent hashing algorithm"""

    def __init__(self, nodes: Optional[List[str]] = None, total_slots=2 ** 32):
        """
        nodes[i] is present at position keys[i]
        """
        self.nodes = []
        self.keys = []
        self.total_slots = total_slots

        for node in nodes:
            self.add_node(node)

    def add_node(self, node: str) -> int:
        if len(self.keys) == self.total_slots:
            raise Exception("hash space is full")

        key = self._hash(node)
        index = bisect(self.keys, key)

        if index > 0 and self.keys[index - 1] == key:
            raise Exception("collision")

        self.nodes.insert(index, node)
        self.keys.insert(index, key)

        return key

    def remove_node(self, node) -> int:
        if len(self.keys) == 0:
            raise Exception("hash space is empty")

        key = self._hash(node)

        index = bisect_left(self.keys, key)

        if index >= len(self.keys) or self.keys[index] != key:
            raise Exception("node does not exist")

        self.keys.pop(index)
        self.nodes.pop(index)

        return key

    def get_node(self, item: str) -> str:
        key = self._hash(item)
        index = bisect_right(self.keys, key) % len(self.keys)
        return self.nodes[index]

    def _hash(self, key: str) -> int:
        hash = hashlib.sha256()
        hash.update(bytes(key, "utf-8"))
        return int(hash.hexdigest(), 16) % self.total_slots
