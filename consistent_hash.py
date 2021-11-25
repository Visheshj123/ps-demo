import hashlib
from bisect import bisect, bisect_left, bisect_right
from typing import List


class ConsistentHashingRing:
    """array based implementation of
    consistent hashing algorithm"""

    def __init__(self, nodes: List[str] = None, total_slots=2 ** 32):
        self.nodes = nodes
        self.total_slots = total_slots
        self._keys = []

    def add_node(self, node: str) -> int:
        if len(self._keys) == self.total_slots:
            raise Exception("hash space is full")

        key = self._hash(node)
        index = bisect(self._keys, key)

        if index > 0 and self._keys[index - 1] == key:
            raise Exception("collision")

        self.nodes.insert(index, node)
        self._keys.insert(index, key)

        return key

    def remove_node(self, node) -> int:
        if len(self._keys) == 0:
            raise Exception("hash space is empty")

        key = self._hash(node)

        index = bisect_left(self._keys, key)

        if index >= len(self._keys) or self._keys[index] != key:
            raise Exception("node does not exist")

        self._keys.pop(index)
        self.nodes.pop(index)

        return key

    def get_node(self, item: str) -> str:
        key = self._hash(item)
        index = bisect_right(self._keys, key) % len(self._keys)
        return self.nodes[index]

    def _hash(self, key: str) -> int:
        hash = hashlib.sha256()
        hash.update(bytes(key, "utf-8"))
        return int(hash.hexdigest(), 16) % self.total_slots
