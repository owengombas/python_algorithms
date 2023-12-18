from typing import List, Optional, Tuple, Union, Generator, Generic, TypeVar
import graphviz
from lib.stack_queue import ListStack, ListQueue

T = TypeVar("T")


class Node(Generic[T]):
    @property
    def value(self) -> T:
        return self._value

    @property
    def parent(self) -> Optional["Node[T]"]:
        return self._parent

    @parent.setter
    def parent(self, parent: Optional["Node[T]"]) -> None:
        self._parent = parent

    @property
    def children(self) -> List["Node[T]"]:
        return self._children

    @property
    def number_of_children(self) -> int:
        return len(self._children)

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_branch(self) -> bool:
        return self.number_of_children > 0

    @property
    def is_leaf(self) -> bool:
        return self.number_of_children == 0

    @property
    def left(self) -> Optional["Node[T]"]:
        if self.is_leaf:
            return None
        return self.children[0]

    @property
    def right(self) -> Optional["Node[T]"]:
        if self.is_leaf:
            return None
        return self.children[-1]

    @property
    def weights(self) -> List[int]:
        return self._weights

    def __init__(
        self, value: T, parent: Optional["Node[T]"] = None, weights: List[int] = None
    ):
        if weights is not None:
            assert len(weights) == len(
                value
            ), "Length of weights must match length of value"

        self._value = value
        self._parent = parent
        self._children: List["Node[T]"] = []
        self._weights = weights

    def add_children(self, *children: "Node[T]") -> None:
        for child in children:
            self.children.append(child)
            child.parent = self

    def retrieve_root(self) -> "Node[T]":
        if self.is_root:
            return self
        return self.parent.retrieve_root()

    def compute_size(self) -> int:
        return sum(child.compute_size() for child in self.children) + 1

    def compute_height(self) -> int:
        if self.is_leaf:
            return 0
        return max(child.compute_height() for child in self.children) + 1

    def compute_depth(self) -> int:
        if self.is_root:
            return 0
        return self.parent.compute_depth() + 1

    def to_graphviz(self, graph: graphviz.Digraph = None, **kwargs) -> graphviz.Digraph:
        if graph is None:
            graph = graphviz.Digraph(**kwargs)
        graph.node(str(id(self)), str(self._value))
        for index, child in enumerate(self.children):
            label = ""
            if self.weights is not None:
                label = str(self.weights[index])
            graph.edge(str(id(self)), str(id(child)), label=label)
            child.to_graphviz(graph)
        return graph

    def post_order(self) -> Generator["Node[T]", None, None]:
        for child in self.children:
            yield from child.post_order()
        yield self

    def pre_order(self) -> Generator["Node[T]", None, None]:
        yield self
        for child in self.children:
            yield from child.pre_order()

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return str(self._value)


class ItemNotFound(Exception):
    pass


class StackNode(Generic[T]):
    def __init__(self, node: Node[T]):
        self.node = node
        self.visit_count = 0


class BinaryTreeIterator:
    def __init__(self, start_node: Node[T]):
        self._start_node = start_node
        self._current_node: Node[T] = None

    def set_current_to_first(self) -> None:
        pass

    def current_is_valid(self) -> bool:
        return self._current_node is not None

    def retrieve_current(self) -> Node[T]:
        if self.current_is_valid():
            return self._current_node
        raise ItemNotFound("Current node is not valid")

    def advance_current(self) -> None:
        pass


class BinaryTreePostOrderIterator(BinaryTreeIterator):
    def __init__(self, start_node: Node[T]):
        super().__init__(start_node)
        self._stack = ListStack[StackNode[T]](start_node.compute_size())
        self._stack.push(StackNode(start_node.retrieve_root()))

    def set_current_to_first(self) -> None:
        self._stack.make_empty()

        root = self._start_node.retrieve_root()

        if root is not None:
            self._stack.push(StackNode(root))
        try:
            self.advance_current()
        except:
            pass

    def advance_current(self) -> None:
        if self._stack.is_empty():
            if self._current_node is None:
                raise ItemNotFound("Current node is not valid")
            self._current_node = None
        else:
            a_node: StackNode[T] = None
            stop = False
            while not stop:
                try:
                    a_node = self._stack.pop()
                except:
                    pass
                else:
                    a_node.visit_count += 1
                    if a_node.visit_count == 1:
                        self._stack.push(a_node)
                        if a_node.node.left is not None:
                            self._stack.push(StackNode(a_node.node.left))
                    if a_node.visit_count == 2:
                        self._stack.push(a_node)
                        if a_node.node.right is not None:
                            self._stack.push(StackNode(a_node.node.right))
                    if a_node.visit_count == 3:
                        self._current_node = a_node.node
                        stop = True


class BinaryTreeInOrderIterator(BinaryTreePostOrderIterator):
    def __init__(self, start_node: Node[T]):
        super().__init__(start_node)

    def advance_current(self) -> None:
        if self._stack.is_empty():
            if self._current_node is None:
                raise ItemNotFound("Current node is not valid")
            self._current_node = None
        else:
            a_node: StackNode[T] = None
            stop = False
            while not stop:
                try:
                    a_node = self._stack.pop()
                except:
                    pass
                else:
                    a_node.visit_count += 1
                    if a_node.visit_count == 1:
                        self._stack.push(a_node)
                        if a_node.node.left is not None:
                            self._stack.push(StackNode(a_node.node.left))
                    if a_node.visit_count == 2:
                        self._current_node = a_node.node
                        if a_node.node.right is not None:
                            self._stack.push(StackNode(a_node.node.right))
                        stop = True


class BinaryTreePreOrderIterator(BinaryTreeIterator):
    def __init__(self, start_node: Node[T]):
        super().__init__(start_node)
        self._stack = ListStack[StackNode[T]](start_node.compute_size())
        self._stack.push(StackNode(start_node.retrieve_root()))

    def set_current_to_first(self) -> None:
        self._stack.make_empty()

        root = self._start_node.retrieve_root()

        if root is not None:
            self._stack.push(StackNode(root))
        try:
            self.advance_current()
        except:
            pass

    def advance_current(self) -> None:
        if self._stack.is_empty():
            if self._current_node is None:
                raise ItemNotFound("Current node is not valid")
            self._current_node = None
        else:
            try:
                self._current_node = self._stack.pop().node
            except:
                pass
            else:
                if self._current_node.right is not None:
                    self._stack.push(StackNode(self._current_node.right))
                if self._current_node.left is not None:
                    self._stack.push(StackNode(self._current_node.left))


class BinaryTreeLevelOrderIterator(BinaryTreeIterator):
    def __init__(self, start_node: Node[T]):
        super().__init__(start_node)
        self._queue = ListQueue[StackNode[T]](start_node.compute_size())
        self._queue.push(StackNode(start_node.retrieve_root()))

    def set_current_to_first(self) -> None:
        self._queue.make_empty()

        root = self._start_node.retrieve_root()

        if root is not None:
            self._queue.push(StackNode(root))
        try:
            self.advance_current()
        except:
            pass

    def advance_current(self) -> None:
        if self._queue.is_empty():
            if self._current_node is None:
                raise ItemNotFound("Current node is not valid")
            self._current_node = None
        else:
            try:
                self._current_node = self._queue.pop().node
            except:
                pass
            else:
                if self._current_node.left is not None:
                    self._queue.push(StackNode(self._current_node.left))
                if self._current_node.right is not None:
                    self._queue.push(StackNode(self._current_node.right))
