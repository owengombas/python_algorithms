from __future__ import annotations
from typing import (
    Dict,
    List,
    Generic,
    TypeVar,
    Optional,
    Type,
    Tuple,
    Set,
    Generator,
    Callable,
    Any,
)
import graphviz
import uuid
import math

T = TypeVar("T")


# Graph elements
class GraphVertex(Generic[T]):
    @property
    def value(self) -> T:
        return self._value

    @property
    def visited(self) -> bool:
        return self._visited

    @visited.setter
    def visited(self, visited: bool):
        self._visited = visited

    @property
    def index(self) -> int:
        return self._index

    @property
    def adjacency_list(self) -> List[GraphVertex[T]]:
        return self._adjacency_list

    @property
    def edges(self) -> List[GraphEdge[T]]:
        return self._edges

    @property
    def id(self) -> uuid.UUID:
        return self._id

    def __init__(self, value: T, index: int = -1):
        """
        Node of a graph.

        Args:
            index (int): The index of the node (used for the matrix)
            value (T): The value of the node.
        """
        self._value = value
        self._index = index
        self._visited = False
        self._adjacency_list: List[GraphVertex[T]] = []
        self._edges: List[GraphEdge[T]] = []
        self._id = uuid.uuid4()

    def add_adjacency(self, edge: "GraphEdge[T]"):
        """
        Add an adjacency to the node.

        Args:
            edge (GraphEdge[T]): The edge to add.
        """
        self._adjacency_list.append(edge.end)
        self._edges.append(edge)

    def __str__(self):
        value = str(f"{self._index} {self._value}")
        if self._visited:
            value += "*"
        for v in self.edges:
            value += " -> " + str(v.end.value) + " (" + str(v.weight) + ")"
        return value.strip()

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return hash(self.value)


class GraphEdge(Generic[T]):
    @property
    def start(self) -> GraphVertex[T]:
        return self._start

    @property
    def end(self) -> GraphVertex[T]:
        return self._end

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def directed(self) -> bool:
        return self._directed

    def __init__(
        self,
        start: GraphVertex[T],
        end: GraphVertex[T],
        weight: float,
        directed: bool = True,
    ):
        """
        Edge of a graph.

        Args:
            start (GraphVertex[T]): The start node.
            end (GraphVertex[T]): The end node.
            weight (float): The weight of the edge.
        """
        self._start = start
        self._end = end
        self._weight = weight
        self._directed = directed

    def reverse(self):
        return GraphEdge(self._end, self._start, self._weight, self._directed)

    def __str__(self):
        return f"{self._start.value} -> {self._end.value} ({self._weight})"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        if self._directed:
            return hash((self._start, self._end, self._weight))
        else:
            return hash(self._start) + hash(self._end) + hash(self._weight)

    def __eq__(self, other: GraphEdge[T]) -> bool:
        if self._directed:
            return (
                self._start == other._start
                and self._end == other._end
                and self._weight == other._weight
            )
        else:
            return (
                self._start == other._start
                and self._end == other._end
                and self._weight == other._weight
            ) or (
                self._start == other._end
                and self._end == other._start
                and self._weight == other._weight
            )


# Iterators
class NodeIterator(Generic[T]):
    def __init__(self, vertices: List[GraphVertex[T]]):
        """
        Iterator for the nodes of a graph.

        Args:
            vertices (List[GraphVertex[T]]): The list of vertices.
        """
        self._index = 0
        self._vertices = vertices

    def __next__(self) -> GraphVertex[T]:
        """
        Get the next node.

        returns:
            GraphVertex[T]: The next node.
        """
        if self._index < len(self._vertices):
            result = self._vertices[self._index]
            self._index += 1
            return result
        raise StopIteration
    
    def more(self) -> bool:
        if self.__has_next__():
            return True
        raise StopIteration

    def first(self) -> GraphVertex[T]:
        return self._vertices[0]

    def reset(self):
        self._index = 0

    def __iter__(self):
        return self

    def __has_next__(self):
        return self._index < len(self._vertices)


class EdgeIterator(Generic[T]):
    def __init__(self, edges: List[GraphEdge[T]]):
        """
        Iterator for the edges of a graph.

        Args:
            edges (List[GraphEdge[T]]): The list of edges.
        """
        self._index = 0
        self._edges = edges

    def __next__(self) -> GraphEdge[T]:
        """
        Get the next edge.

        returns:
            GraphEdge[T]: The next edge.
        """
        if self._index < len(self._edges):
            result = self._edges[self._index]
            self._index += 1
            return result
        raise StopIteration

    def first(self) -> GraphEdge[T]:
        return self._edges[0]

    def reset(self):
        self._index = 0

    def __iter__(self):
        return self

    def __has_next__(self):
        return self._index < len(self._edges)


# Priority queues
class PriorityQueue(Generic[T]):
    @property
    def items(self) -> List[T]:
        return self._queue

    def __init__(self, get_priority: Callable[[T], float] = lambda x: 0):
        self._queue = []
        self._get_priority = get_priority

    def __len__(self):
        return len(self._queue)

    def deposit(self, item: T):
        priority = self._get_priority(item)
        self._queue.append((item, priority))
        self._queue.sort(key=lambda x: x[1])

    def retrieve(self) -> T:
        return self._queue.pop(0)[0]

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def make_empty(self):
        self._queue = []


class Queue(PriorityQueue[T]):
    def __init__(self):
        super().__init__(lambda x: len(self))


class Stack(PriorityQueue[T]):
    def __init__(self):
        super().__init__(lambda x: -len(self))


class Marker(Generic[T]):
    def __init__(self):
        self._elements_set: Set[T] = set()

    def mark(self, element: T):
        self._elements_set.add(element)

    def unmark(self, element: T):
        self._elements_set.remove(element)

    def is_marked(self, element: T) -> bool:
        return element in self._elements_set

    def __str__(self):
        return str(self._elements_set)

    def __repr__(self):
        return str(self)


class AttributeAugmented(Generic[T]):
    @property
    def value(self) -> T:
        return self._value

    def __init__(self, **attributes: Dict[str, Any]):
        self._attributes = attributes
        self._value = None

    def __call__(self, value: T) -> AttributeAugmented[T]:
        new_augmented = AttributeAugmented(**self._attributes)
        new_augmented._value = value
        return new_augmented

    def set_attribute(self, name: str, value: Any):
        self._attributes[name] = value

    def get_attribute(self, name: str) -> Any:
        return self._attributes[name]

    def __str__(self):
        s = str(self._value)
        for name, value in self._attributes.items():
            s += f" {name}: {value}"
        return s

    def __repr__(self):
        return str(self)


class Palette:
    def __init__(self, number_of_colors: int):
        self._number_of_colors = number_of_colors
        self._contains_colors = [True] * self._number_of_colors

    def exclude_color(self, color: int):
        self._contains_colors[color] = False

    def first_color(self) -> int:
        for c in range(self._number_of_colors):
            if self._contains_colors[c]:
                return c
        return self._number_of_colors + 1


# Graph
class Graph(Generic[T]):
    @property
    def vertices(self) -> List[GraphVertex[T]]:
        return self._vertices

    @property
    def edges(self) -> List[GraphEdge[T]]:
        return self._edges

    @property
    def matrix(self) -> List[List[float]]:
        return self._matrix

    def __init__(
        self,
        vertices: List[GraphVertex[T]],
        edges: List[GraphEdge[T]],
    ):
        self._vertices = vertices
        self._edges = edges
        self._matrix = None

    def build(self):
        Graph._add_edges_to_nodes(self._edges)

    def add_vertex(self, vertex: GraphVertex[T]):
        self._vertices.append(vertex)

    def add_edge(self, edge: GraphEdge[T]):
        self._edges.append(edge)

    def compute_matrix(self) -> "Graph[T]":
        self._matrix = Graph._compute_matrix(self._vertices, self._edges)
        return self

    def _search_node(
        self,
        vertex: GraphVertex[T],
        vertices_marked: Marker[GraphVertex[T]],
        edges_marked: Marker[GraphEdge[T]],
        memory: PriorityQueue[GraphVertex[T]],
    ) -> Generator[GraphVertex[T] | GraphEdge[T]]:
        """
        Search algorithm for a node.

        Args:
            value (T): The value of the node.
            marked (Set[GraphVertex[T]]): The marked nodes.
            memory (PriorityQueue[GraphVertex[T]]): The memory type, it's a priority queue. Queue -> BFS, Stack -> DFS.
        """
        memory.make_empty()  # clear the memory
        memory.deposit(vertex)  # add the value to the memory
        vertices_marked.mark(vertex)  # mark the value
        while not memory.is_empty():  # while the memory is not empty
            v = memory.retrieve()  # get the first element of the memory
            yield v  # yield the element
            edge_iterator = EdgeIterator(v.edges)
            for e in edge_iterator:
                if not edges_marked.is_marked(e):
                    edges_marked.mark(e)
                    yield e
                    if not vertices_marked.is_marked(e.end):
                        memory.deposit(e.end)
                        vertices_marked.mark(e.end)

    def search(
        self,
        memory: Type[PriorityQueue[GraphVertex[T]]] = Queue,
    ) -> Generator[GraphVertex[T] | GraphEdge[T]]:
        """
        General search algorithm for a graph.

        Args:
            memory (Type[PriorityQueue[GraphVertex[T]]], optional): The memory type, it's a priority queue. Defaults to Queue. Queue -> BFS, Stack -> DFS.

        Yields:
            Iterator[GraphVertex[T]]: The iterator of the vertices.
        """
        vertices_marked = Marker[GraphVertex[T]]()
        edges_marked = Marker[GraphEdge[T]]()
        iterator = NodeIterator(self.vertices)

        for current_vertex in iterator:
            if not vertices_marked.is_marked(current_vertex):
                for treated_node in self._search_node(
                    current_vertex, vertices_marked, edges_marked, memory
                ):
                    yield treated_node

    def _dfs_transfersal(
        self,
        vertex_infos: Dict[uuid.UUID, AttributeAugmented[GraphVertex[T]]],
        current_visit_number: List[int],
        current_component_number: List[int],
        vertex: GraphVertex[T],
        edges_marked: Marker[GraphEdge[T]] = None,
    ) -> Generator[GraphVertex[T] | Tuple[GraphEdge[T], str]]:
        if edges_marked is None:
            edges_marked = Marker[GraphEdge[T]]()

        current_visit_number[0] += 1
        vertex_infos[vertex.id].set_attribute("visit_number", current_visit_number[0])
        vertex_infos[vertex.id].set_attribute("is_ancestor", True)

        yield vertex

        edge_iterator = EdgeIterator(vertex.edges)
        for e in edge_iterator:
            if not edges_marked.is_marked(e):
                edges_marked.mark(e)
                if vertex_infos[e.end.id].get_attribute("visit_number") == 0:
                    yield e
                    vertex_infos[e.end.id].set_attribute(
                        "component_number",
                        vertex_infos[vertex.id].get_attribute("component_number"),
                    )
                    for treated_node in self._dfs_transfersal(
                        vertex_infos,
                        current_visit_number,
                        current_component_number,
                        e.end,
                        edges_marked,
                    ):
                        yield treated_node
                elif vertex_infos[e.end.id].get_attribute("is_ancestor"):
                    yield (e, "backward")
                elif vertex_infos[e.end.id].get_attribute(
                    "component_number"
                ) == vertex_infos[vertex.id].get_attribute("component_number"):
                    yield (e, "forward")
                else:
                    yield (e, "transverse")

        vertex_infos[vertex.id].set_attribute("is_ancestor", False)

    def recursive_search(self) -> Generator[GraphVertex[T] | Tuple[GraphEdge[T], str]]:
        """
        Recursive search algorithm for a graph, it performs a DFS.

        Yields:
            Iterator[GraphVertex[T]]: The iterator of the vertices.
        """
        current_visit_number: List[int] = [0]
        current_component_number: List[int] = [0]
        augmenter = AttributeAugmented[GraphVertex[T]](
            visit_number=0, is_ancestor=False, component_number=0
        )
        vertex_infos: Dict[uuid.UUID, AttributeAugmented[GraphVertex[T]]] = {}
        iterator = NodeIterator(self.vertices)
        for v in iterator:
            vertex_infos[v.id] = augmenter(v)

        iterator = NodeIterator(self.vertices)
        for v in iterator:
            if vertex_infos[v.id].get_attribute("visit_number") == 0:
                current_component_number[0] += 1
                vertex_infos[v.id].set_attribute(
                    "component_number", current_component_number[0]
                )
                for treated_node in self._dfs_transfersal(
                    vertex_infos, current_visit_number, current_component_number, v
                ):
                    yield treated_node

    def detect_circuit(
        self,
        vertex: GraphVertex[T],
        vertices_marked: Marker[GraphVertex[T]] = None,
        edges_marked: Marker[GraphEdge[T]] = None,
        vertex_infos: Dict[uuid.UUID, Dict[str, bool | int]] = None,
    ) -> bool:
        """
        Detect if a circuit is present in the graph.

        Args:
            vertex (GraphVertex[T]): The vertex to start the search from.

        Returns:
            bool: True if a circuit is present, False otherwise.
        """
        if vertices_marked is None:
            vertices_marked = Marker[GraphVertex[T]]()
        if edges_marked is None:
            edges_marked = Marker[GraphEdge[T]]()
        if vertex_infos is None:
            augmenter = AttributeAugmented[GraphVertex[T]](is_ancestor=False)
            vertex_infos: Dict[uuid.UUID, AttributeAugmented[GraphVertex[T]]] = {}
            iterator = NodeIterator(self.vertices)
            for v in iterator:
                vertex_infos[v.id] = augmenter(v)

        cicuit_detected = False
        vertices_marked.mark(vertex)
        vertex_infos[vertex.id].set_attribute("is_ancestor", True)
        iterator = EdgeIterator(vertex.edges)
        while not cicuit_detected and iterator.__has_next__():
            edge = next(iterator)
            if not edges_marked.is_marked(edge):
                edges_marked.mark(edge)
                if vertices_marked.is_marked(edge.end):
                    cicuit_detected = True
                else:
                    cicuit_detected = self.detect_circuit(
                        edge.end, vertices_marked, edges_marked, vertex_infos
                    )

        vertex_infos[vertex.id].set_attribute("is_ancestor", False)
        return cicuit_detected

    def topological_sort(
        self,
        output: List[GraphVertex[T]] = [],
        vertex: GraphVertex[T] = None,
        marked: Marker[GraphVertex[T]] = None,
    ) -> List[GraphVertex[T]]:
        """
        Topological sort of the graph.

        Args:
            output (GraphVertex[T]): The output list.
            vertex (GraphVertex[T]): The vertex to start the search from.
            marked (Set[GraphVertex[T]]): The marked nodes.
        """
        if marked is None:
            marked = Marker[GraphVertex[T]]()
        if vertex is None:
            vertex = self._vertices[0]
        iterator = EdgeIterator(vertex.edges)
        for e in iterator:
            n = e.end
            if not marked.is_marked(n):
                marked.mark(n)
                self.topological_sort(output, n, marked)
        output.append(vertex)

    def _tree_fault_detected(
        self,
        vertex: GraphVertex[T],
        present_start: GraphVertex[T],
        count: List[int],
        marked: Marker[GraphVertex[T]],
        is_start: Dict[uuid.UUID, bool] = None,
    ) -> bool:
        """
        Check if a fault is present in the graph.

        Args:
            vertex (GraphVertex[T]): The vertex to start the search from.
            present_start (Dict[uuid.UUID, bool]): The present start nodes.
            marked (Set[GraphVertex[T]]): The marked nodes.
            is_start (Dict[uuid.UUID, bool]): The start nodes.

        Returns:
            bool: True if a fault is present, False otherwise.
        """
        fault_detected = False
        marked.mark(vertex)
        iterator = EdgeIterator(vertex.edges)
        while not fault_detected and iterator.__has_next__():
            edge = next(iterator)
            n = edge.end
            if marked.is_marked(n):
                if is_start[n.id] and n != present_start:
                    is_start[n.id] = False
                    count[0] -= 1
                else:
                    fault_detected = True
            else:
                fault_detected = self._tree_fault_detected(
                    n, present_start, count, marked, is_start
                )
        return fault_detected

    def is_tree(
        self, count: List[int] = [0], marked: Marker[GraphVertex[T]] = None
    ) -> bool:
        """
        Check if the graph is a tree.

        Returns:
            bool: True if the graph is a tree, False otherwise.
        """
        if marked is None:
            marked = Marker[GraphVertex[T]]()
        is_start: Dict[uuid.UUID, bool] = {}
        iterator = NodeIterator(self.vertices)
        for v in iterator:
            is_start[v.id] = False

        fault_detected = False
        iterator = NodeIterator(self.vertices)
        while not fault_detected and iterator.__has_next__():
            v = next(iterator)
            if not marked.is_marked(v):
                is_start[v.id] = True
                count[0] += 1
                fault_detected = self._tree_fault_detected(
                    v, v, count, marked, is_start
                )

        return not fault_detected and count[0] == 1

    def find_root(
        self, count: List[int] = [0], marked: Marker[GraphVertex[T]] = None
    ) -> GraphVertex[T]:
        """
        Find the root of the graph.

        Returns:
            GraphVertex[T]: The root of the graph.
        """
        if marked is None:
            marked = Marker[GraphVertex[T]]()
        is_start: Dict[uuid.UUID, bool] = {}
        iterator = NodeIterator(self.vertices)
        for v in iterator:
            is_start[v.id] = False

        root: GraphVertex[T] = None
        fault_detected = False
        iterator = NodeIterator(self.vertices)
        while not fault_detected and iterator.__has_next__():
            v = next(iterator)
            if not marked.is_marked(v):
                is_start[v.id] = True
                count[0] += 1
                root = v
                fault_detected = self._tree_fault_detected(
                    v, v, count, marked, is_start
                )

        if not fault_detected and count[0] == 1:
            return root

        return None

    def least_edges(self, start: GraphVertex[T]) -> Dict[GraphVertex[T], float]:
        distances: Dict[GraphVertex[T], float] = {}
        iterator = NodeIterator(self.vertices)
        for v in iterator:
            distances[v.value] = float("inf")

        distances[start.value] = 0
        memory = Queue[GraphVertex[T]]()

        for treated_edge in self._search_node(start, Marker(), Marker(), memory):
            if isinstance(treated_edge, GraphEdge):
                # Set the distance of the end node
                start = treated_edge.start
                end = treated_edge.end
                if distances[end.value] == float("inf"):
                    distances[end.value] = 1 + distances[start.value]

        return distances

    def dijkstra(self, start: GraphVertex[T]) -> Dict[GraphVertex[T], float]:
        vertex_infos: Dict[GraphVertex[T], AttributeAugmented[GraphVertex[T]]] = {}
        augmenter = AttributeAugmented[GraphVertex[T]](
            distance=float("inf"), predescessor=None
        )
        iterator = NodeIterator(self.vertices)
        for v in iterator:
            vertex_infos[v.value] = augmenter(v)

        vertex_infos[start.value].set_attribute("distance", 0)
        memory = PriorityQueue[GraphVertex[T]](
            lambda x: vertex_infos[x.value].get_attribute("distance")
        )

        for treated_edge in self._search_node(start, Marker(), Marker(), memory):
            if isinstance(treated_edge, GraphEdge):
                # Set the distance of the end node
                start = treated_edge.start
                end = treated_edge.end
                distance = (
                    vertex_infos[start.value].get_attribute("distance")
                    + treated_edge.weight
                )
                if vertex_infos[end.value].get_attribute("distance") == float("inf"):
                    vertex_infos[end.value].set_attribute("distance", distance)
                    vertex_infos[end.value].set_attribute("predescessor", start)
                elif distance < vertex_infos[end.value].get_attribute("distance"):
                    vertex_infos[end.value].set_attribute("distance", distance)

        return {k: v.get_attribute("distance") for k, v in vertex_infos.items()}

    def prim(self, start: GraphVertex[T]) -> Graph[T]:
        memory = PriorityQueue[GraphEdge[T]](lambda x: x.weight)
        mst = Graph(self.vertices, [])
        marker = Marker[GraphVertex[T]]()
        marker.mark(start)
        for e in start.edges:
            memory.deposit(e)

        while not memory.is_empty():
            edge = memory.retrieve()
            if not marker.is_marked(edge.end):
                marker.mark(edge.end)
                mst.add_edge(edge)
                iterator = EdgeIterator(edge.end.edges)
                for next_edge in iterator:
                    if not marker.is_marked(next_edge.end):
                        memory.deposit(next_edge)

        copy_vertices, copy_edges = mst.copy_vertices_and_edges()
        mst = Graph(copy_vertices, copy_edges)
        mst.build()
        mst.compute_matrix()

        return mst

    def find_vertex_with_index(self, index: int) -> Optional[GraphVertex[T]]:
        for v in self._vertices:
            if v.index == index:
                return v
        return None

    def copy_vertices_and_edges(
        self,
    ) -> Tuple[List[GraphVertex[T]], List[GraphEdge[T]]]:
        vertices_copy_dict: Dict[uuid.UUID, GraphVertex[T]] = {}
        for v in self._vertices:
            vertex_copy = GraphVertex(v.value, v.index)
            vertices_copy_dict[v.index] = vertex_copy

        edges_copy: List[GraphEdge[T]] = []
        for e in self._edges:
            edges_copy.append(
                GraphEdge(
                    vertices_copy_dict[e.start.index],
                    vertices_copy_dict[e.end.index],
                    e.weight,
                    e.directed,
                )
            )

        return list(vertices_copy_dict.values()), edges_copy

    def are_neighbors(self, v1: GraphVertex[T], v2: GraphVertex[T]) -> bool:
        return v2 in v1.adjacency_list

    def colorize(self):
        number_of_nodes = len(self._vertices)
        coloring: List[int] = [0] * number_of_nodes
        iterator = NodeIterator(self._vertices)
        node = iterator.first()

        while iterator.__has_next__():
            possible_colors = Palette(number_of_nodes)
            n = iterator.first()
            while n != node:
                if self.are_neighbors(node, n):
                    possible_colors.exclude_color(coloring[n.index])
                n = next(iterator)
            coloring[node.index] = possible_colors.first_color()
            node = next(iterator)

        return coloring
    
    def to_graphviz(self, orientation: str = "horizontal"):
        dot = graphviz.Digraph()
        dot.attr(fontsize="10", fontname="helvetica")

        if orientation == "horizontal":
            dot.attr(rankdir="LR")
        elif orientation == "vertical":
            dot.attr(rankdir="TB")
        else:
            raise ValueError("Invalid orientation")

        for v in self._vertices:
            dot.node(str(v.id), str(v.value))
        for e in self._edges:
            dot.edge(
                str(e.start.id),
                str(e.end.id),
                str(e.weight),
                dir="forward" if e.directed else "none",
            )

        return dot

    def __str__(self):
        value = ""
        for v in self._vertices:
            value += str(v) + "\n"
        return value.strip()

    def __repr__(self):
        return str(self)

    @staticmethod
    def _compute_matrix(
        vertices: List[GraphVertex[T]], edges: List[GraphEdge[T]]
    ) -> List[List[float]]:
        matrix = []
        for i in range(len(vertices)):
            matrix.append([0] * len(vertices))
        for e in edges:
            assert e.start.index != -1, f"Start node {e.start.value} has no index"
            assert e.end.index != -1, f"End node {e.end.value} has no index"
            matrix[e.start.index][e.end.index] = e.weight
            if not e.directed:
                matrix[e.end.index][e.start.index] = e.weight
        return matrix

    @staticmethod
    def _compute_adjacency_list(edges: List[GraphEdge[T]] = None):
        adjacency_list: Dict[T, List[Tuple[T, float]]] = {}
        for e in edges:
            if e.start.value not in adjacency_list:
                adjacency_list[e.start.value] = []
            adjacency_list[e.start.value].append((e.end.value, e.weight))
        return adjacency_list

    @staticmethod
    def _add_edges_to_nodes(edges: List[GraphEdge[T]]):
        for e in edges:
            e.start.add_adjacency(e)
            if not e.directed:
                e.end.add_adjacency(e.reverse())
