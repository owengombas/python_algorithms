from collections import defaultdict
from typing import List, Dict, Set


class TopologicalSort:
    """
    Performs topological sort on a directed acyclic graph (DAG).
    It solves the problem of finding a topological ordering of the nodes in a DAG.

    For more information see: https://en.wikipedia.org/wiki/Topological_sorting

    For instance, the graph:
    A -> C
    B -> C, D
    C -> E
    D -> F
    E
    F
    has the following topological order:
    [A, B, D, C, E, F] or [B, A, D, C, E, F] or [B, A, C, D, E, F or] ...
    because A and B must come before C, D must come before F, etc.

    Topological sorting is not unique, there can be more than one topological sorting for a graph.
    However, it's important to note that not all graphs can have a topological sorting.
    A graph which contains a cycle cannot have a valid topological ordering.

    For the provided graph, we can compute a topological sort by:
    1) Identifying vertices with no incoming edges. In the initial graph, these are A and B.
    2) Choose one of these vertices and add it to the sorted list.
    3) Remove this vertex and all edges coming out of it from the graph.
    4) Repeat steps 1-3 until the graph is empty.

    Let's perform the topological sort step-by-step:
    1) Start with A and B (since they have no incoming edges). Let's pick A first (the choice is arbitrary).
    2) Remove A and its outgoing edge (A -> C). Now, B and C have no incoming edges.
    3) Pick B, then remove B and its outgoing edges (B -> C, B -> D). Now, C and D have no incoming edges.
    4) Pick C, then remove C and its outgoing edge (C -> E). Now, D and E have no incoming edges.
    5) Pick D, then remove D and its outgoing edge (D -> F). Now, E and F have no incoming edges.
    6) Pick E, then remove E. Finally, pick F and remove F.
    So, a possible topological sort of the graph is: [A, B, C, D, E, F]

    Args:
        graph (dict): The graph represented as a dictionary of adjacency lists.
    Returns:
        list: The topologically sorted order of nodes.
    Usage example:
        ```python
        graph = {"A": ["C"], "B": ["C", "D"], "C": ["E"], "D": ["F"], "E": [], "F": []}
        topological_sort = TopologicalSort()
        topological_sort.dfs(graph)
        topological_sort.kahn(graph)
        ```
    """

    def dfs(self, graph: Dict[str, List[str]]):
        """
        Performs topological sort using DFS.

        DFS works by:
        1) Visiting a node.
        2) Recursively visiting all of its neighbors.
        3) Adding the node to the topological order.

        Args:
            graph (dict): The graph represented as a dictionary of adjacency lists.
        Returns:
            list: The topologically sorted order of nodes.
        """

        def dfs_visit(node: str):
            """Visits a node."""
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs_visit(neighbor)
            topological_order.append(node)

        visited = set()
        topological_order = []

        for node in graph:
            if node not in visited:
                dfs_visit(node)

        return topological_order[::-1]

    def kahn(self, graph: Dict[str, List[str]]):
        """
        Performs topological sort using Kahn's algorithm.

        Kahn's algorithm works by:
        1) Computing the in-degree of each node.
        2) Enqueueing all nodes with in-degree 0.
        3) Removing a node from the queue and adding it to the topological order.
        4) Removing all edges coming out of this node.
        5) If the graph is empty, return the topological order.
        6) Otherwise, repeat steps 2-5.

        Args:
            graph (dict): The graph represented as a dictionary of adjacency lists.
        Returns:
            list: The topologically sorted order of nodes.
        """
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        queue = [node for node in graph if in_degree[node] == 0]
        topological_order = []

        while queue:
            node = queue.pop(0)
            topological_order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return topological_order
