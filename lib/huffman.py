from typing import List, Optional, Tuple, Union, Generator, Generic, TypeVar, Dict
from lib.tree import Node
import graphviz


def get_probabilities(text: str) -> Dict[str, float]:
    probabilities: Dict[str, float] = {}
    for char in text:
        if char not in probabilities:
            probabilities[char] = 0
        probabilities[char] += 1
    for char in probabilities:
        probabilities[char] /= len(text)
    return probabilities


def build_tree(probabilities: Dict[str, float]) -> Node[Tuple[str, float]]:
    nodes: List[Node[Tuple[str, float]]] = []
    for char in probabilities:
        nodes.append(Node((char, probabilities[char])))

    while len(nodes) > 1:
        nodes.sort(key=lambda node: node.value[1])
        left: Node[Tuple[str, float]] = nodes.pop(0)
        right: Node[Tuple[str, float]] = nodes.pop(0)
        parent: Node[Tuple[str, float]] = Node(
            (left.value[0] + right.value[0], left.value[1] + right.value[1]),
            weights=[0, 1],
        )
        parent.add_children(left, right)
        nodes.append(parent)

    return nodes[0]


def get_code(char: str, node: Node[Tuple[str, float]], path: str = "") -> Optional[str]:
    if node.value[0] == char:
        return path
    elif node.children:
        left: Optional[str] = get_code(char, node.children[0], path + "0")
        right: Optional[str] = get_code(char, node.children[1], path + "1")
        if left:
            return left
        elif right:
            return right
    return None


def get_codes(
    probabilities: Dict[str, float], tree: Node[Tuple[str, float]]
) -> Dict[str, str]:
    codes: Dict[str, str] = {}
    for char in probabilities:
        codes[char] = get_code(char, tree)
    return codes


def encode(text: str, codes: Dict[str, str]) -> Dict[str, str]:
    encoded_text: str = ""
    for char in text:
        encoded_text += codes[char]
    return encoded_text


def decode(text: str, codes: Dict[str, str]) -> str:
    decoded_text: str = ""
    while text:
        for char in codes:
            if text.startswith(codes[char]):
                decoded_text += char
                text = text[len(codes[char]) :]
    return decoded_text


def to_graphviz(tree: Node[Tuple[str, float]]) -> graphviz.Digraph:
    graph: graphviz.Digraph = graphviz.Digraph()

    for node in tree.pre_order():
        char = node.value[0]
        if char == " ":
            char = "<SPACE>"
        if not node.is_leaf:
            graph.node(str(id(node)), label=f"{node.value[1]*100:.2f}%")
        else:
            code: str = get_code(node.value[0], tree)
            graph.node(
                str(id(node)), label=f"{char} ({code})\n{node.value[1]*100:.2f}%"
            )

        if node.left is not None:
            graph.edge(str(id(node)), str(id(node.left)), label="0")
        if node.right is not None:
            graph.edge(str(id(node)), str(id(node.right)), label="1")

    return graph
