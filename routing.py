# routing.py
import heapq
from typing import Dict, List, Tuple


def load_sample_graph() -> Dict:
    """
    Load a small demo graph for routing.
    """
    return {
        "nodes": ["A", "B", "C", "D", "E"],
        "edges": [
            {"u": "A", "v": "B", "weight": 2.0},
            {"u": "B", "v": "C", "weight": 2.5},
            {"u": "C", "v": "D", "weight": 1.5},
            {"u": "A", "v": "D", "weight": 6.0},
            {"u": "B", "v": "E", "weight": 3.0},
            {"u": "E", "v": "D", "weight": 2.0},
        ],
    }


def _neighbors(graph: Dict, node: str) -> List[Tuple[str, float]]:
    result = []
    for e in graph["edges"]:
        if e["u"] == node:
            result.append((e["v"], e["weight"]))
        elif e["v"] == node:
            result.append((e["u"], e["weight"]))
    return result


def dijkstra_route(graph: Dict, start: str, goal: str) -> Tuple[List[str], float]:
    pq = [(0.0, start, [])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue

        visited.add(node)
        path = path + [node]

        if node == goal:
            return path, cost

        for nxt, w in _neighbors(graph, node):
            if nxt not in visited:
                heapq.heappush(pq, (cost + w, nxt, path))

    return [], float("inf")


def astar_route(graph: Dict, start: str, goal: str) -> Tuple[List[str], float]:
    # No heuristic → behaves like Dijkstra (safe demo)
    return dijkstra_route(graph, start, goal)
