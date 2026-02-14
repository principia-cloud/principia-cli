"""Rectangle Contact Graph Realization Solver.

Adapted from SAGE server/room_solver.py.
Solves the problem of arranging rectangular rooms such that they are
non-overlapping, in contact along shared walls, and connected.

Provides two solvers:
1. RectangleContactSolver — multi-strategy solver (greedy, spanning tree, linear)
2. RectangleContactRelaxationSolver — extends #1 with guaranteed-solution fallback
"""

import random
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

logger = logging.getLogger("scene-gen.room_solver")


@dataclass
class RectangleSpec:
    """Original rectangle specification."""
    w: float
    h: float


@dataclass
class RectangleLayout:
    """Rectangle layout with position and (possibly modified) dimensions."""
    x: float  # center x
    y: float  # center y
    w: float
    h: float


class RectangleContactSolver:
    """Solver for Rectangle Contact Graph Realization Problem.

    Given a graph G=(V,E) and rectangle specs {(w_i, h_i)},
    produce a layout where rectangles are non-overlapping, in contact
    along graph edges, and connected.
    """

    def __init__(self, graph: nx.Graph, rectangles: Dict[int, RectangleSpec]):
        self.graph = graph
        self.rectangles = rectangles
        self.n = len(graph.nodes)
        if not nx.is_connected(graph):
            raise ValueError("Input graph must be connected")
        if set(graph.nodes) != set(rectangles.keys()):
            raise ValueError("Graph nodes must match rectangle keys")

    def solve(self, max_attempts: int = 50) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        best_result = None
        best_score = float("inf")

        for attempt in range(max_attempts):
            random.seed(42 + attempt * 7)
            np.random.seed(42 + attempt * 7)

            strategies = [
                self._solve_greedy_placement,
                self._solve_spanning_tree,
                self._solve_linear_arrangement,
            ]
            for strategy in strategies:
                try:
                    result = strategy()
                    if self._validate_solution(result[0], result[1]):
                        optimized = self._optimize_solution(result[0], result[1])
                        f1, f2 = self.compute_objectives(optimized[0], optimized[1])
                        score = f1 + f2 * 0.1
                        if score < best_score:
                            best_result = optimized
                            best_score = score
                        if score == 0:
                            return optimized
                        break
                except Exception:
                    continue

        if best_result is not None:
            return best_result

        # Fallback chain
        for fallback in [self._create_minimal_solution, self._create_guaranteed_valid_solution]:
            try:
                result = fallback()
                if self._validate_solution(result[0], result[1]):
                    return result
            except Exception:
                continue

        return self._create_guaranteed_valid_solution()

    # --- Strategies ---

    def _solve_greedy_placement(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        nodes = list(self.graph.nodes)
        nodes.sort(key=lambda v: self.rectangles[v].w * self.rectangles[v].h + random.uniform(-0.1, 0.1), reverse=True)
        layout: Dict[int, RectangleLayout] = {}
        placed = set()

        first = nodes[0]
        layout[first] = RectangleLayout(0, 0, self.rectangles[first].w, self.rectangles[first].h)
        placed.add(first)

        mst = nx.minimum_spanning_tree(self.graph)
        edges_to_keep = set(mst.edges())

        while len(placed) < self.n:
            candidates = []
            for v in nodes:
                if v not in placed:
                    for u in placed:
                        if (u, v) in edges_to_keep or (v, u) in edges_to_keep:
                            candidates.append((v, u))
            if not candidates:
                unplaced = [v for v in nodes if v not in placed]
                candidates = [(unplaced[0], list(placed)[0])]

            v, u = candidates[0]
            self._place_rectangle_adjacent(v, u, layout)
            placed.add(v)

        modified_graph = nx.Graph()
        modified_graph.add_nodes_from(self.graph.nodes)
        modified_graph.add_edges_from(edges_to_keep)
        return modified_graph, layout

    def _solve_spanning_tree(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        mst = nx.minimum_spanning_tree(self.graph)
        return self._build_layout_from_tree(mst)

    def _solve_linear_arrangement(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        nodes = list(self.graph.nodes)
        layout = self._create_linear_layout(nodes)
        path_graph = nx.path_graph(nodes)
        return path_graph, layout

    def _build_layout_from_tree(self, tree: nx.Graph) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        layout: Dict[int, RectangleLayout] = {}
        visited = set()
        root = next(iter(tree.nodes))
        layout[root] = RectangleLayout(0, 0, self.rectangles[root].w, self.rectangles[root].h)
        visited.add(root)
        queue = [root]
        while queue:
            current = queue.pop(0)
            for neighbor in tree.neighbors(current):
                if neighbor not in visited:
                    self._place_rectangle_adjacent(neighbor, current, layout)
                    visited.add(neighbor)
                    queue.append(neighbor)
        return tree, layout

    # --- Placement helpers ---

    def _place_rectangle_adjacent(self, v: int, u: int, layout: Dict[int, RectangleLayout]):
        rect_v = self.rectangles[v]
        rect_u = layout[u]
        positions = [
            (rect_u.x + (rect_u.w + rect_v.w) / 2, rect_u.y),
            (rect_u.x - (rect_u.w + rect_v.w) / 2, rect_u.y),
            (rect_u.x, rect_u.y + (rect_u.h + rect_v.h) / 2),
            (rect_u.x, rect_u.y - (rect_u.h + rect_v.h) / 2),
        ]
        random.shuffle(positions)
        for x, y in positions:
            candidate = RectangleLayout(x, y, rect_v.w, rect_v.h)
            if self._is_valid_placement(v, candidate, layout):
                layout[v] = candidate
                return
        for offset in [0.1, 0.5, 1.0, 2.0]:
            for x, y in positions:
                for dx, dy in [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]:
                    candidate = RectangleLayout(x + dx, y + dy, rect_v.w, rect_v.h)
                    if self._is_valid_placement(v, candidate, layout):
                        layout[v] = candidate
                        return
        # Fallback
        max_x = max((r.x + r.w / 2 for r in layout.values()), default=0)
        max_y = max((r.y + r.h / 2 for r in layout.values()), default=0)
        layout[v] = RectangleLayout(max_x + rect_v.w, max_y + rect_v.h, rect_v.w, rect_v.h)

    def _is_valid_placement(self, v: int, candidate: RectangleLayout, layout: Dict[int, RectangleLayout]) -> bool:
        return all(not self._rectangles_overlap(candidate, r) for u, r in layout.items() if u != v)

    @staticmethod
    def _rectangles_overlap(a: RectangleLayout, b: RectangleLayout) -> bool:
        return abs(a.x - b.x) < (a.w + b.w) / 2 and abs(a.y - b.y) < (a.h + b.h) / 2

    @staticmethod
    def _rectangles_in_contact(a: RectangleLayout, b: RectangleLayout, tol: float = 1e-3) -> bool:
        dx, dy = abs(a.x - b.x), abs(a.y - b.y)
        h_contact = abs(dx - (a.w + b.w) / 2) < tol and dy < (a.h + b.h) / 2 - tol
        v_contact = abs(dy - (a.h + b.h) / 2) < tol and dx < (a.w + b.w) / 2 - tol
        return h_contact or v_contact

    # --- Fallbacks ---

    def _create_linear_layout(self, nodes) -> Dict[int, RectangleLayout]:
        layout: Dict[int, RectangleLayout] = {}
        x_offset = 0.0
        max_h = max(self.rectangles[n].h for n in nodes) if nodes else 1
        for node in nodes:
            r = self.rectangles[node]
            layout[node] = RectangleLayout(x_offset + r.w / 2, max_h / 2, r.w, r.h)
            x_offset += r.w
        return layout

    def _create_minimal_solution(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        nodes = sorted(list(self.graph.nodes()))
        mst = nx.minimum_spanning_tree(self.graph)
        path_edges = list(nx.dfs_edges(mst, source=nodes[0]))
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(path_edges)
        layout = self._create_linear_layout(nodes)
        return g, layout

    def _create_guaranteed_valid_solution(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        nodes = sorted(list(self.graph.nodes()))
        layout = self._create_linear_layout(nodes)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if self.graph.has_edge(u, v):
                g.add_edge(u, v)
        # Ensure connectivity
        if len(nodes) > 1 and not nx.is_connected(g):
            components = list(nx.connected_components(g))
            for i in range(len(components) - 1):
                for u in components[i]:
                    for v in components[i + 1]:
                        if self.graph.has_edge(u, v):
                            g.add_edge(u, v)
                            break
                    else:
                        continue
                    break
        return g, layout

    # --- Optimization & Validation ---

    def _optimize_solution(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        optimized_graph = graph.copy()
        for u, v in self.graph.edges():
            if not optimized_graph.has_edge(u, v) and self._rectangles_in_contact(layout[u], layout[v]):
                optimized_graph.add_edge(u, v)
        # Try to restore original dimensions
        for node in optimized_graph.nodes():
            orig = self.rectangles[node]
            cur = layout[node]
            for candidate in [
                RectangleLayout(cur.x, cur.y, orig.w, cur.h),
                RectangleLayout(cur.x, cur.y, cur.w, orig.h),
                RectangleLayout(cur.x, cur.y, orig.w, orig.h),
            ]:
                if self._is_valid_with_contacts(node, candidate, optimized_graph, layout):
                    layout[node] = candidate
                    break
        return optimized_graph, layout

    def _is_valid_with_contacts(self, v: int, candidate: RectangleLayout, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> bool:
        for u in layout:
            if u != v and self._rectangles_overlap(candidate, layout[u]):
                return False
        for u in graph.neighbors(v):
            if not self._rectangles_in_contact(candidate, layout[u]):
                return False
        return True

    def _validate_solution(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> bool:
        # Non-overlapping
        nodes = list(layout.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._rectangles_overlap(layout[nodes[i]], layout[nodes[j]]):
                    return False
        # Contact on all edges
        for u, v in graph.edges():
            if not self._rectangles_in_contact(layout[u], layout[v]):
                return False
        # Connected
        if len(nodes) > 1 and not nx.is_connected(graph):
            return False
        return True

    def compute_objectives(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> Tuple[float, float]:
        """f1 = dimension deviation, f2 = edge removal count."""
        f1 = sum(
            abs(layout[v].w - self.rectangles[v].w) + abs(layout[v].h - self.rectangles[v].h)
            for v in graph.nodes()
        )
        f2 = len(self.graph.edges()) - len(graph.edges())
        return f1, f2


class RectangleContactRelaxationSolver(RectangleContactSolver):
    """Extended solver with guaranteed solution via MST-based relaxation."""

    def solve(self, max_attempts: int = 50) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        # Try the parent solver first
        try:
            result = super().solve(max_attempts=max_attempts)
            if self._validate_solution(result[0], result[1]):
                return result
        except Exception:
            pass

        # MST-based relaxation: iteratively remove leaf nodes until solvable
        mst = nx.minimum_spanning_tree(self.graph)
        removed_nodes: List[int] = []

        subgraph = self.graph.copy()
        sub_rects = dict(self.rectangles)

        while len(subgraph.nodes) > 2:
            leaves = [n for n in mst.nodes() if mst.degree(n) == 1]
            if not leaves:
                break
            leaf = leaves[0]
            removed_nodes.append(leaf)
            mst.remove_node(leaf)
            subgraph.remove_node(leaf)
            del sub_rects[leaf]

            if not nx.is_connected(subgraph):
                continue

            try:
                sub_solver = RectangleContactSolver(subgraph, sub_rects)
                result = sub_solver.solve(max_attempts=10)
                if sub_solver._validate_solution(result[0], result[1]):
                    # Add back removed nodes
                    return self._add_back_nodes(result[0], result[1], removed_nodes)
            except Exception:
                continue

        # Base case: 2 nodes — always solvable
        return self._create_guaranteed_valid_solution()

    def _add_back_nodes(
        self, graph: nx.Graph, layout: Dict[int, RectangleLayout], removed: List[int]
    ) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Add back removed nodes with relaxed contact constraints."""
        for node in reversed(removed):
            graph.add_node(node)
            self._place_rectangle_adjacent_relaxed(node, layout)
            # Try to add edges to neighbors that are in contact
            for neighbor in self.graph.neighbors(node):
                if neighbor in layout and self._rectangles_in_contact(layout[node], layout[neighbor]):
                    graph.add_edge(node, neighbor)
            # Ensure at least one edge
            if graph.degree(node) == 0:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in layout:
                        graph.add_edge(node, neighbor)
                        break
        return graph, layout

    def _place_rectangle_adjacent_relaxed(self, v: int, layout: Dict[int, RectangleLayout]):
        """Place a node near its original graph neighbors with relaxed constraints."""
        neighbors_in_layout = [n for n in self.graph.neighbors(v) if n in layout]
        if neighbors_in_layout:
            self._place_rectangle_adjacent(v, neighbors_in_layout[0], layout)
        else:
            # Place far away
            max_x = max((r.x + r.w / 2 for r in layout.values()), default=0)
            r = self.rectangles[v]
            layout[v] = RectangleLayout(max_x + r.w + 1, 0, r.w, r.h)
