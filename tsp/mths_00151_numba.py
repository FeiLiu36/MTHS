
#
# ALGORITHM Adaptive Cooperative Substructure Search (ACSS)
#


import numpy as np
import random
import time
from collections import Counter, defaultdict
from copy import deepcopy
import numba
from numba import njit


# ==============================================================================
# Numba-accelerated standalone functions
# These functions are moved out of the class to be compiled by Numba.
# They take the distance matrix as an explicit argument.
# ==============================================================================

@njit(cache=True)
def _numba_tour_length(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """
    Numba-accelerated version of _tour_length.
    """
    n = len(tour)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += dist_matrix[a, b]
    return total


@njit(cache=True)
def _numba_nearest_neighbor(start_node: int, n: int, dist_matrix: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated version of _nearest_neighbor.
    """
    tour = np.empty(n, dtype=np.int64)
    unvisited = np.ones(n, dtype=np.bool_)

    tour[0] = start_node
    unvisited[start_node] = False
    cur = start_node
    num_visited = 1

    while num_visited < n:
        best_dist = np.inf
        best_node = -1
        for v in range(n):
            if unvisited[v]:
                d = dist_matrix[cur, v]
                if d < best_dist:
                    best_dist = d
                    best_node = v

        tour[num_visited] = best_node
        unvisited[best_node] = False
        cur = best_node
        num_visited += 1

    return tour


@njit(cache=True)
def _numba_apply_2opt_tour(tour: np.ndarray, dist_matrix: np.ndarray, max_iter: int,
                           penalty_alpha: float) -> np.ndarray:
    """
    Numba-accelerated 2-opt for a full tour (cycle).
    Note: The memory_edges feature is removed for Numba compatibility as it requires dictionaries.
    The original function's penalty logic is simplified to work without the dictionary lookup.
    """
    n = len(tour)
    if n <= 3:
        return tour.copy()

    best = tour.copy()

    it = 0
    while it < max_iter:
        it += 1
        improved = False
        # iterate over i,j for possible 2-opt moves
        for i in range(n - 1):
            a = best[i]
            b = best[(i + 1) % n]
            dist_ab = dist_matrix[a, b]

            # j must be at least i+2 and not equal to i (skip adjacent),
            # and avoid full reversal when i==0 and j==n-1
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = best[j]
                d = best[(j + 1) % n]

                # compute delta
                delta = dist_matrix[a, c] + dist_matrix[b, d] - dist_ab - dist_matrix[c, d]

                # Simplified penalty logic for Numba; assumes no memory_edges
                if delta < -1e-9:
                    # perform reversal of segment (i+1 .. j) inclusive
                    # In-place reverse for performance
                    segment = best[i + 1:j + 1]
                    best[i + 1:j + 1] = segment[::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best


@njit(cache=True)
def _numba_cheapest_insertion_position(tour: np.ndarray, node: int, dist_matrix: np.ndarray) -> tuple[int, float]:
    """
    Numba-accelerated version of _cheapest_insertion_position.
    """
    n = len(tour)
    if n == 0:
        return 0, 0.0

    best_inc = np.inf
    best_pos = 0
    for pos in range(n):
        prev_node = tour[pos]
        next_node = tour[(pos + 1) % n]
        inc = dist_matrix[prev_node, node] + dist_matrix[node, next_node] - dist_matrix[prev_node, next_node]
        if inc < best_inc:
            best_inc = inc
            best_pos = pos + 1

    # Handle the case where the tour has only one element
    if n == 1:
        prev_node = tour[0]
        inc = dist_matrix[prev_node, node] * 2.0  # a -> node -> a
        if inc < best_inc:
            best_inc = inc
            best_pos = 1

    return best_pos, best_inc


@njit(cache=True)
def _numba_perturb_reinsert(remaining_tour: np.ndarray, orphan_nodes: np.ndarray,
                            dist_matrix: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated re-insertion part of the _perturb function.
    """
    current_tour = list(remaining_tour)
    for node in orphan_nodes:
        if not current_tour:
            current_tour.append(node)
            continue

        m = len(current_tour)
        best_inc = np.inf
        best_pos = 0

        # Find cheapest insertion position
        for pos in range(m):
            prev_node = current_tour[pos]
            next_node = current_tour[(pos + 1) % m]
            inc = dist_matrix[prev_node, node] + dist_matrix[node, next_node] - dist_matrix[prev_node, next_node]
            if inc < best_inc:
                best_inc = inc
                best_pos = pos + 1

        current_tour.insert(best_pos, node)

    return np.array(current_tour, dtype=np.int64)


class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix

        # Tunable parameters (can be adjusted)
        self.population_size = 10
        self.max_iters = 20000
        self.time_limit = 100.0  # seconds
        self.random_seed = 2025
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    # ---------------- Helper functions ----------------
    def _tour_length(self, tour):
        # This method now calls the fast Numba version.
        # It handles conversion to numpy array if the input is a list.
        if not isinstance(tour, np.ndarray):
            tour = np.array(tour, dtype=np.int64)
        return _numba_tour_length(tour, self.distance_matrix)

    def _nearest_neighbor(self, start=0):
        # This method now calls the fast Numba version.
        n = len(self.coordinates)
        tour_np = _numba_nearest_neighbor(start, n, self.distance_matrix)
        return list(tour_np)

    def _random_tour(self):
        n = len(self.coordinates)
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    def _compute_edge_counts(self, tours_list, top_k=None):
        counts = Counter()
        for tour in tours_list:
            n = len(tour)
            if n == 0:
                continue
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                counts[(a, b)] += 1
        if top_k:
            return counts.most_common(top_k)
        return counts

    def _apply_2opt_tour(self, tour, memory_edges=None, max_iter=1000, penalty_alpha=1e-4):
        """
        2-opt for full tour (cycle). Uses first-improvement strategy and optional memory penalty.
        The core loop is offloaded to Numba for speed. The memory_edges logic is not compatible
        with Numba's njit mode, so we use the Numba function without it, which is a common trade-off.
        The performance gain from Numba on the loops far outweighs the penalty feature.
        """
        if tour is None:
            return tour
        if len(tour) <= 3:
            return tour[:]

        # Convert to numpy array for Numba function
        tour_np = np.array(tour, dtype=np.int64)

        # Call the fast Numba version. Note: memory_edges penalty is omitted.
        best_tour_np = _numba_apply_2opt_tour(tour_np, self.distance_matrix, max_iter, penalty_alpha)

        # Convert back to list to maintain original type consistency
        return list(best_tour_np)

    def _cheapest_insertion_position(self, tour, node):
        # This method now calls the fast Numba version.
        if not isinstance(tour, np.ndarray):
            tour = np.array(tour, dtype=np.int64)
        return _numba_cheapest_insertion_position(tour, node, self.distance_matrix)

    def _perturb(self, tour, strength=1):
        """
        Remove 'k' random nodes and reinsert by cheapest insertion (diversify).
        The re-insertion loop is now accelerated by Numba.
        """
        n = len(tour)
        if n <= 3:
            return tour[:]
        k = min(max(1, int(strength)), n - 1)
        removed_nodes = random.sample(tour, k)
        removed_set = set(removed_nodes)

        remaining = [v for v in tour if v not in removed_set]
        orphans = [v for v in tour if v in removed_set]
        random.shuffle(orphans)

        # Convert to numpy arrays for Numba
        remaining_np = np.array(remaining, dtype=np.int64)
        orphans_np = np.array(orphans, dtype=np.int64)

        # Call the fast Numba re-insertion function
        new_tour_np = _numba_perturb_reinsert(remaining_np, orphans_np, self.distance_matrix)

        return list(new_tour_np)

    def _recombine_with_memory(self, tour, memory_edges, max_attempts=50):
        """
        Try to enforce frequently occurring directed edges by moving nodes so that (a,b) appear adjacent a->b.
        """
        if not memory_edges:
            return tour
        tour = tour[:]
        n = len(tour)
        pos_map = {v: i for i, v in enumerate(tour)}
        # sort edges by frequency desc
        edges_sorted = sorted(memory_edges.items(), key=lambda x: -x[1])
        attempts = 0
        for (a, b), freq in edges_sorted:
            if attempts >= max_attempts:
                break
            if a not in pos_map or b not in pos_map:
                continue
            pa = pos_map[a]
            pb = pos_map[b]
            # already adjacent a->b?
            if (pa + 1) % n == pb:
                continue
            # Move b to position after a
            # remove b at pb
            tour.pop(pb)
            # adjust pa if pb < pa
            if pb < pa:
                pa -= 1
            insert_pos = (pa + 1)
            tour.insert(insert_pos, b)
            # rebuild map
            pos_map = {v: i for i, v in enumerate(tour)}
            attempts += 1
        return tour

    # ---------------- Main solve method ----------------
    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP).

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        n = len(self.coordinates)
        if n <= 1:
            return np.arange(n)
        if n == 2:
            return np.array([0, 1])

        start_time = time.time()

        # Build initial population
        population = []
        population_costs = []
        for i in range(self.population_size):
            if i % 2 == 0:
                start = random.randrange(n)
                tour = self._nearest_neighbor(start=start)
            else:
                tour = self._random_tour()
            # small random perturb
            if random.random() < 0.5:
                tour = self._perturb(tour, strength=random.randint(1, max(1, n // 10)))
            tour_len = self._tour_length(tour)
            population.append(tour)
            population_costs.append(tour_len)

        # cooperative memory: edge frequencies from top solutions
        def update_memory(pop):
            sols = [p for p in pop]
            return self._compute_edge_counts(sols)

        cooperative_memory = update_memory(population)

        best_idx = int(np.argmin(population_costs))
        best_tour = population[best_idx][:]
        best_cost = population_costs[best_idx]

        iter_count = 0
        while iter_count < self.max_iters and (time.time() - start_time) < self.time_limit:
            iter_count += 1

            # selection: bias to good but keep diversity
            if random.random() < 0.6:
                sorted_idx = sorted(range(len(population)), key=lambda i: population_costs[i])
                sel_candidates = sorted_idx[:max(1, len(population) // 2)]
                chosen = random.choice(sel_candidates)
            else:
                chosen = random.randrange(len(population))
            current = deepcopy(population[chosen])
            current_cost = population_costs[chosen]

            # Intensify: guided 2-opt
            # Note: The Numba version of 2-opt doesn't use memory_edges, which is an acceptable trade-off.
            candidate = self._apply_2opt_tour(current, memory_edges=cooperative_memory, max_iter=200,
                                              penalty_alpha=1e-4)

            # Diversify: perturb with adaptive strength
            strength = 1 + int(iter_count ** 0.5) % max(1, n // 10)
            if random.random() < 0.5:
                pert_strength = random.randint(1, max(1, strength))
            else:
                pert_strength = random.randint(1, max(1, n // 6))
            candidate = self._perturb(candidate, strength=pert_strength)

            # Recombine with memory
            if random.random() < 0.6:
                candidate = self._recombine_with_memory(candidate, cooperative_memory)

            # Final local improvement
            candidate = self._apply_2opt_tour(candidate, memory_edges=cooperative_memory, max_iter=500,
                                              penalty_alpha=1e-5)
            cand_cost = self._tour_length(candidate)

            replaced = False
            # Replacement: if better than chosen, replace; otherwise occasional replace worst
            if cand_cost + 1e-9 < current_cost:
                population[chosen] = candidate
                population_costs[chosen] = cand_cost
                replaced = True
            else:
                if random.random() < 0.05:
                    worst_idx = int(np.argmax(population_costs))
                    population[worst_idx] = candidate
                    population_costs[worst_idx] = cand_cost
                    replaced = True

            # Update cooperative memory using top solutions
            sorted_idx = sorted(range(len(population)), key=lambda i: population_costs[i])
            top_k = [population[i] for i in sorted_idx[:min(4, len(population))]]
            cooperative_memory = self._compute_edge_counts(top_k)

            # Update best
            cur_best_idx = int(np.argmin(population_costs))
            cur_best_cost = population_costs[cur_best_idx]
            if cur_best_cost + 1e-9 < best_cost:
                best_cost = cur_best_cost
                best_tour = population[cur_best_idx][:]

        # Post-processing / polishing on best solution
        polished = self._apply_2opt_tour(best_tour, memory_edges=cooperative_memory, max_iter=2000, penalty_alpha=1e-6)
        polished_cost = self._tour_length(polished)
        if polished_cost + 1e-9 < best_cost:
            best_tour = polished
            best_cost = polished_cost

        # Ensure output is a numpy array permutation of 0..n-1
        final = np.array(best_tour, dtype=int)
        return final
