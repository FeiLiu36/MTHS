import numpy as np
import numpy.typing as npt
import numba as nb

# Type Aliases and Numba settings from the original code
FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
usecache = True


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16), nogil=True, cache=usecache)
def _two_opt_once( distmat, tour, fixed_i=0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i == 0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (distmat[node_prev, node_j]
                      + distmat[node_i, node_next]
                      - distmat[node_prev, node_i]
                      - distmat[node_j, node_next])
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q + 1] = np.flip(tour[p: q + 1])
        return delta
    else:
        return 0.0


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16), nogil=True, cache=usecache)
def _relocate_once( distmat, tour, fixed_i=0):
    n = distmat.shape[0]
    delta = p = q = 0
    for i in range(1, n) if fixed_i == 0 else range(fixed_i, fixed_i + 1):
        node = tour[i]
        prev_node = tour[i - 1]
        next_node = tour[(i + 1) % n]
        for j in range(n):
            if j == i or j == i - 1:
                continue
            prev_insert = tour[j]
            next_insert = tour[(j + 1) % n]
            cost = (- distmat[prev_node, node]
                    - distmat[node, next_node]
                    - distmat[prev_insert, next_insert]
                    + distmat[prev_insert, node]
                    + distmat[node, next_insert]
                    + distmat[prev_node, next_node])
            if cost < delta:
                delta, p, q = cost, i, j
    if delta >= 0:
        return 0.0
    if p < q:
        tour[p:q + 1] = np.roll(tour[p:q + 1], -1)
    else:
        tour[q:p + 1] = np.roll(tour[q:p + 1], 1)
    return delta

class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, perturbation_moves: int = 30,
                 iter_limit: int = 100):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
            perturbation_moves: The number of perturbation moves in the GLS algorithm.
            iter_limit: The number of iterations for the GLS algorithm.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.perturbation_moves = perturbation_moves
        self.iter_limit = iter_limit

    # --- All original functions are placed here as static methods ---


    #@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:]), nogil=True, cache=usecache)
    def _calculate_cost(self,distmat, tour):
        cost = distmat[tour[-1], tour[0]]
        for i in range(len(tour) - 1):
            cost += distmat[tour[i], tour[i + 1]]
        return cost

    #@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16, nb.uint16), nogil=True, cache=usecache)
    def _local_search(self,distmat, cur_tour, fixed_i=0, count=1000):
        sum_delta = 0.0
        delta = -1
        while delta < 0 and count > 0:
            delta = 0
            # Note: The original code calls the static methods directly, which is correct for Numba.
            delta += _two_opt_once(distmat, cur_tour, fixed_i)
            delta += _relocate_once(distmat, cur_tour, fixed_i)
            count -= 1
            sum_delta += delta
        return sum_delta

    #@nb.njit(nb.void(nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.uint16[:], nb.float32, nb.uint32), nogil=True, cache=usecache)
    def _perturbation(self,distmat, guide, penalty, cur_tour, k, perturbation_moves=30):
        moves = 0
        n = distmat.shape[0]
        while moves < perturbation_moves:
            # penalize edge
            max_util = 0
            max_util_idx = 0
            for i in range(n - 1):
                j = i + 1
                u, v = cur_tour[i], cur_tour[j]
                util = guide[u, v] / (1.0 + penalty[u, v])
                if util > max_util:
                    max_util_idx, max_util = i, util

            penalty[cur_tour[max_util_idx], cur_tour[max_util_idx + 1]] += 1.0
            edge_weight_guided = distmat + k * penalty

            for fixed_i in (max_util_idx, max_util_idx + 1):
                if fixed_i == 0 or fixed_i + 1 == n:
                    continue
                # Note: The original code calls the static methods directly, which is correct for Numba.
                delta = self._local_search(edge_weight_guided, cur_tour, fixed_i, 1)
                if delta < 0:
                    moves += 1

    #@nb.njit(nb.uint16[:](nb.float32[:, :], nb.uint16), nogil=True, cache=usecache)
    def _init_nearest_neighbor(self,distmat, start):
        n = distmat.shape[0]
        tour = np.zeros(n, dtype=np.uint16)
        visited = np.zeros(n, dtype=np.bool_)
        visited[start] = True
        tour[0] = start
        for i in range(1, n):
            min_dist = np.inf
            min_idx = -1
            for j in range(n):
                if not visited[j] and distmat[tour[i - 1], j] < min_dist:
                    min_dist = distmat[tour[i - 1], j]
                    min_idx = j
            tour[i] = min_idx
            visited[min_idx] = True
        return tour


    #@nb.njit(nb.uint16[:](nb.float32[:, :], nb.float32[:, :], nb.uint16, nb.int32, nb.uint16), nogil=True, cache=usecache)
    def _guided_local_search(self,
            distmat, guide, start, perturbation_moves=30, iter_limit=1000
    ) -> npt.NDArray[np.uint16]:
        penalty = np.zeros_like(distmat)

        best_tour = self._init_nearest_neighbor(distmat, start)
        self._local_search(distmat, best_tour, 0, 1000)
        best_cost = self._calculate_cost(distmat, best_tour)
        k = 0.1 * best_cost / distmat.shape[0]
        cur_tour = best_tour.copy()

        for _ in range(iter_limit):
            self._perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves)
            self._local_search(distmat, cur_tour, 0, 1000)
            cur_cost = self._calculate_cost(distmat, cur_tour)
            if cur_cost < best_cost:
                best_tour, best_cost = cur_tour.copy(), cur_cost
        return best_tour

    def _guide(self,distance_matrix):
        num_nodes = distance_matrix.shape[0]
        heuristics_matrix = np.zeros((num_nodes, num_nodes))

        # Calculate the sum of distances and the maximum distance from each node to all others
        for i in range(num_nodes):
            total_distance = np.sum(distance_matrix[i, :])
            max_distance = np.max(distance_matrix[i, :])

            for j in range(num_nodes):
                # Avoid self-distance and compute heuristic for edge (i, j)
                if i != j:
                    # Modified penalty factor based on maximum distance
                    heuristics_matrix[i, j] = (distance_matrix[i, j] / total_distance) * (
                            1 + distance_matrix[i, j] / max_distance)

        return heuristics_matrix

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using Guided Local Search.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        # --- The logic from the original `guided_local_search` wrapper is now here ---

        # 1. Create the guide matrix
        guide_matrix = self._guide(self.distance_matrix)

        # 2. Call the main Numba-jitted GLS function
        tour = self._guided_local_search(
            distmat=self.distance_matrix.astype(np.float32),
            guide=guide_matrix.astype(np.float32),
            start=0,  # As per the original wrapper function
            perturbation_moves=self.perturbation_moves,
            iter_limit=self.iter_limit,
        )

        return tour
