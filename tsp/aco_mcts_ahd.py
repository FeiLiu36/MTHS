import numpy as np
# import torch # No longer needed
# from torch.distributions import Categorical # No longer needed
import random
import numba
import traceback


@numba.jit(nopython=True)
def calculate_heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    """
    This is the first part of the original algorithm. It runs a preliminary, simpler
    ACO to generate a heuristic matrix for the main ACO algorithm.
    This version is optimized for Numba.
    """
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((n, n))
    pheromone_matrix = np.ones((n, n))
    alpha = 1.0
    beta = 2.0
    base_evaporation_rate = 0.1
    iterations = 100
    penalty_factor = 0.5

    best_global_length = np.inf

    for iter_num in range(iterations):
        # Numba can't compile the `print` function in nopython mode.
        # If you need to see progress, you can run the code without the decorator first.
        # print(f"ite:{iter_num}")

        best_length = np.inf
        # Numba requires arrays to be typed, so we pre-allocate the best_path.
        best_path = np.zeros(n, dtype=np.int32)

        # This loop is for each ant
        for ant in range(n):
            # Numba Change: Replaced `path = []` with a pre-allocated NumPy array.
            path = np.zeros(n, dtype=np.int32)
            # Numba Change: Replaced `visited = set()` with a boolean array for fast lookups.
            visited = np.zeros(n, dtype=np.bool_)

            current_city = np.random.randint(0, n)
            path[0] = current_city
            visited[current_city] = True
            path_idx = 1

            while path_idx < n:
                # Numba Change: Pre-allocate probabilities array instead of building a list.
                probabilities = np.zeros(n)

                for j in range(n):
                    if not visited[j]:
                        # Inlined the visibility calculation
                        visibility_val = 1 / (distance_matrix[current_city, j] ** 2) if distance_matrix[
                                                                                            current_city, j] != 0 else 0
                        probabilities[j] = (pheromone_matrix[current_city, j] ** alpha) * (visibility_val ** beta)

                total = np.sum(probabilities)
                if total > 0:
                    probabilities /= total

                    # Numba Change: `np.random.choice` with probabilities is not supported.
                    # This is the standard Numba replacement: using cumulative sum and searchsorted.
                    cum_probs = np.cumsum(probabilities)
                    random_val = np.random.rand()
                    next_city = np.searchsorted(cum_probs, random_val)
                else:
                    # This handles cases where an ant gets stuck.
                    # Numba Change: Replaced list comprehension with a simple loop to find any unvisited city.
                    found_city = False
                    for j in range(n):
                        if not visited[j]:
                            next_city = j
                            found_city = True
                            break
                    if not found_city:
                        # This should not happen if path_idx < n, but it's a safe fallback.
                        break

                path[path_idx] = next_city
                visited[next_city] = True
                current_city = next_city
                path_idx += 1

            if path_idx == n:
                # Numba Change: Path length calculation using a loop, which is fast in Numba.
                path_length = 0.0
                for i in range(n - 1):
                    path_length += distance_matrix[path[i], path[i + 1]]
                path_length += distance_matrix[path[n - 1], path[0]]

                adjusted_length = path_length + penalty_factor * (path_length / n)
                if adjusted_length < best_length:
                    best_length = adjusted_length
                    # Numba Change: Use array copy for best_path
                    best_path[:] = path[:]

        if best_length < best_global_length:
            best_global_length = best_length

        if best_length != np.inf:
            reward = max(1 / best_length, 0.1)
            for i in range(n):
                # Use modulo for the wrap-around from the last city to the first
                start_node = best_path[i]
                end_node = best_path[(i + 1) % n]
                pheromone_matrix[start_node, end_node] += reward

        evaporation_rate = base_evaporation_rate + (best_global_length / n) * 0.01
        pheromone_matrix *= (1 - evaporation_rate)

    for i in range(n):
        for j in range(n):
            if i != j:
                denom = distance_matrix[i, j] if distance_matrix[i, j] != 0 else np.inf
                heuristics_matrix[i, j] = pheromone_matrix[i, j] * (1 / denom)

    return heuristics_matrix


class TSPSolver:
    """
    Solves the Traveling Salesman Problem (TSP) using a two-stage Ant Colony Optimization (ACO) algorithm.

    The implementation is based on the provided code, which consists of:
    1. A preliminary ACO run to generate a heuristic matrix.
    2. A main, vectorized ACO algorithm (now using NumPy) that uses the generated
       heuristic matrix to find the final solution.

    All original functions and settings have been preserved and encapsulated within this class.
    """

    # The main ACO algorithm, implemented as a nested class to keep the original structure.
    class _ACO():
        def __init__(self,
                     distances,
                     heuristic,
                     n_ants=30,
                     decay=0.9,
                     alpha=1,
                     beta=1,
                     # device='cpu' # Removed, not needed for NumPy
                     ):
            self.problem_size = len(distances)
            self.distances = np.array(distances) if not isinstance(distances, np.ndarray) else distances
            self.n_ants = n_ants
            self.decay = decay
            self.alpha = alpha
            self.beta = beta

            self.pheromone = np.ones_like(self.distances)
            self.heuristic = np.array(heuristic) if not isinstance(heuristic, np.ndarray) else heuristic

            self.shortest_path = None
            self.lowest_cost = float('inf')

            # self.device = device # Removed

        def run(self, n_iterations):
            for i in range(n_iterations):
                #print(f"iteration: {i}")
                paths = self.gen_path(require_prob=False)
                costs = self.gen_path_costs(paths)

                best_cost = costs.min()
                best_idx = costs.argmin()

                if best_cost < self.lowest_cost:
                    self.shortest_path = paths[:, best_idx]
                    self.lowest_cost = best_cost

                self.update_pheronome(paths, costs)
            return self.lowest_cost

        def update_pheronome(self, paths, costs):
            '''
            Args:
                paths: numpy array with shape (problem_size, n_ants)
                costs: numpy array with shape (n_ants,)
            '''
            self.pheromone = self.pheromone * self.decay
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                # Use np.roll to get the next city in the path for each city
                rolled_path = np.roll(path, shift=-1)
                self.pheromone[path, rolled_path] += 1.0 / cost
                self.pheromone[rolled_path, path] += 1.0 / cost

        def gen_path_costs(self, paths):
            '''
            Args:
                paths: numpy array with shape (problem_size, n_ants)
            Returns:
                Lengths of paths: numpy array with shape (n_ants,)
            '''
            assert paths.shape == (self.problem_size, self.n_ants)
            u = paths.T  # shape: (n_ants, problem_size)
            v = np.roll(u, shift=1, axis=1)  # shape: (n_ants, problem_size)
            assert (self.distances[u, v] > 0).all()
            return np.sum(self.distances[u, v], axis=1)

        def gen_path(self, require_prob=False):
            '''
            Tour contruction for all ants
            Returns:
                paths: numpy array with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
                log_probs: numpy array with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
            '''
            start = np.random.randint(low=0, high=self.problem_size, size=(self.n_ants,))
            mask = np.ones(shape=(self.n_ants, self.problem_size))
            mask[np.arange(self.n_ants), start] = 0

            paths_list = []  # paths_list[i] is the ith move (array) for all ants
            paths_list.append(start)

            log_probs_list = []  # log_probs_list[i] is the ith log_prob (array) for all ants' actions

            prev = start
            for _ in range(self.problem_size - 1):
                actions, log_probs = self.pick_move(prev, mask, require_prob)
                paths_list.append(actions)
                if require_prob:
                    log_probs_list.append(log_probs)
                    mask = mask.copy()  # Use copy to ensure mask is not modified in place
                prev = actions
                mask[np.arange(self.n_ants), actions] = 0

            if require_prob:
                return np.stack(paths_list), np.stack(log_probs_list)
            else:
                return np.stack(paths_list)

        def pick_move(self, prev, mask, require_prob):
            '''
            Args:
                prev: array with shape (n_ants,), previous nodes for all ants
                mask: bool array with shape (n_ants, p_size), masks (0) for the visited cities
            '''
            pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
            heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
            dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask)  # shape: (n_ants, p_size)

            # Normalize to get probabilities, adding epsilon to avoid division by zero
            row_sums = dist.sum(axis=1, keepdims=True)
            probabilities = dist / (row_sums + 1e-10)

            # Vectorized sampling for each ant (row)
            cum_probs = probabilities.cumsum(axis=1)
            rand_vals = np.random.rand(self.n_ants, 1)
            actions = (cum_probs > rand_vals).argmax(axis=1)

            log_probs = None
            if require_prob:
                # Get the probability of the chosen action for each ant
                action_probs = probabilities[np.arange(self.n_ants), actions]
                # Add a small epsilon to avoid log(0)
                log_probs = np.log(action_probs + 1e-10)

            return actions, log_probs

    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.n = len(coordinates)

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP).

        This method orchestrates the two-stage ACO process:
        1. It calls `_calculate_heuristics` to generate a problem-specific heuristic matrix.
        2. It initializes and runs the main `_ACO` algorithm using this matrix.
        3. It returns the best tour found.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.
        """
        try:
            # Stage 1: Generate the heuristic matrix using the preliminary ACO.
            heuristic_matrix = calculate_heuristics(self.distance_matrix)

            # Stage 2: Run the main, vectorized ACO algorithm.
            # Settings are taken from the defaults in the original ACO class `__init__`.
            # The number of iterations is chosen to be 100, consistent with the heuristic calculation part.
            aco_solver = self._ACO(
                distances=self.distance_matrix,
                heuristic=heuristic_matrix,
                n_ants=30,
                decay=0.9,
                alpha=1,
                beta=1,
            )


            aco_solver.run(n_iterations=100)

            # Retrieve the best tour found.
            best_tour = aco_solver.shortest_path

            # If no tour was found (e.g., n_iterations=0), return a default tour.
            if best_tour is None:
                tour = np.arange(self.n)
            else:
                # The result is already a NumPy array.
                tour = best_tour

            return tour

        except Exception as e:
            traceback.print_exc()
