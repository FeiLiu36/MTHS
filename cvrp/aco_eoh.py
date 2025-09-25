import numpy as np
import random


class CVRPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: list, vehicle_capacity: int):
        """
        Initialize the CVRP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each node, including the depot.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between nodes.
            demands: List of integers representing the demand of each node (first node is typically the depot with zero demand).
            vehicle_capacity: Integer representing the maximum capacity of each vehicle.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity

    # --- The original ACO code, reconstructed with NumPy, is encapsulated below ---

    class _ACO():

        def __init__(self,  # 0: depot
                     distances,  # (n, n)
                     demand,  # (n, )
                     heuristic,  # (n, n)
                     capacity,
                     n_ants=30,
                     decay=0.9,
                     alpha=1,
                     beta=1,
                     ):

            self.problem_size = len(distances)
            self.distances = np.array(distances) if not isinstance(distances, np.ndarray) else distances
            self.demand = np.array(demand) if not isinstance(demand, np.ndarray) else demand
            self.capacity = capacity

            self.n_ants = n_ants
            self.decay = decay
            self.alpha = alpha
            self.beta = beta

            self.pheromone = np.ones_like(self.distances)
            self.heuristic = np.array(heuristic) if not isinstance(heuristic, np.ndarray) else heuristic

            self.shortest_path = None
            self.lowest_cost = float('inf')

        def run(self, n_iterations):
            for _ in range(n_iterations):
                paths = self.gen_path()
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
                # Use np.roll to get the next node in the path
                self.pheromone[path[:-1], np.roll(path, shift=-1)[:-1]] += 1.0 / cost
            self.pheromone[self.pheromone < 1e-10] = 1e-10

        def gen_path_costs(self, paths):
            u = paths.T  # shape: (n_ants, max_seq_len)
            v = np.roll(u, shift=-1, axis=1)
            return np.sum(self.distances[u[:, :-1], v[:, :-1]], axis=1)

        def gen_path(self):
            actions = np.zeros((self.n_ants,), dtype=np.int64)
            visit_mask = np.ones(shape=(self.n_ants, self.problem_size))
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity = np.zeros(shape=(self.n_ants,))

            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

            paths_list = [actions]  # paths_list[i] is the ith move (array) for all ants

            done = self.check_done(visit_mask, actions)
            while not done:
                actions = self.pick_move(actions, visit_mask, capacity_mask)
                paths_list.append(actions)
                visit_mask = self.update_visit_mask(visit_mask, actions)
                used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
                done = self.check_done(visit_mask, actions)

            return np.stack(paths_list, axis=0)

        def pick_move(self, prev, visit_mask, capacity_mask):
            pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
            heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
            dist = ((pheromone ** self.alpha) * (
                        heuristic ** self.beta) * visit_mask * capacity_mask)  # shape: (n_ants, p_size)

            actions = np.zeros(self.n_ants, dtype=np.int64)
            # Sample next move for each ant
            for i in range(self.n_ants):
                probabilities = dist[i]
                total_prob = np.sum(probabilities)
                if total_prob > 0:
                    probabilities = probabilities / total_prob  # Normalize
                    actions[i] = np.random.choice(self.problem_size, p=probabilities)
                else:
                    # Fallback for stuck ants: choose randomly from allowed moves
                    allowed_mask = (visit_mask[i] * capacity_mask[i]).astype(bool)
                    allowed_indices = np.where(allowed_mask)[0]
                    if len(allowed_indices) > 0:
                        actions[i] = np.random.choice(allowed_indices)
                    else:
                        actions[i] = 0  # Failsafe to depot
            return actions

        def update_visit_mask(self, visit_mask, actions):
            visit_mask[np.arange(self.n_ants), actions] = 0
            visit_mask[:, 0] = 1  # depot can be revisited with one exception
            # one exception is here: if an ant is at the depot but hasn't visited all customers, it can't stay at the depot
            visit_mask[(actions == 0) & (visit_mask[:, 1:] != 0).any(axis=1), 0] = 0
            return visit_mask

        def update_capacity_mask(self, cur_nodes, used_capacity):
            '''
            Args:
                cur_nodes: shape (n_ants, )
                used_capacity: shape (n_ants, )
            Returns:
                used_capacity: updated capacity
                capacity_mask: updated mask
            '''
            capacity_mask = np.ones(shape=(self.n_ants, self.problem_size))
            # update capacity
            used_capacity[cur_nodes == 0] = 0
            used_capacity = used_capacity + self.demand[cur_nodes]
            # update capacity_mask using broadcasting
            remaining_capacity = self.capacity - used_capacity  # (n_ants,)
            # self.demand will broadcast from (p_size,) to (n_ants, p_size) for the comparison
            capacity_mask[self.demand > remaining_capacity[:, np.newaxis]] = 0

            return used_capacity, capacity_mask

        def check_done(self, visit_mask, actions):
            return np.all(visit_mask[:, 1:] == 0) and np.all(actions == 0)

    def _heuristics(self, distance_matrix, coordinates, demands, capacity):

        n = distance_matrix.shape[0]

        heuristics_matrix = np.zeros((n, n))


        for i in range(n):
            for j in range(n):
                if i != j:
                    if demands[j] <= capacity:  # Check capacity constraint
                        demand_weighted_score = (demands[j] ** 0.5) / (distance_matrix[i, j] + 1e-5)  # Square root weighting for demand
                        heuristics_matrix[i, j] = demand_weighted_score / (distance_matrix[i, j] ** 3 + 1e-5)  # Cubic penalty for distance
                    else:
                        heuristics_matrix[i, j] = -1 * (demands[j] - capacity)  # Strong penalty for exceeding capacity

        return heuristics_matrix

    def solve(self) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP) using Ant Colony Optimization.

        Returns:
            A one-dimensional list of integers representing the sequence of nodes visited by all vehicles.
            The depot (node 0) is used to separate different vehicle routes and appears at the start and end
            of each route. For example: [0, 1, 4, 0, 2, 3, 0] represents:
              - Route 1: 0 → 1 → 4 → 0
              - Route 2: 0 → 2 → 3 → 0

            Requirements:
            - Each route must start and end at the depot (node 0)
            - The total demand of nodes in each route must not exceed vehicle capacity
            - Each customer node (non-zero) must be visited exactly once across all routes
            - The output must be a flat list (not nested lists)
            - Depot nodes (0) separate routes and mark route boundaries
        """
        # --- ACO algorithm integration ---

        # 1. Calculate the heuristic matrix based on the provided function
        heuristic_matrix = self._heuristics(
            self.distance_matrix,
            self.coordinates,
            np.array(self.demands),
            self.vehicle_capacity
        )

        # 2. Initialize and run the ACO solver with the original settings
        aco_solver = self._ACO(
            distances=self.distance_matrix,
            demand=self.demands,
            heuristic=heuristic_matrix,
            capacity=self.vehicle_capacity,
            n_ants=30,
            decay=0.9,
            alpha=1,
            beta=1,
        )

        # Run for a fixed number of iterations (e.g., 100)
        aco_solver.run(n_iterations=100)

        # 3. Format and return the best found solution
        if aco_solver.shortest_path is not None:
            # Convert the resulting numpy array to a flat list
            solution = aco_solver.shortest_path.tolist()
        else:
            # Fallback if no solution is found
            solution = []

        return solution
