import numpy as np
import random


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
        self.n = len(coordinates)

        # --- ALNS Parameters ---
        # Iteration and temperature controls
        self.max_iterations = 1000
        self.start_temperature = 100
        self.end_temperature = 0.01
        self.cooling_rate = 0.9995

        # Degree of destruction (q)
        self.min_removal_rate = 0.1
        self.max_removal_rate = 0.4

        # Operator weight update parameters
        self.reaction_factor = 0.5  # How much weights are adjusted
        self.score_best = 33  # Score for finding a new global best
        self.score_better = 9  # Score for finding a better solution
        self.score_accepted = 5  # Score for finding an accepted worse solution

    # --- Helper Functions ---

    def _calculate_tour_cost(self, tour: np.ndarray) -> float:
        """Calculates the total length of a given tour."""
        cost = 0.0
        for i in range(self.n):
            cost += self.distance_matrix[tour[i], tour[(i + 1) % self.n]]
        return cost

    def _generate_initial_solution(self) -> np.ndarray:
        """Generates a greedy initial solution (nearest neighbor)."""
        current_city = 0
        unvisited = set(range(1, self.n))
        tour = [current_city]

        while unvisited:
            nearest_neighbor = min(unvisited, key=lambda city: self.distance_matrix[current_city, city])
            tour.append(nearest_neighbor)
            unvisited.remove(nearest_neighbor)
            current_city = nearest_neighbor

        return np.array(tour)

    # --- Destroy Operators ---

    def _random_removal(self, tour: np.ndarray, q: int) -> tuple[np.ndarray, list[int]]:
        """Removes q random cities from the tour."""
        tour_list = list(tour)
        removed_cities = []
        for _ in range(q):
            idx_to_remove = random.randint(0, len(tour_list) - 1)
            removed_cities.append(tour_list.pop(idx_to_remove))
        return np.array(tour_list), removed_cities

    def _shaw_removal(self, tour: np.ndarray, q: int) -> tuple[np.ndarray, list[int]]:
        """
        Removes q 'related' cities based on Shaw's relatedness measure.
        A random city is chosen first, and subsequent cities are chosen based
        on their proximity and tour position relative to already removed cities.
        """
        tour_list = list(tour)
        removed_cities = []

        # Pick a random first city to remove
        rand_idx = random.randint(0, len(tour_list) - 1)
        removed_cities.append(tour_list.pop(rand_idx))

        while len(removed_cities) < q:
            # Pick a random city 'r' from the set of already removed cities
            r = random.choice(removed_cities)

            # Find the city 'c' in the current tour that is most related to 'r'
            best_relatedness = -1
            most_related_city_idx = -1

            for i, c in enumerate(tour_list):
                # Relatedness R(r, c) = 1 / (distance(r, c) + tour_distance(r, c))
                # We use a simplified version for efficiency, just focusing on distance.
                # A higher score means more related.
                dist = self.distance_matrix[r, c]
                if dist == 0: continue  # Avoid division by zero
                relatedness = 1 / dist

                if relatedness > best_relatedness:
                    best_relatedness = relatedness
                    most_related_city_idx = i

            # Remove the most related city
            if most_related_city_idx != -1:
                removed_cities.append(tour_list.pop(most_related_city_idx))
            else:
                # Fallback if no related city is found (should not happen in a connected graph)
                break

        return np.array(tour_list), removed_cities

    # --- Repair Operators ---

    def _greedy_insertion(self, partial_tour: np.ndarray, removed_cities: list[int]) -> np.ndarray:
        """
        Inserts each removed city into the position in the partial tour
        that results in the minimum increase in tour cost.
        """
        tour_list = list(partial_tour)

        for city_to_insert in removed_cities:
            best_insertion_cost = float('inf')
            best_insertion_pos = -1

            # Iterate through all possible insertion positions
            for i in range(len(tour_list) + 1):
                prev_city = tour_list[i - 1]
                next_city = tour_list[i % len(tour_list)]

                # Cost change = dist(prev, new) + dist(new, next) - dist(prev, next)
                cost_change = (self.distance_matrix[prev_city, city_to_insert] +
                               self.distance_matrix[city_to_insert, next_city] -
                               self.distance_matrix[prev_city, next_city])

                if cost_change < best_insertion_cost:
                    best_insertion_cost = cost_change
                    best_insertion_pos = i

            tour_list.insert(best_insertion_pos, city_to_insert)

        return np.array(tour_list)

    # --- ALNS Core ---

    def _select_operator(self, weights: np.ndarray) -> int:
        """Selects an operator based on its weight using roulette wheel selection."""
        total_weight = np.sum(weights)
        if total_weight == 0:
            return np.random.randint(0, len(weights))

        pick = np.random.uniform(0, total_weight)
        current = 0
        for i, weight in enumerate(weights):
            current += weight
            if current > pick:
                return i
        return len(weights) - 1

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using Adaptive Large Neighborhood Search.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.
        """
        # 1. Initialize solutions
        current_tour = self._generate_initial_solution()
        current_cost = self._calculate_tour_cost(current_tour)

        best_tour = np.copy(current_tour)
        best_cost = current_cost

        # 2. Initialize operators and their weights/scores
        destroy_operators = [self._random_removal, self._shaw_removal]
        repair_operators = [self._greedy_insertion]

        destroy_weights = np.ones(len(destroy_operators))
        repair_weights = np.ones(len(repair_operators))

        destroy_scores = np.zeros(len(destroy_operators))
        repair_scores = np.zeros(len(repair_operators))

        destroy_counts = np.zeros(len(destroy_operators), dtype=int)
        repair_counts = np.zeros(len(repair_operators), dtype=int)

        # 3. Initialize temperature for simulated annealing acceptance
        temperature = self.start_temperature

        # 4. Main ALNS loop
        for i in range(self.max_iterations):
            # Select operators based on weights
            destroy_idx = self._select_operator(destroy_weights)
            repair_idx = self._select_operator(repair_weights)

            destroy_op = destroy_operators[destroy_idx]
            repair_op = repair_operators[repair_idx]

            # Determine degree of destruction (q)
            q = random.randint(int(self.n * self.min_removal_rate), int(self.n * self.max_removal_rate))

            # Create new solution
            partial_tour, removed_cities = destroy_op(np.copy(current_tour), q)
            new_tour = repair_op(partial_tour, removed_cities)
            new_cost = self._calculate_tour_cost(new_tour)

            # 5. Acceptance criterion (Simulated Annealing)
            if new_cost < current_cost:
                current_tour = new_tour
                current_cost = new_cost

                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    destroy_scores[destroy_idx] += self.score_best
                    repair_scores[repair_idx] += self.score_best
                else:
                    destroy_scores[destroy_idx] += self.score_better
                    repair_scores[repair_idx] += self.score_better

            elif random.random() < np.exp((current_cost - new_cost) / temperature):
                current_tour = new_tour
                current_cost = new_cost
                destroy_scores[destroy_idx] += self.score_accepted
                repair_scores[repair_idx] += self.score_accepted

            destroy_counts[destroy_idx] += 1
            repair_counts[repair_idx] += 1

            # 6. Update temperature
            temperature *= self.cooling_rate

            # 7. Update operator weights periodically (e.g., every 100 iterations)
            if i % 100 == 0 and i > 0:
                for j in range(len(destroy_weights)):
                    if destroy_counts[j] > 0:
                        destroy_weights[j] = (1 - self.reaction_factor) * destroy_weights[j] + \
                                             self.reaction_factor * destroy_scores[j] / destroy_counts[j]

                for j in range(len(repair_weights)):
                    if repair_counts[j] > 0:
                        repair_weights[j] = (1 - self.reaction_factor) * repair_weights[j] + \
                                            self.reaction_factor * repair_scores[j] / repair_counts[j]

                # Reset scores and counts for the next segment
                destroy_scores.fill(0)
                repair_scores.fill(0)
                destroy_counts.fill(0)
                repair_counts.fill(0)

        print(f"ALNS finished. Best cost found: {best_cost:.2f}")
        return best_tour
