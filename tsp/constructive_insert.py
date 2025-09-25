import numpy as np


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
        self.n_cities = len(coordinates)
        if self.n_cities < 2:
            raise ValueError("TSP requires at least 2 cities.")

    # --- your code here ---

    def _calculate_insertion_cost(self, city_k, tour_edge_i, tour_edge_j):
        """Calculates the cost of inserting city k between i and j."""
        # Cost increase = dist(i, k) + dist(k, j) - dist(i, j)
        return self.distance_matrix[tour_edge_i, city_k] + \
            self.distance_matrix[city_k, tour_edge_j] - \
            self.distance_matrix[tour_edge_i, tour_edge_j]

    def _simple_insertion(self):
        """
        Implements the simple insertion heuristic for the TSP.

        1. Start with a small sub-tour (e.g., city 0 and 1).
        2. Iteratively select the next unvisited city in a fixed order (e.g., 2, 3, 4...)
           and insert it into the tour at the position that minimizes the tour length increase.
        """
        # List of all cities to be inserted, in a fixed order.
        cities_to_insert = list(range(self.n_cities))

        # 1. Initialization: Start with a sub-tour of the first two cities.
        # This is the simplest initialization.
        initial_city_1 = cities_to_insert.pop(0)
        initial_city_2 = cities_to_insert.pop(0)
        tour = [initial_city_1, initial_city_2]

        # 2. Iteration: Add remaining cities one by one based on their original index order.
        for city_k in cities_to_insert:
            best_insertion_cost = float('inf')
            best_insertion_index = -1

            # Find the best place to insert this specific city_k into the current tour.
            for i in range(len(tour)):
                # The edge is between tour[i] and tour[(i + 1) % len(tour)]
                city_i = tour[i]
                city_j = tour[(i + 1) % len(tour)]

                cost = self._calculate_insertion_cost(city_k, city_i, city_j)

                if cost < best_insertion_cost:
                    best_insertion_cost = cost
                    # We will insert at index i + 1
                    best_insertion_index = i + 1

            # 3. Insert the city at the best position found.
            tour.insert(best_insertion_index, city_k)

        return np.array(tour)


    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using the Simple Insertion heuristic.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        if self.n_cities <= 2:
            return np.arange(self.n_cities)

        # --- your code here ---
        tour = self._simple_insertion()

        return tour
