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

    # --- your code here ---

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using a simple nearest neighbor heuristic.

        This implementation starts at city 0 and greedily selects the nearest
        unvisited city until all cities have been visited. It does not iterate
        from different starting points or perform any post-tour improvement.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        if self.n_cities == 0:
            return np.array([], dtype=int)

        # Start the tour at the first city (index 0).
        start_node = 0
        current_city = start_node

        # A boolean array to keep track of visited cities.
        visited = np.zeros(self.n_cities, dtype=bool)

        # The tour starts with the chosen starting node.
        tour = [current_city]
        visited[current_city] = True

        # Loop until all cities have been added to the tour.
        # We need to add n-1 more cities.
        while len(tour) < self.n_cities:
            # Get the distances from the current city to all other cities.
            # Make a copy to avoid modifying the original distance matrix.
            distances_from_current = self.distance_matrix[current_city].copy()

            # Use a large number (infinity) to mask already visited cities,
            # ensuring they are not chosen as the nearest neighbor.
            distances_from_current[visited] = np.inf

            # Find the index (city) of the minimum distance among unvisited cities.
            next_city = np.argmin(distances_from_current)

            # Add the nearest city to the tour and mark it as visited.
            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city

        return np.array(tour)
