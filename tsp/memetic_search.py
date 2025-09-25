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
        self.n_cities = len(coordinates)

    # --- MEMETIC ALGORITHM COMPONENTS ---

    def _calculate_tour_distance(self, tour: np.ndarray) -> float:
        """Calculates the total distance of a given tour."""
        total_distance = 0
        for i in range(self.n_cities):
            u = tour[i]
            v = tour[(i + 1) % self.n_cities]
            total_distance += self.distance_matrix[u, v]
        return total_distance

    def _generate_nearest_neighbor_tour(self, start_node: int) -> np.ndarray:
        """Generates a single tour using the nearest neighbor heuristic."""
        tour = [start_node]
        unvisited = set(range(self.n_cities))
        unvisited.remove(start_node)
        current_city = start_node

        while unvisited:
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city, city])
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city

        return np.array(tour)

    def _initialize_population(self, population_size: int) -> list:
        """Initializes the population using a mix of random and nearest neighbor tours."""
        population = []
        # Create a few high-quality tours to seed the population
        for i in range(min(population_size, self.n_cities)):
            population.append(self._generate_nearest_neighbor_tour(i))

        # Fill the rest with random tours
        while len(population) < population_size:
            tour = np.random.permutation(self.n_cities)
            population.append(tour)

        return population

    def _tournament_selection(self, population: list, fitnesses: list, k: int = 5) -> np.ndarray:
        """Selects a parent from the population using tournament selection."""
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        winner_index_in_tournament = np.argmin(tournament_fitnesses)
        winner_index_in_population = tournament_indices[winner_index_in_tournament]

        return population[winner_index_in_population]

    def _edge_recombination_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Performs Edge Recombination Crossover (ERX)."""
        adj_map = {i: set() for i in range(self.n_cities)}

        def add_neighbors(parent):
            for i in range(self.n_cities):
                p, n = parent[i - 1], parent[(i + 1) % self.n_cities]
                adj_map[parent[i]].update([p, n])

        add_neighbors(parent1)
        add_neighbors(parent2)

        offspring = []
        current_node = parent1[0]
        unvisited = set(range(self.n_cities))

        while len(offspring) < self.n_cities:
            offspring.append(current_node)
            unvisited.remove(current_node)

            if not unvisited:
                break

            # Remove current node from all adjacency lists
            for node in adj_map:
                adj_map[node].discard(current_node)

            neighbors = adj_map[current_node]
            if not neighbors:
                # If no neighbors, pick a random unvisited node
                next_node = random.choice(list(unvisited))
            else:
                # Pick neighbor with the smallest number of other neighbors
                min_len = float('inf')
                next_node = -1
                for neighbor in neighbors:
                    if len(adj_map[neighbor]) < min_len:
                        min_len = len(adj_map[neighbor])
                        next_node = neighbor
                    # Tie-breaking
                    elif len(adj_map[neighbor]) == min_len:
                        if random.random() < 0.5:
                            next_node = neighbor
            current_node = next_node

        return np.array(offspring)

    def _two_opt_local_search(self, tour: np.ndarray) -> np.ndarray:
        """Improves a tour using the 2-opt local search algorithm."""
        best_tour = tour.copy()
        best_distance = self._calculate_tour_distance(best_tour)
        improved = True

        while improved:
            improved = False
            for i in range(1, self.n_cities - 1):
                for j in range(i + 1, self.n_cities):
                    # Current segment edges: (i-1, i) and (j, j+1)
                    # Proposed new edges: (i-1, j) and (i, j+1)
                    # We are reversing the segment from i to j
                    current_dist = self.distance_matrix[best_tour[i - 1], best_tour[i]] + \
                                   self.distance_matrix[best_tour[j], best_tour[(j + 1) % self.n_cities]]

                    new_dist = self.distance_matrix[best_tour[i - 1], best_tour[j]] + \
                               self.distance_matrix[best_tour[i], best_tour[(j + 1) % self.n_cities]]

                    if new_dist < current_dist:
                        # Reverse the segment to improve the tour
                        best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                        best_distance = self._calculate_tour_distance(best_tour)  # Recalculate full distance
                        improved = True

            # After iterating through all pairs, if an improvement was made,
            # the outer while loop will restart the search from the beginning of the new tour.

        return best_tour

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using a Memetic Algorithm.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.
        """
        # --- Algorithm Parameters ---
        population_size = 30
        generations = 50
        tournament_size = 5
        # Stop if no improvement is seen for this many generations
        patience = 40

        # --- Initialization ---
        population = self._initialize_population(population_size)
        fitnesses = [self._calculate_tour_distance(tour) for tour in population]

        best_tour_overall = population[np.argmin(fitnesses)]
        best_dist_overall = min(fitnesses)

        generations_no_improvement = 0

        #print(f"Initial best distance: {best_dist_overall:.2f}")

        # --- Main Evolutionary Loop ---
        for gen in range(generations):
            # --- Selection ---
            parent1 = self._tournament_selection(population, fitnesses, k=tournament_size)
            parent2 = self._tournament_selection(population, fitnesses, k=tournament_size)

            # --- Crossover ---
            offspring = self._edge_recombination_crossover(parent1, parent2)

            # --- Local Search (The "Meme" part) ---
            # This is the key step of a Memetic Algorithm
            improved_offspring = self._two_opt_local_search(offspring)

            # --- Evaluation and Replacement ---
            offspring_fitness = self._calculate_tour_distance(improved_offspring)

            # Replace the worst individual in the population if the new one is better
            worst_fitness_idx = np.argmax(fitnesses)
            if offspring_fitness < fitnesses[worst_fitness_idx]:
                population[worst_fitness_idx] = improved_offspring
                fitnesses[worst_fitness_idx] = offspring_fitness

            # --- Update Best Solution Found ---
            current_best_idx = np.argmin(fitnesses)
            if fitnesses[current_best_idx] < best_dist_overall:
                best_dist_overall = fitnesses[current_best_idx]
                best_tour_overall = population[current_best_idx]
                generations_no_improvement = 0
                #print(f"Generation {gen + 1}: New best distance: {best_dist_overall:.2f}")
            else:
                generations_no_improvement += 1

            # --- Termination Condition ---
            if generations_no_improvement >= patience:
                #print(f"Stopping early after {patience} generations with no improvement.")
                break

        #print(f"Finished. Final best distance: {best_dist_overall:.2f}")
        return best_tour_overall
