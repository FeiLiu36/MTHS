import numpy as np
import random
import math
from numba import njit
from typing import List, Tuple, Set


# ==============================================================================
# NUMBA-ACCELERATED STANDALONE FUNCTIONS
# These are placed outside the class to allow for clean @njit decoration.
# ==============================================================================

@njit(fastmath=True)
def calculate_route_cost_numba(route: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Calculates the total distance of a single route using Numba for speed.
    """
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i], route[i + 1]]
    return cost


@njit(fastmath=True)
def local_search_2opt_route_numba(route: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """
    Improves a single route using an iterative 2-opt heuristic (best improvement strategy).
    This version is robust against infinite loops by only accepting strict improvements.
    Numba-accelerated for high performance.
    """
    if len(route) <= 3:
        return route

    improved = True
    n_iter = 0
    while improved and n_iter<10000:
        n_iter += 1
        improved = False
        best_delta = -1e-9  # Use a small negative epsilon for strict improvement
        best_swap = (-1, -1)

        # Iterate through all possible 2-opt swaps
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # Cost change (delta) calculation
                current_cost = distance_matrix[route[i - 1], route[i]] + distance_matrix[route[j], route[j + 1]]
                new_cost = distance_matrix[route[i - 1], route[j]] + distance_matrix[route[i], route[j + 1]]
                delta = new_cost - current_cost

                if delta < best_delta:
                    best_delta = delta
                    best_swap = (i, j)
                    improved = True

        if improved:
            i, j = best_swap
            # Perform the swap by reversing the segment
            route[i:j + 1] = route[i:j + 1][::-1]

    return route


# ==============================================================================
# CVRP SOLVER CLASS
# ==============================================================================

class CVRPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: List[int], vehicle_capacity: int):
        """
        Initialize the CVRP solver.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.n = len(coordinates)
        self.customers = list(range(1, self.n))

    # --- UTILITY & COST FUNCTIONS ---

    def _split_into_routes(self, solution: List[int]) -> List[List[int]]:
        """
        Splits a flat solution list into a list of routes.
        """
        routes = []
        current_route = [0]
        for node in solution[1:-1]:
            if node == 0:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
            else:
                current_route.append(node)
        current_route.append(0)
        routes.append(current_route)
        return routes

    def _calculate_route_cost(self, route: List[int]) -> float:
        """Wrapper for the Numba-accelerated route cost calculation."""
        route_np = np.array(route, dtype=np.int32)
        return calculate_route_cost_numba(route_np, self.distance_matrix)

    def _calculate_solution_cost(self, solution: List[int]) -> float:
        """
        Calculates the total distance of a full solution.
        """
        cost = 0.0
        for i in range(len(solution) - 1):
            cost += self.distance_matrix[solution[i], solution[i + 1]]
        return cost

    def _is_valid(self, solution: List[int]) -> bool:
        """Checks if a solution is valid (capacity constraints and customer visits)."""
        routes = self._split_into_routes(solution)
        routes_customers_only = [r[1:-1] for r in routes if len(r) > 2]

        visited_customers: Set[int] = set()
        total_visited_count = 0
        for route in routes_customers_only:
            route_demand = self.demands[route].sum()
            if route_demand > self.vehicle_capacity:
                # print(f"Validation Error: Route demand {route_demand} exceeds capacity {self.vehicle_capacity}")
                return False
            for node in route:
                if node in visited_customers:
                    # print(f"Validation Error: Customer {node} visited more than once.")
                    return False
                visited_customers.add(node)
            total_visited_count += len(route)

        all_customers_visited = len(visited_customers) == self.n - 1
        if not all_customers_visited:
            # print(f"Validation Error: Not all customers visited. Visited: {len(visited_customers)}, Total: {self.n-1}")
            return False

        return True

    # --- MEMETIC ALGORITHM CORE ---

    def _generate_random_solution(self) -> List[int]:
        """Generates a valid random solution using a simple greedy insertion heuristic."""
        customers = list(range(1, self.n))
        random.shuffle(customers)

        solution = [0]
        current_capacity = 0
        for customer in customers:
            if current_capacity + self.demands[customer] <= self.vehicle_capacity:
                solution.append(customer)
                current_capacity += self.demands[customer]
            else:
                solution.append(0)
                solution.append(customer)
                current_capacity = self.demands[customer]
        solution.append(0)
        return solution

    def _tournament_selection(self, population: List[List[int]], fitnesses: List[float], k: int = 3) -> int:
        """
        Selects a parent's INDEX from the population using tournament selection.
        """
        tournament_indices = random.sample(range(len(population)), k)
        best_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if fitnesses[idx] < fitnesses[best_idx]:
                best_idx = idx
        return best_idx

    def _repair_solution(self, customers_order: List[int]) -> List[int]:
        """
        Takes a list of customers and builds a valid solution by inserting depots
        based on vehicle capacity.
        """
        if not customers_order:
            return [0, 0]
        repaired_solution = [0]
        current_capacity = 0
        for customer in customers_order:
            if current_capacity + self.demands[customer] > self.vehicle_capacity:
                repaired_solution.append(0)
                current_capacity = 0
            repaired_solution.append(customer)
            current_capacity += self.demands[customer]
        repaired_solution.append(0)
        return repaired_solution

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Performs Order Crossover (OX1) and repairs the solution."""
        p1_customers = [c for c in parent1 if c != 0]
        p2_customers = [c for c in parent2 if c != 0]

        size = len(p1_customers)
        if size == 0: return [0, 0]

        child_customers = [None] * size
        start, end = sorted(random.sample(range(size), 2))

        child_customers[start:end + 1] = p1_customers[start:end + 1]
        child_set = set(child_customers[start:end + 1])

        p2_idx = 0
        for i in range(size):
            if child_customers[i] is None:
                while p2_idx < len(p2_customers) and p2_customers[p2_idx] in child_set:
                    p2_idx += 1
                if p2_idx < len(p2_customers):
                    child_customers[i] = p2_customers[p2_idx]
                    p2_idx += 1

        final_child_customers = [c for c in child_customers if c is not None]
        return self._repair_solution(final_child_customers)

    def _mutate(self, solution: List[int], mutation_rate: float) -> List[int]:
        """Applies swap or inversion mutation to the customer sequence."""
        if random.random() > mutation_rate:
            return solution

        customers = [c for c in solution if c != 0]
        if len(customers) < 2:
            return solution

        idx1, idx2 = random.sample(range(len(customers)), 2)

        if random.random() < 0.5:  # Swap mutation
            customers[idx1], customers[idx2] = customers[idx2], customers[idx1]
        else:  # Inversion mutation
            start, end = sorted((idx1, idx2))
            customers[start:end + 1] = reversed(customers[start:end + 1])

        return self._repair_solution(customers)

    # --- LOCAL SEARCH (THE "MEME") ---

    def _local_search_2opt(self, solution: List[int]) -> List[int]:
        """
        Improves a solution by applying an iterative 2-opt heuristic to each route.
        """
        routes = self._split_into_routes(solution)
        improved_routes = []

        for route in routes:
            route_np = np.array(route, dtype=np.int32)
            # Call the fast, Numba-compiled iterative 2-opt function
            improved_route_np = local_search_2opt_route_numba(route_np, self.distance_matrix)
            improved_routes.append(list(improved_route_np))

        # Reconstruct the flat solution list from the improved routes
        new_solution = [0]
        for route in improved_routes:
            new_solution.extend(route[1:])
        return new_solution

    def _relocate_operator(self, solution: List[int]) -> List[int]:
        """
        Inter-route relocate operator. Finds the best move of a customer
        from one route to another.
        """
        routes = self._split_into_routes(solution)
        best_delta = -1e-9  # For strict improvement
        best_move = {'from_route_idx': -1, 'to_route_idx': -1, 'cust_idx': -1, 'insert_pos': -1}

        for from_route_idx in range(len(routes)):
            route_from = routes[from_route_idx]
            if len(route_from) <= 2: continue

            for cust_idx in range(1, len(route_from) - 1):
                customer = route_from[cust_idx]

                # Cost of removing customer from its current route
                prev_node = route_from[cust_idx - 1]
                next_node = route_from[cust_idx + 1]
                delta_remove = (self.distance_matrix[prev_node, next_node] -
                                self.distance_matrix[prev_node, customer] -
                                self.distance_matrix[customer, next_node])

                for to_route_idx in range(len(routes)):
                    if from_route_idx == to_route_idx: continue
                    route_to = routes[to_route_idx]

                    # Check capacity constraint
                    route_to_demand = self.demands[[c for c in route_to if c != 0]].sum()
                    if route_to_demand + self.demands[customer] > self.vehicle_capacity:
                        continue

                    for insert_pos in range(1, len(route_to)):
                        # Cost of inserting customer into the new route
                        prev_insert_node = route_to[insert_pos - 1]
                        next_insert_node = route_to[insert_pos]
                        delta_insert = (self.distance_matrix[prev_insert_node, customer] +
                                        self.distance_matrix[customer, next_insert_node] -
                                        self.distance_matrix[prev_insert_node, next_insert_node])

                        total_delta = delta_remove + delta_insert
                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = {'from_route_idx': from_route_idx, 'to_route_idx': to_route_idx,
                                         'cust_idx': cust_idx, 'insert_pos': insert_pos}

        # If a beneficial move was found, apply it
        if best_move['from_route_idx'] != -1:
            m = best_move
            from_route = routes[m['from_route_idx']]
            to_route = routes[m['to_route_idx']]

            customer_to_move = from_route.pop(m['cust_idx'])
            to_route.insert(m['insert_pos'], customer_to_move)

            # Reconstruct the flat solution list
            new_solution = [0]
            for route in routes:
                new_solution.extend(route[1:])
            return new_solution

        return solution

    def solve(self, population_size=30, generations=3000, tournament_size=5, mutation_rate=0.2) -> Tuple[
        List[int], float]:
        """
        Solve the CVRP using an improved Memetic Algorithm.
        """
        # --- INITIALIZATION ---
        population = []
        for _ in range(population_size):
            sol = self._generate_random_solution()
            sol = self._local_search_2opt(sol)  # Apply local search to initial population
            population.append(sol)

        fitnesses = [self._calculate_solution_cost(ind) for ind in population]

        best_idx = np.argmin(fitnesses)
        best_solution_overall = population[best_idx]
        best_cost_overall = fitnesses[best_idx]

        # print(f"Initial best cost: {best_cost_overall:.2f}")

        for gen in range(generations):
            # --- SELECTION ---
            p1_idx = self._tournament_selection(population, fitnesses, k=tournament_size)
            p2_idx = self._tournament_selection(population, fitnesses, k=tournament_size)
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # --- CROSSOVER & MUTATION ---
            offspring = self._order_crossover(parent1, parent2)
            offspring = self._mutate(offspring, mutation_rate)

            # --- LOCAL SEARCH (MEME) ---
            # Apply both intra-route and inter-route local search operators
            offspring = self._local_search_2opt(offspring)
            offspring = self._relocate_operator(offspring)

            # --- EVALUATION & SURVIVOR SELECTION (Steady-State) ---
            offspring_fitness = self._calculate_solution_cost(offspring)

            # Replace one of the parents (the worst of the two)
            target_idx = p1_idx if fitnesses[p1_idx] > fitnesses[p2_idx] else p2_idx

            if offspring_fitness < fitnesses[target_idx]:
                population[target_idx] = offspring
                fitnesses[target_idx] = offspring_fitness

                if offspring_fitness < best_cost_overall:
                    best_cost_overall = offspring_fitness
                    best_solution_overall = offspring
                    # Optional: uncomment to see progress
                    # print(f"Gen {gen + 1}/{generations}: New best cost = {best_cost_overall:.2f}")

        # Final check for validity, which is good practice
        if not self._is_valid(best_solution_overall):
            # This should ideally not happen with the repair mechanisms in place
            # print("Warning: The best solution found is not valid.")
            return [], float('inf')

        return best_solution_overall
