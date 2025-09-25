import numpy as np
import random
import math


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
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.n_customers = len(coordinates) - 1
        self.nodes = list(range(len(coordinates)))
        self.depot = 0

    # --- Helper Functions ---

    def _calculate_cost(self, solution: list) -> float:
        """Calculates the total distance of a solution."""
        total_distance = 0
        for i in range(len(solution) - 1):
            u, v = solution[i], solution[i + 1]
            total_distance += self.distance_matrix[u, v]
        return total_distance

    def _to_routes(self, solution: list) -> list:
        """Converts a flat solution list to a list of routes."""
        routes = []
        route = []
        for node in solution:
            if node == self.depot:
                if route:  # if route is not empty
                    routes.append([self.depot] + route + [self.depot])
                    route = []
            else:
                route.append(node)
        return routes

    def _to_flat_solution(self, routes: list) -> list:
        """Converts a list of routes to a flat solution list."""
        if not routes:
            return [0, 0]
        flat_list = []
        for i, route in enumerate(routes):
            # A valid route is [0, c1, c2, ..., 0]
            # We take the customers [c1, c2, ...]
            customers_in_route = route[1:-1]
            if i == 0:
                flat_list.extend([0] + customers_in_route)
            else:
                flat_list.extend([0] + customers_in_route)
        flat_list.append(0)
        return flat_list

    def _is_feasible(self, solution: list) -> bool:
        """Checks if a solution is feasible."""
        routes = self._to_routes(solution)
        visited_customers = set()
        for route in routes:
            route_demand = sum(self.demands[node] for node in route)
            if route_demand > self.vehicle_capacity:
                # print(f"Capacity violation: {route_demand} > {self.vehicle_capacity}")
                return False
            for node in route[1:-1]:  # Exclude depots
                if node in visited_customers:
                    # print(f"Duplicate customer visit: {node}")
                    return False
                visited_customers.add(node)

        all_customers = set(self.nodes) - {self.depot}
        if visited_customers != all_customers:
            # print(f"Missing customers: {all_customers - visited_customers}")
            return False

        return True

    # --- Initial Solution ---

    def _generate_initial_solution(self) -> list:
        """
        Generates an initial solution using Solomon's I1 heuristic (simplified).
        It prioritizes nodes that are far from the depot and have a tight time window (here simplified to just distance).
        """
        unvisited = set(self.nodes) - {self.depot}
        routes = []

        while unvisited:
            route = [self.depot]
            current_capacity = self.vehicle_capacity

            # Start a new route with the unvisited customer furthest from the depot
            seed_customer = max(unvisited, key=lambda c: self.distance_matrix[self.depot, c])
            route.append(seed_customer)
            unvisited.remove(seed_customer)
            current_capacity -= self.demands[seed_customer]

            last_customer = seed_customer

            while True:
                best_candidate = None
                min_insertion_cost = float('inf')

                # Find the best customer to insert next based on a weighted cost
                for customer in unvisited:
                    if self.demands[customer] <= current_capacity:
                        # Solomon's insertion cost: c = alpha1 * d(i,j) + alpha2 * (d(0,j) - d(0,i))
                        # We use a simplified version focusing on proximity to the last customer
                        cost = self.distance_matrix[last_customer, customer]
                        if cost < min_insertion_cost:
                            min_insertion_cost = cost
                            best_candidate = customer

                if best_candidate:
                    route.append(best_candidate)
                    unvisited.remove(best_candidate)
                    current_capacity -= self.demands[best_candidate]
                    last_customer = best_candidate
                else:
                    break  # No more customers can fit

            route.append(self.depot)
            routes.append(route)

        return self._to_flat_solution(routes)

    # --- Destroy Operators ---

    def _random_removal(self, solution: list, n_remove: int) -> tuple[list, list]:
        """Removes n_remove random customers from the solution."""
        removed_customers = []
        current_solution = solution[:]

        # Get all customer nodes from the solution
        customers = [node for node in solution if node != self.depot]

        for _ in range(min(n_remove, len(customers))):
            customer_to_remove = random.choice(customers)
            current_solution.remove(customer_to_remove)
            customers.remove(customer_to_remove)
            removed_customers.append(customer_to_remove)

        # Clean up empty routes [0, 0] that might have been created
        cleaned_solution = []
        for i in range(len(current_solution)):
            if current_solution[i] == self.depot and i > 0 and current_solution[i - 1] == self.depot:
                continue
            cleaned_solution.append(current_solution[i])

        return cleaned_solution, removed_customers

    def _worst_removal(self, solution: list, n_remove: int) -> tuple[list, list]:
        """Removes n_remove customers that contribute most to the total cost."""
        costs = []
        for i in range(1, len(solution) - 1):
            if solution[i] != self.depot:
                prev_node, curr_node, next_node = solution[i - 1], solution[i], solution[i + 1]
                # Cost saving if curr_node is removed
                cost_saving = (self.distance_matrix[prev_node, curr_node] +
                               self.distance_matrix[curr_node, next_node] -
                               self.distance_matrix[prev_node, next_node])
                costs.append((cost_saving, curr_node))

        costs.sort(key=lambda x: x[0], reverse=True)

        removed_customers = []
        current_solution = solution[:]

        for _, customer in costs[:n_remove]:
            if customer in current_solution:
                current_solution.remove(customer)
                removed_customers.append(customer)

        # Clean up empty routes
        cleaned_solution = []
        for i in range(len(current_solution)):
            if current_solution[i] == self.depot and i > 0 and current_solution[i - 1] == self.depot:
                continue
            cleaned_solution.append(current_solution[i])

        return cleaned_solution, removed_customers

    def _related_removal(self, solution: list, n_remove: int) -> tuple[list, list]:
        """Removes customers that are 'related' to a randomly chosen one."""
        customers = [node for node in solution if node != self.depot]
        if not customers:
            return solution, []

        start_node = random.choice(customers)
        removed_customers = [start_node]

        # Calculate relatedness (similarity) to the start_node for all other customers
        # Relatedness metric: R(i, j) = 1 / (d(i,j) + |demand_i - demand_j|)
        relatedness = []
        for cust in customers:
            if cust != start_node:
                # Normalize distance and demand difference to avoid scaling issues
                norm_dist = self.distance_matrix[start_node, cust] / np.max(self.distance_matrix)
                norm_demand_diff = abs(self.demands[start_node] - self.demands[cust]) / self.vehicle_capacity
                # Lower value means more related
                relatedness_score = norm_dist + norm_demand_diff
                relatedness.append((relatedness_score, cust))

        relatedness.sort(key=lambda x: x[0])  # Sort by score, ascending

        for _, cust in relatedness:
            if len(removed_customers) < n_remove:
                removed_customers.append(cust)
            else:
                break

        current_solution = [node for node in solution if node not in removed_customers]

        # Clean up empty routes
        cleaned_solution = []
        for i in range(len(current_solution)):
            if current_solution[i] == self.depot and i > 0 and current_solution[i - 1] == self.depot:
                continue
            cleaned_solution.append(current_solution[i])

        return cleaned_solution, removed_customers

    # --- Repair Operators ---

    def _greedy_insertion(self, partial_solution: list, customers_to_insert: list) -> list:
        """Inserts customers one by one at the position that results in the minimum cost increase."""
        routes = self._to_routes(partial_solution)
        route_demands = [sum(self.demands[node] for node in r) for r in routes]

        for customer in customers_to_insert:
            best_insertion_cost = float('inf')
            best_insertion_pos = None  # (route_idx, position_in_route)

            for r_idx, route in enumerate(routes):
                if route_demands[r_idx] + self.demands[customer] <= self.vehicle_capacity:
                    for pos in range(1, len(route)):
                        prev_node, next_node = route[pos - 1], route[pos]
                        cost_increase = (self.distance_matrix[prev_node, customer] +
                                         self.distance_matrix[customer, next_node] -
                                         self.distance_matrix[prev_node, next_node])

                        if cost_increase < best_insertion_cost:
                            best_insertion_cost = cost_increase
                            best_insertion_pos = (r_idx, pos)

            # If no position in existing routes, check if a new route can be created
            if self.demands[customer] <= self.vehicle_capacity:
                cost_increase = self.distance_matrix[self.depot, customer] + self.distance_matrix[customer, self.depot]
                if best_insertion_pos is None or cost_increase < best_insertion_cost:
                    best_insertion_cost = cost_increase
                    best_insertion_pos = (len(routes), 1)  # New route

            if best_insertion_pos:
                r_idx, pos = best_insertion_pos
                if r_idx == len(routes):  # New route
                    routes.append([self.depot, customer, self.depot])
                    route_demands.append(self.demands[self.depot] + self.demands[customer])
                else:  # Existing route
                    routes[r_idx].insert(pos, customer)
                    route_demands[r_idx] += self.demands[customer]

        return self._to_flat_solution(routes)

    def _regret_insertion(self, partial_solution: list, customers_to_insert: list, k: int = 3) -> list:
        """Inserts customers based on the regret-k heuristic."""
        routes = self._to_routes(partial_solution)
        route_demands = [sum(self.demands[node] for node in r) for r in routes]

        uninserted = customers_to_insert[:]

        while uninserted:
            best_customer = None
            max_regret = -float('inf')
            best_insertion_pos_for_best_cust = None

            for customer in uninserted:
                insertion_costs = []
                # Find all possible insertion positions and their costs
                for r_idx, route in enumerate(routes):
                    if route_demands[r_idx] + self.demands[customer] <= self.vehicle_capacity:
                        for pos in range(1, len(route)):
                            prev_node, next_node = route[pos - 1], route[pos]
                            cost = (self.distance_matrix[prev_node, customer] +
                                    self.distance_matrix[customer, next_node] -
                                    self.distance_matrix[prev_node, next_node])
                            insertion_costs.append((cost, (r_idx, pos)))

                # Option to create a new route
                if self.demands[customer] <= self.vehicle_capacity:
                    cost = self.distance_matrix[self.depot, customer] + self.distance_matrix[customer, self.depot]
                    insertion_costs.append((cost, (len(routes), 1)))

                if not insertion_costs:
                    continue  # Cannot insert this customer

                insertion_costs.sort(key=lambda x: x[0])

                # Calculate regret: difference between the best and k-th best insertion
                regret = 0
                if len(insertion_costs) > 1:
                    k_val = min(k, len(insertion_costs))
                    for i in range(1, k_val):
                        regret += insertion_costs[i][0] - insertion_costs[0][0]

                if regret > max_regret:
                    max_regret = regret
                    best_customer = customer
                    best_insertion_pos_for_best_cust = insertion_costs[0][1]

            if best_customer:
                r_idx, pos = best_insertion_pos_for_best_cust
                if r_idx == len(routes):  # New route
                    routes.append([self.depot, best_customer, self.depot])
                    route_demands.append(self.demands[self.depot] + self.demands[best_customer])
                else:  # Existing route
                    routes[r_idx].insert(pos, best_customer)
                    route_demands[r_idx] += self.demands[best_customer]
                uninserted.remove(best_customer)
            else:
                # This should not happen if all customers can fit in an empty vehicle
                # but as a safeguard, we break to avoid an infinite loop.
                break

        return self._to_flat_solution(routes)

    # --- ALNS Core ---

    def solve(self, iterations: int = 1000, n_remove_percentage: float = 0.3,
              start_temp: float = 100, cooling_rate: float = 0.995,
              sigma1: int = 33, sigma2: int = 9, sigma3: int = 13, reaction_factor: float = 0.1) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP) using Adaptive Large Neighborhood Search.

        Args:
            iterations: The number of iterations to run the algorithm.
            n_remove_percentage: The percentage of customers to remove in each destroy step.
            start_temp: The initial temperature for simulated annealing.
            cooling_rate: The cooling rate for the temperature.
            sigma1, sigma2, sigma3: Scores for finding a new global best, a better solution, or an accepted worse solution.
            reaction_factor: The reaction factor for updating operator weights.

        Returns:
            A one-dimensional list representing the best found solution.
        """
        # 1. Initialize
        current_solution = self._generate_initial_solution()
        best_solution = current_solution
        best_cost = self._calculate_cost(best_solution)

        destroy_operators = [self._random_removal, self._worst_removal, self._related_removal]
        repair_operators = [self._greedy_insertion, self._regret_insertion]

        # Initialize operator weights and scores
        destroy_weights = np.ones(len(destroy_operators))
        repair_weights = np.ones(len(repair_operators))
        destroy_scores = np.zeros(len(destroy_operators))
        repair_scores = np.zeros(len(repair_operators))
        destroy_counts = np.zeros(len(destroy_operators))
        repair_counts = np.zeros(len(repair_operators))

        temp = start_temp
        n_remove = int((self.n_customers) * n_remove_percentage)

        for i in range(iterations):
            # 2. Select operators based on weights
            destroy_idx = np.random.choice(len(destroy_operators), p=destroy_weights / np.sum(destroy_weights))
            repair_idx = np.random.choice(len(repair_operators), p=repair_weights / np.sum(repair_weights))

            destroy_op = destroy_operators[destroy_idx]
            repair_op = repair_operators[repair_idx]

            # 3. Destroy and Repair to create a new solution
            partial_solution, removed = destroy_op(current_solution, n_remove)
            new_solution = repair_op(partial_solution, removed)
            new_cost = self._calculate_cost(new_solution)

            current_cost = self._calculate_cost(current_solution)

            # 4. Acceptance criterion (Simulated Annealing)
            score = 0
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                current_solution = new_solution
                score = sigma1
                # print(f"Iter {i}: New best solution found! Cost: {best_cost:.2f}")
            elif new_cost < current_cost:
                current_solution = new_solution
                score = sigma2
            elif random.random() < math.exp((current_cost - new_cost) / temp):
                current_solution = new_solution
                score = sigma3

            # 5. Update scores and counts for selected operators
            if score > 0:
                destroy_scores[destroy_idx] += score
                repair_scores[repair_idx] += score
            destroy_counts[destroy_idx] += 1
            repair_counts[repair_idx] += 1

            # 6. Update weights periodically
            if (i + 1) % 100 == 0:
                for j in range(len(destroy_operators)):
                    destroy_weights[j] = (1 - reaction_factor) * destroy_weights[j] + \
                                         reaction_factor * (
                                             destroy_scores[j] / destroy_counts[j] if destroy_counts[j] > 0 else 0)
                for j in range(len(repair_operators)):
                    repair_weights[j] = (1 - reaction_factor) * repair_weights[j] + \
                                        reaction_factor * (
                                            repair_scores[j] / repair_counts[j] if repair_counts[j] > 0 else 0)

                # Reset scores and counts
                destroy_scores.fill(0)
                repair_scores.fill(0)
                destroy_counts.fill(0)
                repair_counts.fill(0)

            # 7. Cool down
            temp *= cooling_rate

        # Final check for feasibility
        if not self._is_feasible(best_solution):
            print("Warning: The best solution found is not feasible.")

        return best_solution
