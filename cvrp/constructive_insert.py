import numpy as np


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
        self.n_nodes = len(coordinates)

    # --- your code here ---

    def _calculate_insertion_cost(self, u, route):
        """
        Calculates the minimum cost to insert customer u into a given route.

        Returns:
            A tuple (min_cost, best_pos) where min_cost is the increase in distance
            and best_pos is the index where the customer should be inserted.
            Returns (float('inf'), -1) if insertion is not possible.
        """
        min_cost = float('inf')
        best_pos = -1

        for i in range(len(route) - 1):
            pred_node = route[i]
            succ_node = route[i + 1]

            # Cost = dist(i, u) + dist(u, j) - dist(i, j)
            cost = (self.distance_matrix[pred_node, u] +
                    self.distance_matrix[u, succ_node] -
                    self.distance_matrix[pred_node, succ_node])

            if cost < min_cost:
                min_cost = cost
                # We insert *after* index i, so the position is i + 1
                best_pos = i + 1

        return min_cost, best_pos

    def solve(self) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP) using a cheapest insertion heuristic.

        Returns:
            A one-dimensional list of integers representing the sequence of nodes visited by all vehicles.
            The depot (node 0) is used to separate different vehicle routes and appears at the start and end
            of each route. For example: [0, 1, 4, 0, 2, 3, 0] represents:
              - Route 1: 0 → 1 → 4 → 0
              - Route 2: 0 → 2 → 3 → 0
        """
        # List of customer nodes to be visited (all nodes except the depot)
        unvisited_customers = list(range(1, self.n_nodes))

        # Initialize routes and their current loads
        routes = []
        route_loads = []

        # Start with a single empty route if there are customers to visit
        if unvisited_customers:
            routes.append([0, 0])
            route_loads.append(0)

        while unvisited_customers:
            best_insertion_cost = float('inf')
            best_customer = -1
            best_route_idx = -1
            best_position = -1

            # Iterate through all unvisited customers to find the best one to insert
            for customer_u in unvisited_customers:
                # Iterate through all existing routes
                for r_idx, route in enumerate(routes):
                    # Check if adding the customer exceeds vehicle capacity
                    if route_loads[r_idx] + self.demands[customer_u] <= self.vehicle_capacity:
                        # Find the cheapest place to insert this customer in this route
                        cost, pos = self._calculate_insertion_cost(customer_u, route)

                        if cost < best_insertion_cost:
                            best_insertion_cost = cost
                            best_customer = customer_u
                            best_route_idx = r_idx
                            best_position = pos

            # If a valid insertion was found, perform it
            if best_customer != -1:
                # Insert the customer into the best found position in the best route
                routes[best_route_idx].insert(best_position, best_customer)
                # Update the load of that route
                route_loads[best_route_idx] += self.demands[best_customer]
                # Remove the customer from the list of unvisited nodes
                unvisited_customers.remove(best_customer)
            else:
                # If no customer could be inserted (due to capacity), start a new route.
                # A good heuristic is to start a new route with the unvisited customer
                # that is farthest from the depot.
                farthest_customer = -1
                max_dist = -1
                for cust in unvisited_customers:
                    if self.distance_matrix[0, cust] > max_dist:
                        max_dist = self.distance_matrix[0, cust]
                        farthest_customer = cust

                # Create a new route for this customer
                new_route = [0, farthest_customer, 0]
                routes.append(new_route)
                route_loads.append(self.demands[farthest_customer])
                unvisited_customers.remove(farthest_customer)

        # --- Flatten the list of routes into the required format ---
        final_solution = []
        for route in routes:
            # Add all nodes from the current route
            final_solution.extend(route)
            # Remove the leading depot (0) of the next route to avoid duplicates like [..., 0, 0, ...]
            if final_solution:
                final_solution.pop()

                # Add the final closing depot if the list is not empty
        if routes:
            final_solution.append(0)

        # Handle edge case of no customers
        if not final_solution and self.n_nodes > 0:
            return [0, 0]
        elif not final_solution:
            return []

        return final_solution

