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
        self.num_nodes = len(coordinates)

    # --- your code here ---

    def solve(self) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP) using a nearest neighbor greedy heuristic.

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
        # --- your code here ---

        # Set of unvisited customer nodes (nodes 1 to n-1)
        unvisited_nodes = set(range(1, self.num_nodes))
        solution = []

        # Continue until all customer nodes have been visited
        while unvisited_nodes:
            # Start a new route from the depot
            current_node = 0
            current_capacity = self.vehicle_capacity
            current_route = [0]

            while True:
                # Find the nearest, valid, unvisited neighbor
                best_neighbor = -1
                min_distance = float('inf')

                # Iterate through all unvisited nodes to find the best candidate
                for neighbor in unvisited_nodes:
                    # Check if the vehicle has enough capacity for this neighbor's demand
                    if self.demands[neighbor] <= current_capacity:
                        distance = self.distance_matrix[current_node][neighbor]
                        if distance < min_distance:
                            min_distance = distance
                            best_neighbor = neighbor

                # If a valid neighbor was found
                if best_neighbor != -1:
                    # Add the neighbor to the current route
                    current_route.append(best_neighbor)
                    # Update vehicle capacity
                    current_capacity -= self.demands[best_neighbor]
                    # Update current location
                    current_node = best_neighbor
                    # Remove the visited node from the unvisited set
                    unvisited_nodes.remove(best_neighbor)
                else:
                    # No valid neighbor found (either due to capacity or all nodes visited)
                    # End the current route by returning to the depot
                    break

            # Add the completed route (ending with a return to the depot) to the overall solution
            current_route.append(0)
            solution.extend(current_route)

        return solution

