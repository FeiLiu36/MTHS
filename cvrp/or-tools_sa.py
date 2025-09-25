import numpy as np
# You must install ortools first: pip install ortools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


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
        self.num_locations = len(coordinates)
        #print(self.num_locations)
        # For CVRP, the number of vehicles is a variable, but we can set an upper bound.
        # A reasonable upper bound is the number of locations (worst case: one vehicle per location).
        self.num_vehicles = self.num_locations

    def _create_data_model(self) -> dict:
        """Creates the data model for the routing problem."""
        data = {}

        # OR-Tools works with integers. We scale numerical values to maintain precision.
        # Using a single, larger scaling factor is often a good practice.
        scaling_factor = 1000

        # Scale distances
        data['distance_matrix'] = (self.distance_matrix * scaling_factor).astype(int).tolist()

        # --- FIX: Scale demands and capacity to integers ---
        data['demands'] = [int(d * scaling_factor) for d in self.demands]
        data['vehicle_capacities'] = [int(self.vehicle_capacity * scaling_factor)] * self.num_vehicles

        data['num_vehicles'] = self.num_vehicles
        data['depot'] = 0  # The starting and ending point of all routes
        return data

    def solve(self) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP).

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
        # --- Step 1: Create the data model ---
        data = self._create_data_model()

        # --- Step 2: Create the routing index manager ---
        manager = pywrapcp.RoutingIndexManager(
            self.num_locations,
            data['num_vehicles'],
            data['depot']
        )

        # --- Step 3: Create the routing model ---
        routing = pywrapcp.RoutingModel(manager)

        # --- Step 4: Define the cost function (distance callback) ---
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # --- Step 5: Add Capacity Constraints ---
        def demand_callback(from_index):
            """Returns the demand of the node."""
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )

        # --- Step 6: Set search parameters ---
        # Use a good heuristic to find an initial solution and then improve it.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        )
        search_parameters.time_limit.seconds = 60  # Set a time limit for the search
        search_parameters.log_search = False

        # --- Step 7: Solve the problem ---
        solution = routing.SolveWithParameters(search_parameters)

        # --- Step 8: Extract and return the solution ---
        if solution:
            return self._get_routes_from_solution(manager, routing, solution)
        else:
            # It's better to log or handle this case than just print
            print('No solution found for the given constraints.')
            return []

    def _get_routes_from_solution(self, manager, routing, solution) -> list:
        solution_path = []
        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)

            if not routing.IsEnd(solution.Value(routing.NextVar(index))):
                # This is a valid, used route.
                route_nodes = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_nodes.append(node_index)
                    index = solution.Value(routing.NextVar(index))

                route_nodes.append(manager.IndexToNode(index))  # Add final depot

                if not solution_path:
                    solution_path.extend(route_nodes)
                else:
                    solution_path.extend(route_nodes[1:])

        return solution_path

