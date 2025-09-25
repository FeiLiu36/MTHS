import numpy as np
# You must install ortools first: pip install ortools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


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
        self.num_locations = len(coordinates)

    def _create_data_model(self) -> dict:
        """Creates the data model for the routing problem."""
        data = {}
        # Example of scaling the distance matrix
        # Do this before creating the data model
        scaling_factor = 1000
        int_distance_matrix = (self.distance_matrix * scaling_factor).astype(int)

        # Then in _create_data_model:
        data['distance_matrix'] = int_distance_matrix.tolist()
        #
        # data['distance_matrix'] = self.distance_matrix.tolist()
        data['num_vehicles'] = 1  # TSP has one "salesperson"
        data['depot'] = 0  # The starting and ending point of the tour
        return data

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using OR-Tools heuristics.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        # --- Step 1: Create the data model ---
        data = self._create_data_model()

        # --- Step 2: Create the routing index manager ---
        # The manager handles the conversion between OR-Tools' internal node indices
        # and our problem's location indices (0 to n-1).
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
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        # Register the callback with the routing model.
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # --- Step 5: Set search parameters ---
        # This determines the heuristic used to find the first solution.
        # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        # search_parameters.first_solution_strategy = (
        #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        # )
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
        search_parameters.time_limit.seconds = 60
        search_parameters.log_search = False
        # You can also set a time limit for the solver
        # search_parameters.time_limit.FromSeconds(30)

        # --- Step 6: Solve the problem ---
        solution = routing.SolveWithParameters(search_parameters)

        # --- Step 7: Extract and return the solution ---
        if solution:
            tour = self._get_tour_from_solution(manager, routing, solution)
            return tour
        else:
            print('No solution found !')
            # Fallback to a naive tour if no solution is found
            return np.arange(self.num_locations)

    def _get_tour_from_solution(self, manager, routing, solution) -> np.ndarray:
        """
        Extracts the tour from the OR-Tools solution object.
        """
        tour = []
        index = routing.Start(0)  # Get the starting index for vehicle 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            tour.append(node_index)
            # CORRECTED LINE: Use solution.Value() to get the next node in the solved route
            index = solution.Value(routing.NextVar(index))

        # The loop terminates when `index` is the end node.
        # The problem asks for a permutation of 0 to n-1, and this tour
        # correctly represents that permutation.
        # e.g., for 4 cities, the tour will be [0, 2, 1, 3]. The path implicitly
        # returns from 3 to 0.
        return np.array(tour)


