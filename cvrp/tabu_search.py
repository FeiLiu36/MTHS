import numpy as np
import collections
from numba import njit, types
from numba.typed import List


# ==============================================================================
# --- UTILITY & COST FUNCTIONS (UNCHANGED) ---
# ==============================================================================

@njit(cache=True)
def _calculate_solution_cost_numba(solution: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Calculates the total distance of a CVRP solution (giant tour representation)."""
    total_distance = 0.0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i], solution[i + 1]]
    return total_distance


# ==============================================================================
# --- NEIGHBORHOOD SEARCH FUNCTIONS (EXPANDED) ---
# ==============================================================================

# --- Move Type 1: Relocation (Existing, but refactored for clarity) ---
@njit(cache=True)
def _find_best_relocation_move(
        solution: np.ndarray, route_loads: np.ndarray, route_breaks: np.ndarray,
        distance_matrix: np.ndarray, demands: np.ndarray, vehicle_capacity: int
) -> tuple:
    """Finds the best relocation move. Returns (delta, move_info)."""
    best_delta = np.inf
    best_move = (-1, -1)  # (customer_node, new_position_index)
    num_routes = len(route_breaks) - 1

    for i in range(1, len(solution) - 1):
        customer_node = solution[i]
        if customer_node == 0: continue

        # Identify source route
        source_route_idx = -1
        for r in range(num_routes):
            if route_breaks[r] < i < route_breaks[r + 1]:
                source_route_idx = r
                break

        # Calculate cost change of removing the customer
        prev_node = solution[i - 1]
        next_node = solution[i + 1]
        delta_remove = distance_matrix[prev_node, next_node] - \
                       (distance_matrix[prev_node, customer_node] + distance_matrix[customer_node, next_node])

        # Iterate through all possible new positions
        for j in range(1, len(solution)):
            if i == j or i + 1 == j: continue  # Skip trivial insertions

            # Identify destination route and check capacity
            dest_route_idx = -1
            for r in range(num_routes):
                if route_breaks[r] < j <= route_breaks[r + 1]:
                    dest_route_idx = r
                    break

            is_same_route = (source_route_idx == dest_route_idx)
            if not is_same_route and route_loads[dest_route_idx] + demands[customer_node] > vehicle_capacity:
                continue

            p_prev = solution[j - 1]
            p_next = solution[j]
            delta_insert = (distance_matrix[p_prev, customer_node] + distance_matrix[customer_node, p_next]) - \
                           distance_matrix[p_prev, p_next]

            delta = delta_remove + delta_insert
            if delta < best_delta:
                best_delta = delta
                best_move = (i, j)

    return best_delta, best_move


# --- Move Type 2: Swap (NEW) ---
@njit(cache=True)
def _find_best_swap_move(
        solution: np.ndarray, route_loads: np.ndarray, route_breaks: np.ndarray,
        distance_matrix: np.ndarray, demands: np.ndarray, vehicle_capacity: int
) -> tuple:
    """Finds the best swap move. Returns (delta, move_info)."""
    best_delta = np.inf
    best_move = (-1, -1)  # (index_i, index_j)
    num_routes = len(route_breaks) - 1

    for i in range(1, len(solution) - 2):
        node1 = solution[i]
        if node1 == 0: continue

        # Find route of node1
        route1_idx = -1
        for r in range(num_routes):
            if route_breaks[r] < i < route_breaks[r + 1]:
                route1_idx = r
                break

        for j in range(i + 1, len(solution) - 1):
            node2 = solution[j]
            if node2 == 0: continue

            # Find route of node2
            route2_idx = -1
            for r in range(num_routes):
                if route_breaks[r] < j < route_breaks[r + 1]:
                    route2_idx = r
                    break

            # Capacity Check
            if route1_idx != route2_idx:
                if (route_loads[route1_idx] - demands[node1] + demands[node2] > vehicle_capacity or
                        route_loads[route2_idx] - demands[node2] + demands[node1] > vehicle_capacity):
                    continue

            # Calculate Delta Cost
            prev1, next1 = solution[i - 1], solution[i + 1]
            prev2, next2 = solution[j - 1], solution[j + 1]

            # Special case: adjacent nodes
            if j == i + 1:
                removed = distance_matrix[prev1, node1] + distance_matrix[node1, node2] + distance_matrix[node2, next2]
                added = distance_matrix[prev1, node2] + distance_matrix[node2, node1] + distance_matrix[node1, next2]
            else:
                removed = distance_matrix[prev1, node1] + distance_matrix[node1, next1] + \
                          distance_matrix[prev2, node2] + distance_matrix[node2, next2]
                added = distance_matrix[prev1, node2] + distance_matrix[node2, next1] + \
                        distance_matrix[prev2, node1] + distance_matrix[node1, next2]

            delta = added - removed
            if delta < best_delta:
                best_delta = delta
                best_move = (i, j)

    return best_delta, best_move


# --- Move Type 3: 2-Opt (NEW, adapted for intra-route) ---
@njit(cache=True)
def _find_best_2opt_move(
        solution: np.ndarray, route_breaks: np.ndarray, distance_matrix: np.ndarray
) -> tuple:
    """Finds the best intra-route 2-opt move. Returns (delta, move_info)."""
    best_delta = np.inf
    best_move = (-1, -1)  # (index_i, index_j) for reversal
    num_routes = len(route_breaks) - 1

    for r in range(num_routes):
        start_idx = route_breaks[r] + 1
        end_idx = route_breaks[r + 1] - 1
        route_len = end_idx - start_idx + 1
        if route_len < 2: continue

        for i_rel in range(route_len - 1):
            for j_rel in range(i_rel + 1, route_len):
                i = start_idx + i_rel
                j = start_idx + j_rel

                node_i_prev = solution[i - 1]
                node_i = solution[i]
                node_j = solution[j]
                node_j_next = solution[j + 1]

                removed = distance_matrix[node_i_prev, node_i] + distance_matrix[node_j, node_j_next]
                added = distance_matrix[node_i_prev, node_j] + distance_matrix[node_i, node_j_next]
                delta = added - removed

                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)

    return best_delta, best_move


# --- Master Neighborhood Search Function ---
@njit(cache=True)
def _find_best_neighbor_numba(
        solution: np.ndarray, current_cost: float, distance_matrix: np.ndarray,
        demands: np.ndarray, vehicle_capacity: int, tabu_list: np.ndarray,
        best_overall_cost: float, iteration: int
) -> tuple:
    """
    Searches multiple neighborhoods (Relocate, Swap, 2-Opt) and returns the best valid move.
    """
    # Pre-calculate route boundaries and loads
    route_breaks = np.where(solution == 0)[0]
    num_routes = len(route_breaks) - 1
    route_loads = np.zeros(num_routes)
    for r in range(num_routes):
        start, end = route_breaks[r], route_breaks[r + 1]
        for i in range(start + 1, end):
            route_loads[r] += demands[solution[i]]

    # --- Search Neighborhoods ---
    delta_relocate, move_relocate = _find_best_relocation_move(solution, route_loads, route_breaks, distance_matrix,
                                                               demands, vehicle_capacity)
    delta_swap, move_swap = _find_best_swap_move(solution, route_loads, route_breaks, distance_matrix, demands,
                                                 vehicle_capacity)
    delta_2opt, move_2opt = _find_best_2opt_move(solution, route_breaks, distance_matrix)

    # --- Select Best Move, Considering Tabu and Aspiration ---
    best_delta = np.inf
    best_move = (-1, -1)
    best_move_type = ""  # "relocate", "swap", "2opt"

    moves = [
        ("relocate", delta_relocate, move_relocate),
        ("swap", delta_swap, move_swap),
        ("2opt", delta_2opt, move_2opt)
    ]

    for move_type, delta, move in moves:
        if move[0] == -1: continue  # Invalid move from the search function

        # Determine which customers are involved for the tabu check
        nodes_involved = []
        if move_type == "relocate":
            nodes_involved.append(solution[move[0]])
        elif move_type == "swap":
            nodes_involved.append(solution[move[0]])
            nodes_involved.append(solution[move[1]])
        elif move_type == "2opt":  # For 2-opt, we make the endpoints of the reversed segment tabu
            nodes_involved.append(solution[move[0]])
            nodes_involved.append(solution[move[1]])

        is_tabu = False
        for node in nodes_involved:
            if tabu_list[node] > iteration:
                is_tabu = True
                break

        # Aspiration Criterion
        if (current_cost + delta) < best_overall_cost:
            if delta < best_delta:
                best_delta = delta
                best_move = move
                best_move_type = move_type
        # Standard non-tabu move
        elif not is_tabu:
            if delta < best_delta:
                best_delta = delta
                best_move = move
                best_move_type = move_type

    return best_move, best_delta, best_move_type


# ==============================================================================
# --- CVRP SOLVER CLASS (UPDATED) ---
# ==============================================================================

class CVRPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: list, vehicle_capacity: int):
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.num_nodes = len(coordinates)
        if self.num_nodes == 0:
            raise ValueError("Input coordinates cannot be empty.")
        if self.demands[0] != 0:
            print("Warning: Demand for the depot (node 0) should be 0.")

    def _generate_initial_solution(self) -> np.ndarray:
        """
        Generates an initial solution using a parallel savings-like heuristic.
        A bit better than pure random sequential assignment.
        """
        customers = list(range(1, self.num_nodes))
        np.random.shuffle(customers)

        routes = []
        for cust in customers:
            routes.append([0, cust, 0])

        while True:
            best_merge = None
            max_savings = -np.inf

            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    route1 = routes[i]
                    route2 = routes[j]

                    # Try merging route2 into route1
                    # Merge point: end of route1 (before depot) and start of route2 (after depot)
                    node1_end = route1[-2]
                    node2_start = route2[1]

                    load1 = sum(self.demands[c] for c in route1)
                    load2 = sum(self.demands[c] for c in route2)

                    if load1 + load2 <= self.vehicle_capacity:
                        # Clarke-Wright savings formula
                        savings = self.distance_matrix[node1_end, 0] + self.distance_matrix[0, node2_start] - \
                                  self.distance_matrix[node1_end, node2_start]
                        if savings > max_savings:
                            max_savings = savings
                            best_merge = (i, j)

            if best_merge is not None:
                i, j = best_merge
                # Merge routes[j] into routes[i]
                routes[i] = routes[i][:-1] + routes[j][1:]
                routes.pop(j)
            else:
                break  # No more profitable merges found

        # Flatten the list of routes into the giant tour format
        solution = []
        for route in routes:
            solution.extend(route[:-1])
        solution.append(0)

        return np.array(solution)

    def _apply_move(self, solution: np.ndarray, move: tuple, move_type: str) -> np.ndarray:
        """Applies a move based on its type."""
        sol_list = list(solution)

        if move_type == "relocate":
            source_idx, dest_idx = move
            customer_node = sol_list.pop(source_idx)
            if source_idx < dest_idx: dest_idx -= 1
            sol_list.insert(dest_idx, customer_node)

        elif move_type == "swap":
            i, j = move
            sol_list[i], sol_list[j] = sol_list[j], sol_list[i]

        elif move_type == "2opt":
            i, j = move
            segment = sol_list[i: j + 1]
            sol_list[i: j + 1] = segment[::-1]

        return np.array(sol_list)

    def solve(self, max_iterations: int = 10000, tabu_tenure: int = 20, verbose: bool = False) -> list:
        if self.num_nodes <= 1: return [0, 0] if self.num_nodes == 1 else []

        # --- Initialization ---
        current_solution = self._generate_initial_solution()
        current_cost = _calculate_solution_cost_numba(current_solution, self.distance_matrix)

        best_solution = current_solution.copy()
        best_cost = current_cost

        tabu_list = np.zeros(self.num_nodes, dtype=np.int32)

        if verbose:
            print(f"Initial solution cost: {current_cost:.2f}")

        # --- Tabu Search Main Loop ---
        for iteration in range(max_iterations):
            best_move, best_delta, best_move_type = _find_best_neighbor_numba(
                current_solution, current_cost, self.distance_matrix, self.demands,
                self.vehicle_capacity, tabu_list, best_cost, iteration
            )

            if not best_move_type:
                if verbose: print(f"Iter {iteration + 1}: No improving moves found. Stopping.")
                break

            # --- Apply the move and update state ---
            current_solution = self._apply_move(current_solution, best_move, best_move_type)
            current_cost += best_delta

            # --- Update Tabu List ---
            nodes_to_make_tabu = []
            if best_move_type == "relocate":
                nodes_to_make_tabu.append(current_solution[best_move[1]])  # Node at new position
            elif best_move_type == "swap":
                nodes_to_make_tabu.append(current_solution[best_move[0]])
                nodes_to_make_tabu.append(current_solution[best_move[1]])
            elif best_move_type == "2opt":
                nodes_to_make_tabu.append(current_solution[best_move[0]])
                nodes_to_make_tabu.append(current_solution[best_move[1]])

            for node in nodes_to_make_tabu:
                tabu_list[node] = iteration + tabu_tenure

            # --- Update Best Solution Found ---
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                if verbose:
                    print(f"Iter {iteration + 1}: New best cost: {best_cost:.2f} (via {best_move_type})")

        if verbose:
            print(f"\nSearch complete. Final best cost: {best_cost:.2f}")
            recalculated_cost = _calculate_solution_cost_numba(best_solution, self.distance_matrix)
            print(f"Recalculated best cost for verification: {recalculated_cost:.2f}")
            assert np.isclose(best_cost, recalculated_cost), "Final cost mismatch!"

        return best_solution.tolist()
