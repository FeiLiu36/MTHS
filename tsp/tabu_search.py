import numpy as np
import collections
from numba import njit, types
from numba.typed import List


# --- Numba JIT Compiled Functions ---
# These functions are computationally intensive and benefit from JIT compilation.
# They must be defined at the top level, not inside the class.

@njit(cache=True)
def _calculate_tour_distance_numba(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Calculates the total distance of a given tour."""
    num_cities = len(tour)
    total_distance = 0.0
    for i in range(num_cities):
        # Correctly handles the wrap-around from the last city to the first
        total_distance += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
    return total_distance


# --- CORRECTED AND REFACTORED VERSION ---
@njit(cache=True)
def _find_best_neighbor_numba(
        current_tour: np.ndarray,
        current_distance: float,
        distance_matrix: np.ndarray,
        tabu_list_tuples: list,
        best_overall_distance: float
) -> tuple:
    """
    Finds the best neighbor using a 2-opt swap, considering the tabu list and aspiration criterion.
    This version uses O(1) delta evaluation and efficient neighbor selection.
    """
    num_cities = len(current_tour)
    best_delta = np.inf
    best_move = (-1, -1)
    aspiration_activated = False

    # Iterate through all unique pairs of indices (i, j) where i < j
    # This defines the segment tour[i:j+1] to be reversed.
    for i in range(num_cities - 1):
        for j in range(i + 1, num_cities):

            # ===================================================================
            # ===== CRITICAL FIX: CORRECT O(1) DELTA CALCULATION FOR REVERSAL =====
            # ===================================================================
            # A 2-opt reversal of segment tour[i:j+1] breaks the edges
            # (tour[i-1], tour[i]) and (tour[j], tour[j+1])
            # and creates the new edges
            # (tour[i-1], tour[j]) and (tour[i], tour[j+1]).

            # Get the city IDs for the nodes involved in the swap.
            # Numba handles negative indices correctly (e.g., tour[-1] is the last element)
            node_i_prev = current_tour[i - 1]
            node_i = current_tour[i]
            node_j = current_tour[j]
            node_j_next = current_tour[(j + 1) % num_cities]

            # Handle the edge case where j+1 wraps around to become i.
            # This happens if the reversed segment is almost the whole tour.
            # In this case, the move is invalid as it doesn't change the tour.
            if node_j_next == node_i:
                continue

            # Calculate the change in distance (delta)
            removed_len = distance_matrix[node_i_prev, node_i] + distance_matrix[node_j, node_j_next]
            added_len = distance_matrix[node_i_prev, node_j] + distance_matrix[node_i, node_j_next]
            delta = added_len - removed_len
            # ===================================================================
            # ======================= END OF CRITICAL FIX =======================
            # ===================================================================

            # The move is defined by the indices of the tour array (i, j).
            # Since our loops ensure i < j, this is already the canonical form.
            canonical_move = (i, j)

            is_tabu = False
            for tabu_move in tabu_list_tuples:
                if canonical_move == tabu_move:
                    is_tabu = True
                    break

            # Aspiration Criterion: If the move leads to a new overall best solution,
            # we take it even if it's tabu.
            if (current_distance + delta) < best_overall_distance:
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
                    aspiration_activated = True

            # Standard non-tabu move selection
            elif not is_tabu:
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
                    aspiration_activated = False  # This move is not via aspiration

    return best_move, best_delta, aspiration_activated


class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.num_cities = len(coordinates)
        if self.num_cities == 0:
            raise ValueError("Input coordinates cannot be empty.")

    def _calculate_tour_distance(self, tour: np.ndarray) -> float:
        """Wrapper for the Numba-compiled distance calculation function."""
        return _calculate_tour_distance_numba(tour, self.distance_matrix)

    def _generate_initial_solution(self) -> np.ndarray:
        """Generates a random initial tour."""
        return np.random.permutation(self.num_cities)

    def _apply_2_opt_swap(self, tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """Applies a 2-opt swap (reversal) to a tour between indices i and j."""
        new_tour = tour.copy()
        # The segment to be reversed is from index i to j, inclusive.
        segment = new_tour[i: j + 1]
        new_tour[i: j + 1] = segment[::-1]
        return new_tour

    def solve(self, max_iterations: int = 1000, tabu_tenure: int = 20, verbose: bool = False) -> tuple:
        """
        Solve the TSP using Tabu Search.
        """
        if self.num_cities < 4:
            if verbose:
                print("Too few cities for a meaningful 2-opt search. Returning initial tour.")
            tour = np.arange(self.num_cities)
            return tour, self._calculate_tour_distance(tour)

        current_tour = self._generate_initial_solution()
        current_distance = self._calculate_tour_distance(current_tour)

        best_tour = current_tour.copy()
        best_distance = current_distance

        # Using a deque is efficient for managing the tabu list
        tabu_list = collections.deque(maxlen=tabu_tenure)

        if verbose:
            print(f"Initial random tour distance: {current_distance:.2f}")

        for iteration in range(max_iterations):
            # Convert deque to Numba typed list for the JIT function
            tuple_type = types.UniTuple(types.int64, 2)
            tabu_list_tuples = List.empty_list(tuple_type)
            for move in tabu_list:
                tabu_list_tuples.append(move)

            best_move, best_delta, aspiration = _find_best_neighbor_numba(
                current_tour, current_distance, self.distance_matrix, tabu_list_tuples, best_distance
            )

            # If no valid move is found (e.g., all are tabu and none meet aspiration)
            if best_move[0] == -1:
                if verbose:
                    print(f"Iter {iteration + 1}: No non-tabu improving moves found. Stopping search.")
                break

            # Apply the best move found
            i, j = best_move
            current_tour = self._apply_2_opt_swap(current_tour, i, j)

            # Update distance using the calculated delta - much faster than recalculating
            current_distance += best_delta

            # Add the move that led to this state to the tabu list
            tabu_list.append(best_move)

            # Update the overall best solution if the new one is better
            if current_distance < best_distance:
                best_tour = current_tour.copy()
                best_distance = current_distance
                if verbose:
                    status = " (Aspiration)" if aspiration else ""
                    print(f"Iter {iteration + 1}: New best distance: {best_distance:.2f}{status}")

        if verbose:
            print(f"\nSearch complete. Final best distance: {best_distance:.2f}")
            # Final check to ensure delta calculation was correct throughout
            recalculated_dist = self._calculate_tour_distance(best_tour)
            print(f"Recalculated best distance for verification: {recalculated_dist:.2f}")
            assert np.isclose(best_distance, recalculated_dist), "Final distance mismatch!"
            print("Final tour is valid and distance is correct.")

        return best_tour

# import numpy as np
# import collections
# from numba import njit, types
# from numba.typed import List
#
#
# # --- Numba JIT Compiled Functions ---
# # These functions are computationally intensive and benefit from JIT compilation.
# # They must be defined at the top level, not inside the class.
#
# @njit(cache=True)
# def _calculate_tour_distance_numba(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
#     """Calculates the total distance of a given tour."""
#     num_cities = len(tour)
#     total_distance = 0.0
#     for i in range(num_cities):
#         # Distance from the current city to the next one in the tour
#         # The modulo operator handles the wrap-around from the last city to the first
#         total_distance += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
#     return total_distance
#
#
# @njit(cache=True)
# def _find_best_neighbor_numba(
#         current_tour: np.ndarray,
#         distance_matrix: np.ndarray,
#         tabu_list_tuples: list,  # This will now be a Numba typed list
#         best_distance: float
# ) -> tuple:
#     """
#     Finds the best neighbor using a 2-opt swap, considering the tabu list and aspiration criterion.
#     This is a JIT-compiled version for performance.
#     """
#     num_cities = len(current_tour)
#     best_neighbor = np.empty_like(current_tour)
#     best_neighbor_distance = np.inf
#     best_move = (-1, -1)  # Use a tuple for Numba compatibility
#
#     # Explore Neighborhood (using 2-opt)
#     for i in range(num_cities):
#         for j in range(i + 1, num_cities):
#             # Create the neighbor tour by performing the 2-opt swap
#             neighbor_tour = current_tour.copy()
#             # Reverse the segment between i and j
#             segment = neighbor_tour[i:j + 1]
#             neighbor_tour[i:j + 1] = segment[::-1]
#
#             neighbor_distance = _calculate_tour_distance_numba(neighbor_tour, distance_matrix)
#
#             # Use a sorted tuple for the move to ensure order doesn't matter
#             # Note: sorting inside the loop is okay, but for maximum performance,
#             # one could ensure i is always less than j by swapping them if needed.
#             # Here, the range `j in range(i + 1, ...)` already ensures i < j.
#             move = (i, j)
#
#             # Check if the move is NOT in the tabu list
#             is_tabu = False
#             for tabu_move in tabu_list_tuples:
#                 if move == tabu_move:
#                     is_tabu = True
#                     break
#
#             if not is_tabu:
#                 # If this is the best neighbor found so far in this iteration
#                 if neighbor_distance < best_neighbor_distance:
#                     best_neighbor = neighbor_tour
#                     best_neighbor_distance = neighbor_distance
#                     best_move = move
#             else:
#                 # Aspiration Criterion: If the tabu move leads to a new overall best solution,
#                 # we override the tabu status and accept it.
#                 if neighbor_distance < best_distance:
#                     best_neighbor = neighbor_tour
#                     best_neighbor_distance = neighbor_distance
#                     best_move = move
#                     # Take this move immediately
#                     return best_neighbor, best_neighbor_distance, best_move
#
#     return best_neighbor, best_neighbor_distance, best_move
#
#
# class TSPSolver:
#     def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
#         """
#         Initialize the TSP solver.
#
#         Args:
#             coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
#             distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
#         """
#         self.coordinates = coordinates
#         self.distance_matrix = distance_matrix
#         self.num_cities = len(coordinates)
#
#     # --- your code here ---
#
#     def _calculate_tour_distance(self, tour: np.ndarray) -> float:
#         """Calculates the total distance of a given tour. (Wrapper for Numba version)"""
#         return _calculate_tour_distance_numba(tour, self.distance_matrix)
#
#     def _generate_initial_solution(self) -> np.ndarray:
#         """Generates a random initial tour."""
#         # A random permutation of cities is a good starting point
#         return np.random.permutation(self.num_cities)
#
#     def _find_best_neighbor(self, current_tour: np.ndarray, tabu_list: collections.deque,
#                             best_distance: float) -> tuple:
#         """
#         Neighborhood search function. Finds the best 2-opt neighbor.
#         (Wrapper for Numba version)
#         """
#         # --- START: MODIFIED SECTION ---
#         # Define the type of the elements in the list for Numba
#         # It's a list of tuples, where each tuple contains two 64-bit integers.
#         tuple_type = types.UniTuple(types.int64, 2)
#
#         # Create a Numba typed list. This works even if the input is empty.
#         tabu_list_tuples = List.empty_list(tuple_type)
#
#         # Populate the typed list from the deque
#         for move in tabu_list:
#             # Ensure the tuple is sorted for consistent representation
#             tabu_list_tuples.append(tuple(sorted(move)))
#         # --- END: MODIFIED SECTION ---
#
#         neighbor, dist, move = _find_best_neighbor_numba(
#             current_tour, self.distance_matrix, tabu_list_tuples, best_distance
#         )
#
#         # Convert move back to frozenset for consistency with the main loop's logic
#         best_move_frozenset = frozenset(move) if move[0] != -1 else None
#
#         return neighbor, dist, best_move_frozenset
#
#     def solve(self, max_iterations: int = 1000, tabu_tenure: int = 20) -> np.ndarray:
#         """
#         Solve the Traveling Salesman Problem (TSP) using Tabu Search.
#
#         Args:
#             max_iterations: The maximum number of iterations to run the search.
#             tabu_tenure: The number of iterations a move (swap) remains in the tabu list.
#
#         Returns:
#             A numpy array of shape (n,) containing a permutation of integers
#             [0, 1, ..., n-1] representing the order in which the cities are visited.
#         """
#         # 1. Initialization
#         current_tour = self._generate_initial_solution()
#         current_distance = self._calculate_tour_distance(current_tour)
#
#         best_tour = current_tour
#         best_distance = current_distance
#
#         # The tabu list will store "moves". A move is defined by the pair of indices (i, j)
#         # that were swapped. We use a deque for efficient addition and removal.
#         # A move is stored as a frozenset to be order-independent, i.e., (i, j) is the same as (j, i).
#         tabu_list = collections.deque(maxlen=tabu_tenure)
#
#         # print(f"Initial random tour distance: {best_distance:.2f}")
#
#         # 2. Main Search Loop
#         for iteration in range(max_iterations):
#
#             # 3. Explore Neighborhood (using the extracted function)
#             best_neighbor, best_neighbor_distance, best_move = self._find_best_neighbor(
#                 current_tour, tabu_list, best_distance
#             )
#
#             # 4. Move to the best neighbor found
#             if best_move is None:
#                 # This can happen if all possible moves are tabu and none meet the aspiration criterion.
#                 # It's rare with a reasonably sized tabu list but good to handle.
#                 print("No valid non-tabu moves found. Stopping search.")
#                 break
#
#             current_tour = best_neighbor
#             current_distance = best_neighbor_distance
#
#             # 5. Update Tabu List and Best Solution
#             # Add the accepted move to the tabu list
#             tabu_list.append(best_move)
#
#             # If the current tour is the best one found so far in the entire search
#             if current_distance < best_distance:
#                 # Check if aspiration criterion was met (move was already tabu)
#                 # This is just for logging/understanding the search process
#                 # if best_move in list(tabu_list)[:-1]:
#                 #     print(f"  (Aspiration criterion met at iteration {iteration+1})")
#
#                 best_tour = current_tour
#                 best_distance = current_distance
#                 # print(f"Iteration {iteration+1}: New best distance = {best_distance:.2f}")
#
#         # print(f"\nSearch finished. Best tour distance: {best_distance:.2f}")
#         return best_tour
