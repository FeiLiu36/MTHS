# File: cvrp_evaluation.py

from __future__ import annotations

import os
import sys
import time
import pickle as pkl
import csv
import importlib
import traceback
import concurrent.futures
from typing import List, Dict, Any, Tuple

import numpy as np

# It's highly recommended to install tqdm for a nice progress bar
# pip install tqdm
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: 'tqdm' library not found. Progress bar will not be shown.")
    print("Install it with: pip install tqdm")


    # Create a dummy tqdm function if it's not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


# --- Mocking the base class for standalone execution ---
class Evaluation:
    def __init__(self, **kwargs):
        pass


# --- End of Mock ---

__all__ = ['CVRPEvaluation']


class CVRPEvaluation(Evaluation):
    """
    Evaluator for the Capacitated Vehicle Routing Problem (CVRP).
    It can evaluate multiple solvers on a dataset of CVRP instances in parallel
    and write the results to a CSV file.
    """

    def __init__(self,
                 solver_modules: List[str],
                 num_threads: int = 4,
                 output_csv_path: str = 'cvrp_results.csv',
                 dataset_path: str = "./cvrp_instances_train.pkl",
                 timeout_seconds: int = 30,
                 is_scale: bool = False,
                 **kwargs):
        """
        Args:
            solver_modules (List[str]): A list of solver module names to evaluate (e.g., ['my_cvrp_solver']).
            num_threads (int): The number of threads to use for parallel evaluation.
            output_csv_path (str): Path to write the final results CSV file.
            dataset_path (str): Path to the CVRP instances pickle file.
            timeout_seconds (int): Timeout for each solver run (Note: not strictly enforced by ThreadPoolExecutor).
        """
        super().__init__(**kwargs)

        self.solver_modules = solver_modules
        self.num_threads = num_threads
        self.output_csv_path = output_csv_path
        self.timeout_seconds = timeout_seconds  # Stored for potential future use with ProcessPoolExecutor
        self.is_scale = is_scale

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self._datasets = pkl.load(f)

        print(f"Loaded {len(self._datasets)} CVRP instances.")
        print(f"Solvers to be evaluated: {self.solver_modules}")
        print(f"Running evaluation with {self.num_threads} threads.")

    def _normalize_and_update(self, coordinates: np.ndarray, distance_matrix: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Normalizes coordinates based on the max range of x or y, and updates the distance matrix.

        Args:
            coordinates (np.ndarray): The original coordinates.
            distance_matrix (np.ndarray): The original distance matrix (can be used as a flag or ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the normalized coordinates
                                           and the updated distance matrix.
        """
        # Find the min and max for both x and y coordinates
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Calculate the scaling factor
        scale_factor = max(x_max - x_min, y_max - y_min)

        # Avoid division by zero if all points are the same
        if scale_factor < 1e-9:
            return coordinates, distance_matrix

        # Normalize coordinates
        normalized_coords = coordinates.copy()
        normalized_coords[:, 0] = (x_coords - x_min) / scale_factor
        normalized_coords[:, 1] = (y_coords - y_min) / scale_factor

        # Recalculate the distance matrix based on normalized coordinates
        num_locations = len(normalized_coords)
        updated_dist_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(i + 1, num_locations):
                dist = np.linalg.norm(normalized_coords[i] - normalized_coords[j])
                updated_dist_matrix[i, j] = dist
                updated_dist_matrix[j, i] = dist

        return normalized_coords, updated_dist_matrix

    def calculate_total_cost(self, coordinates: np.ndarray, solution: List[int]) -> float:
        """
        Calculates the total cost of a CVRP solution given in flattened format.
        A flattened solution is a single list where routes are concatenated and
        separated by the depot (0), e.g., [0, 1, 2, 0, 3, 4, 0].
        """
        cost = 0.0
        if not solution:
            return 0.0

        # The total cost is the sum of distances between consecutive points in the flattened list
        for i in range(len(solution) - 1):
            u, v = solution[i], solution[i + 1]
            cost += np.linalg.norm(coordinates[u] - coordinates[v])
        return cost

    def check_feasibility(self, solution: List[int], demands: List[float], vehicle_capacity: float) -> bool:
        """
        Checks if the given CVRP solution is feasible.
        Assumes a flattened list format where '0' represents the depot.
        """
        if not isinstance(solution, list) or not solution:
            return False
        if solution[0] != 0 or solution[-1] != 0:
            print("solution must start and end at the depot")
            # Solution must start and end at the depot
            return False

        current_capacity = 0.0
        visited_customers = set()

        for node in solution:
            if node == 0:
                # A new route starts, reset capacity. The previous route ending at 0 is implicitly valid.
                current_capacity = 0.0
            else:
                # Check for duplicate customer visits
                if node in visited_customers:
                    print("dubplicate customer visits")
                    return False
                visited_customers.add(node)

                # Add demand and check capacity
                current_capacity += demands[node]
                if current_capacity > vehicle_capacity + 1e-6:  # Use tolerance for float comparison
                    print(f"values = {current_capacity},{vehicle_capacity}")
                    print("capacity exceeded")
                    return False

        # Ensure all customers (nodes 1 to N-1) are visited exactly once
        num_customers = len(demands) - 1
        if len(visited_customers) != num_customers:
            print("demands does not have enough customers")
            return False

        return True

    def _run_single_solve(self, task_args: Tuple) -> Tuple[str, str, Any, float]:
        """
        Worker function to run one solver on one instance. Executed by each thread.
        Returns a tuple: (instance_name, solver_name, gap, execution_time).
        """
        instance_name, coordinates, distance_matrix, demands, vehicle_capacity, baseline, solver_name = task_args
        gap = 'error'
        solve_time = 0.0
        try:
            # Dynamically import the solver module
            solver_module = importlib.import_module(solver_name)
            SolverClass = getattr(solver_module, 'CVRPSolver')

            # Instantiate and run the solver
            cvrp_solver = SolverClass(coordinates, distance_matrix, demands, vehicle_capacity)
            solve_start_time = time.time()
            cvrp_solution = cvrp_solver.solve()
            solve_time = time.time() - solve_start_time

            # --- Solution Formatting ---
            # If solution is a list of lists (multiple routes), flatten it.
            if isinstance(cvrp_solution, np.ndarray):  # More specific type
                cvrp_solution = cvrp_solution.tolist()

            if isinstance(cvrp_solution, list) and len(cvrp_solution) > 0 and isinstance(cvrp_solution[0], list):
                flat_solution = [0]  # Start at depot
                for route in cvrp_solution:
                    # Ensure each sub-route doesn't contain the depot
                    if 0 in route:
                        route.remove(0)
                    flat_solution.extend(route)
                    flat_solution.append(0)  # End route at depot
                cvrp_solution = flat_solution
            cvrp_solution.append(0)

            # --- Evaluation ---
            if not self.check_feasibility(cvrp_solution, demands, vehicle_capacity):
                print(cvrp_solution)
                gap = 'infeasible'
                print("infeasible")
            else:
                llm_cost = self.calculate_total_cost(coordinates, cvrp_solution)
                gap = (llm_cost - baseline) / baseline if baseline > 1e-6 else float('inf')
                print(f"{solver_name}, {instance_name}, time: {solve_time:.3f}, gap: {gap:.3f}")
            return (instance_name, solver_name, gap, solve_time)

        except ImportError:
            return (instance_name, solver_name, 'import_error', 0)
        except AttributeError:
            return (instance_name, solver_name, 'class_not_found', 0)
        except Exception as e:
            traceback.print_exc()
            return (instance_name, solver_name, 'runtime_error', 0)

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Evaluates all specified solvers against all CVRP instances in parallel.
        """
        start_time = time.time()

        # 1. Create a list of all tasks to be executed
        tasks = []
        for i, (instance_name, coords, dist_mat, demands, cap, baseline) in enumerate(self._datasets):

            if self.is_scale:
                # Normalize coordinates and update the distance matrix for each instance
                norm_coords, norm_dist_mat = self._normalize_and_update(coords, dist_mat)

                # The baseline cost should also be normalized to be comparable with the new costs
                # We assume the original cost is Euclidean. We can find the scale factor and apply it.
                x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
                y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
                scale_factor = max(x_max - x_min, y_max - y_min)
                norm_baseline = baseline / scale_factor if scale_factor > 1e-9 else baseline

                for solver_name in self.solver_modules:
                    # Pass the NORMALIZED data to the solver
                    tasks.append((instance_name, norm_coords, norm_dist_mat, demands, cap, norm_baseline, solver_name))

            else:
                for solver_name in self.solver_modules:
                    tasks.append((instance_name, coords, dist_mat, demands, cap, baseline, solver_name))

                # In the evaluate() method

        # 2. Use ProcessPoolExecutor to run tasks in parallel.
        # This is crucial to avoid deadlocks with Numba's JIT compilation in a parallel setting.
        # Processes have separate memory spaces, preventing conflicts during the first-time compilation.
        flat_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._run_single_solve, task): task for task in tasks}

            print(f"Submitting {len(tasks)} tasks to the thread pool...")
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks),
                               desc="Evaluating Solvers"):
                try:
                    result = future.result()
                    flat_results.append(result)
                except Exception as exc:
                    task_info = future_to_task[future]
                    print(f"Task {task_info[0]}/{task_info[6]} generated an exception: {exc}")

        # 3. Aggregate flat results into a structured format (one dict per instance)
        results_by_instance = {}
        for instance_name, solver_name, gap, solve_time in flat_results:
            if instance_name not in results_by_instance:
                results_by_instance[instance_name] = {'instance_name': instance_name}
            # Add two columns for each solver: gap and time
            results_by_instance[instance_name][f"{solver_name}_gap"] = gap
            results_by_instance[instance_name][f"{solver_name}_time"] = solve_time

        # 4. Convert the dictionary to a list, preserving original instance order
        ordered_results = []
        # Assuming instance names are like 'instance_0', 'instance_1', etc.
        for i in range(len(self._datasets)):
            name = self._datasets[i][0]
            if name in results_by_instance:
                ordered_results.append(results_by_instance[name])

        # 5. Write results to CSV and return
        self.write_results_to_csv(ordered_results)

        total_time = time.time() - start_time
        print(f"\nEvaluation finished in {total_time:.2f} seconds.")

        return ordered_results

    def write_results_to_csv(self, results_data: List[Dict[str, Any]]):
        """Writes the evaluation results to a CSV file."""
        if not results_data:
            print("No results to write.")
            return

        # Generate headers with a _gap and _time column for each solver
        headers = ['instance_name']
        for solver in self.solver_modules:
            headers.extend([f"{solver}_gap", f"{solver}_time"])

        try:
            with open(self.output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers, restval='N/A')
                writer.writeheader()
                writer.writerows(results_data)
            print(f"Successfully wrote results to '{self.output_csv_path}'")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")


if __name__ == '__main__':
    # --- How to use this script ---
    # 1. Make sure your solver file (e.g., 'my_cvrp_solver.py') is in the same directory
    #    or in Python's path.
    # 2. The solver file must contain a class named 'CVRPSolver'.
    # 3. This 'CVRPSolver' class must have an __init__ that accepts
    #    (coordinates, distance_matrix, demands, vehicle_capacity) and a 'solve()' method.

    # Define the list of solver modules you want to evaluate.
    # The names should be the python file names without the '.py' extension.
    solvers_to_evaluate = [
        #'eop_MODACR',
        'mths_00395_numba',
        #'gls_eoh',
        #'gls_reevo'
        #'or-tools_sa',
        #'or-tools_ts',
        # 'constructive_nn',
        # 'constructive_insert',
        # 'or-tools',
        #'aco_mcts_ahd_numba',
        #'eop_cvrp',# <--- REPLACE with your solver file name
        # 'your_cvrp_solver_filename_2',
    ]

    # Instantiate the evaluator.
    # A good starting point for num_threads is the number of your CPU cores.
    cvrp_evaluator = CVRPEvaluation(
        solver_modules=solvers_to_evaluate,
        num_threads=8,
        output_csv_path='cvrp_evaluation_results_mths_new.csv',
        dataset_path="./vrplib_dataset_160.pkl",
        is_scale=False,
    )

    # Run the parallel evaluation.
    results = cvrp_evaluator.evaluate()

    # Optional: Print the first few results for a quick check
    if results:
        print("\n--- Summary of Results ---")
        # Calculate and print average gap and time for each solver
        for solver in solvers_to_evaluate:
            gap_col = f"{solver}_gap"
            time_col = f"{solver}_time"

            valid_gaps = [r[gap_col] for r in results if isinstance(r.get(gap_col), (int, float))]
            if valid_gaps:
                avg_gap = np.mean(valid_gaps)
                print(f"Average Gap for '{solver}': {avg_gap:.4%}")
            else:
                print(f"No valid gap results found for '{solver}'")

            valid_times = [r[time_col] for r in results if isinstance(r.get(time_col), (int, float))]
            if valid_times:
                avg_time = np.mean(valid_times)
                print(f"Average Time for '{solver}': {avg_time:.4f} seconds")
            else:
                print(f"No valid time results found for '{solver}'")
            print("-" * 30)

        print("\n--- First 5 Rows of Result Data ---")
        for i in range(min(5, len(results))):
            print(results[i])
