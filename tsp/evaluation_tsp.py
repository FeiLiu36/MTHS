from __future__ import annotations
import sys
import os
import numpy as np
import pickle as pkl
import time
import importlib
import csv
from typing import List, Dict, Any, Tuple
import concurrent.futures

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

# Adjust the path to find your solver modules if necessary
sys.path.append('../')


# --- Mocking the base class for standalone execution ---
class Evaluation:
    def __init__(self, **kwargs):
        pass


# --- End of Mock ---

__all__ = ['TSPEvaluation']


class TSPEvaluation(Evaluation):
    """
    Evaluator for the Traveling Salesman Problem.
    It can evaluate multiple solvers on a dataset of TSP instances in parallel
    and write the results to a CSV file.
    """

    def __init__(self,
                 solver_modules: List[str],
                 num_threads: int = 4,
                 output_csv_path: str = 'tsp_results.csv',
                 dataset_path: str = "./TSPLIB_EUC2D_1000.pkl",
                 timeout_seconds: int = 30,
                 **kwargs):
        """
        Args:
            solver_modules (List[str]): A list of solver module names to evaluate.
            num_threads (int): The number of threads to use for parallel evaluation.
            output_csv_path (str): Path to write the final results CSV file.
            dataset_path (str): Path to the TSP instances pickle file.
            timeout_seconds (int): Timeout for each solver run (Note: not strictly enforced by ThreadPoolExecutor).
        """
        super().__init__(**kwargs)

        self.solver_modules = solver_modules
        self.num_threads = num_threads
        self.output_csv_path = output_csv_path
        self.timeout_seconds = timeout_seconds

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self._datasets = pkl.load(f)

        print(f"Loaded {len(self._datasets)} TSP instances.")
        print(f"Solvers to be evaluated: {self.solver_modules}")
        print(f"Running evaluation with {self.num_threads} threads.")

    def tour_cost(self, instance: np.ndarray, solution: List[int] | np.ndarray, problem_size: int) -> float:
        """Calculates the total cost of a given tour."""
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def check_feasibility(self, solution: List[int] | np.ndarray, problem_size: int) -> bool:
        """Checks if the TSP solution is feasible."""
        if not isinstance(solution, (list, np.ndarray)):
            print("solution must be a list or numpy array.")
            return False
        if len(solution) != problem_size:
            print("solution must be of length " + str(problem_size))
            return False
        if set(solution) != set(range(problem_size)):
            print("solution must be of length " + str(problem_size) +"and a set of the same size")
            return False
        return True

    def _run_single_solve(self, task_args: Tuple) -> Tuple[str, str, Any, float]:
        """
        Worker function to run one solver on one instance. This is executed by each thread.
        Returns a tuple: (instance_name, solver_name, gap, execution_time).
        """
        instance_name, coordinates, distance_matrix, baseline, solver_name = task_args
        problem_size = len(coordinates)
        gap = 'error'

        try:
            solver_module = importlib.import_module(solver_name)
            SolverClass = getattr(solver_module, 'TSPSolver')

            tsp_solver = SolverClass(coordinates, distance_matrix)
            solve_start_time = time.time()
            tsp_solution = tsp_solver.solve()

            solve_time = time.time() - solve_start_time

            if not self.check_feasibility(tsp_solution, problem_size):
                gap = 'infeasible'
            else:
                llm_dis = self.tour_cost(coordinates, tsp_solution, problem_size)

                gap = (llm_dis - baseline) / baseline if baseline > 0 else float('inf')

            print(f"method={solver_name}, instance_name={instance_name}, gap={gap}, solve_time={solve_time}")

            return (instance_name, solver_name, gap, solve_time)

        except ImportError:
            return (instance_name, solver_name, 'import_error', 0)
        except AttributeError:
            return (instance_name, solver_name, 'class_not_found', 0)
        except Exception as e:
            # It can be useful to log the specific error
            print(f"Runtime error in {solver_name} on {instance_name}: {e}")
            return (instance_name, solver_name, 'runtime_error', 0)

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Evaluates all specified solvers against all TSP instances in parallel.
        """
        start_time = time.time()

        # 1. Create a list of all tasks to be executed
        tasks = []
        for name, coordinates, distance_matrix, baseline in self._datasets:
            for solver_name in self.solver_modules:
                tasks.append((name, coordinates, distance_matrix, baseline, solver_name))

        # 2. Use ThreadPoolExecutor to run tasks in parallel
        flat_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks and create a future-to-task mapping
            future_to_task = {executor.submit(self._run_single_solve, task): task for task in tasks}

            # Use tqdm for a progress bar as tasks complete
            print(f"Submitting {len(tasks)} tasks to the thread pool...")
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks),
                               desc="Evaluating Solvers"):
                try:
                    result = future.result()
                    flat_results.append(result)
                except Exception as exc:
                    task_info = future_to_task[future]
                    print(f"Task {task_info[0]}/{task_info[4]} generated an exception: {exc}")

        # 3. Aggregate the flat results into a structured format (one dict per instance)
        results_by_instance = {}
        for instance_name, solver_name, gap, solve_time in flat_results:
            if instance_name not in results_by_instance:
                results_by_instance[instance_name] = {'instance_name': instance_name}

            # Store the gap, you could also store solve_time if needed
            results_by_instance[instance_name][f"{solver_name}_gap"] = gap
            results_by_instance[instance_name][f"{solver_name}_time"] = solve_time
            # Example: results_by_instance[instance_name][f"{solver_name}_time"] = solve_time

        # 4. Convert the dictionary of results into a list, preserving original instance order
        ordered_results = []
        instance_names_ordered = [d[0] for d in self._datasets]
        for name in instance_names_ordered:
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
    # Define the list of solver modules you want to evaluate.
    solvers_to_evaluate = [
        "mths_00148_numba"
        # 'or-tools_sa',
        # 'or-tools_ts',
        # 'constructive_nn',
        # 'constructive_insert',
        # 'aco_eoh',
        # 'aco_mcts_ahd',
        # 'gls_eoh',
        # 'gls_reevo',
        # 'gls_mcts_ahd',
        #'mteop_tsp_numba',
    ]

    # Instantiate the evaluator with the desired number of threads.
    # A good starting point is the number of CPU cores.
    tsp_evaluator = TSPEvaluation(
        solver_modules=solvers_to_evaluate,
        num_threads=8,  # <-- Set the number of threads here
        output_csv_path='tsp_evaluation_results_mths_new.csv'
    )

    # Run the parallel evaluation.
    results = tsp_evaluator.evaluate()

    # Optional: Print the first few results for a quick check
    if results:
        print("\n--- Sample of Results ---")
        for i in range(min(5, len(results))):
            print(results[i])
