from __future__ import annotations

import os
import sys
import time
import importlib
import csv
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
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


# --- Mocking the base class for standalone execution ---
# In a real application, this would be imported from your library's base module.
class Evaluation:
    def __init__(self, **kwargs):
        pass


# --- End of Mock ---


class FSSPEvaluation(Evaluation):
    """
    Evaluator for the Job Shop Scheduling Problem (specifically Flow Shop variant).
    It can evaluate multiple solvers on a dataset of JSSP/FSSP instances in parallel
    and write the results to a CSV file. This class is structured similarly to
    the TSPEvaluation for consistency.
    """

    def __init__(self,
                 solver_modules: List[str],
                 num_threads: int = 4,
                 output_csv_path: str = 'fssp_results.csv',
                 dataset_path: str = "./fssp_taillard.pkl",
                 timeout_seconds: int = 60,
                 n_instance: int = 16,
                 **kwargs):
        """
        Args:
            solver_modules (List[str]): A list of solver module names to evaluate.
            num_threads (int): The number of threads to use for parallel evaluation.
            output_csv_path (str): Path to write the final results CSV file.
            dataset_path (str): Path to the JSSP/FSSP instances pickle file.
            timeout_seconds (int): Timeout for each solver run (Note: not strictly enforced by ThreadPoolExecutor).
            n_instance (int): The number of instances to load from the dataset.
        """
        super().__init__(**kwargs)

        self.solver_modules = solver_modules
        self.num_threads = num_threads
        self.output_csv_path = output_csv_path
        self.timeout_seconds = timeout_seconds
        self.n_instance = n_instance

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            all_instances = pkl.load(f)

        # The original dataset is a list of tuples (name, instance_dict)
        # self._datasets = {}
        # n=0
        # for name, instance in all_instances.items():
        #
        #     n+=1
        #     if n >80:
        #         self._datasets[name] = instance

        self._datasets = all_instances


        print(f"Loaded {len(self._datasets)} JSSP/FSSP instances.")
        print(f"Solvers to be evaluated: {self.solver_modules}")
        print(f"Running evaluation with {self.num_threads} threads.")

    def _run_single_solve(self, task_args: Tuple) -> Tuple[str, str, Any, float]:
        """
        Worker function to run one solver on one instance. This is executed by each thread.
        Returns a tuple: (instance_name, solver_name, gap, execution_time).
        """
        instance_name, instance_data, solver_name = task_args
        gap = 'error'

        try:
            # Dynamically import the solver module. Assumes the solver class is named 'FSSPSolver'.
            solver_module = importlib.import_module(solver_name)
            SolverClass = getattr(solver_module, 'FSSPSolver')

            # Create a solver instance
            fssp_solver = SolverClass(
                num_jobs=instance_data['num_jobs'],
                num_machines=instance_data['num_machines'],
                processing_times=instance_data['processing_times']
            )

            # Solve the problem and time it
            solve_start_time = time.time()
            job_sequence = fssp_solver.solve()
            solve_time = time.time() - solve_start_time

            # Check feasibility and calculate the optimality gap
            if not self.check_feasibility(instance_data, job_sequence):
                gap = 'infeasible'
            else:
                makespan = self.calculate_makespan_from_sequence(instance_data, job_sequence)
                best_known = instance_data['best_known']
                gap = (makespan - best_known) / best_known if best_known > 0 else float('inf')
            print(f"instance_name: {instance_name}, gap: {gap}, solve_time: {solve_time}")
            return (instance_name, solver_name, gap, solve_time)

        except ImportError:
            return (instance_name, solver_name, 'import_error', 0)
        except AttributeError:
            return (instance_name, solver_name, 'class_not_found', 0)
        except Exception as e:
            print(f"Error in {solver_name} on {instance_name}: {e}")
            # For debugging, you can uncomment the line below
            # print(f"Runtime error in {solver_name} on {instance_name}: {e}")
            return (instance_name, solver_name, 'runtime_error', 0)

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Evaluates all specified solvers against all FSSP instances in parallel.
        """
        start_time = time.time()

        # 1. Create a list of all tasks to be executed
        tasks = []
        for name, instance_data in self._datasets.items():
            for solver_name in self.solver_modules:
                tasks.append((name, instance_data, solver_name))

        # 2. Use ThreadPoolExecutor to run tasks in parallel
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
                    # task_info is (instance_name, instance_data, solver_name)
                    print(f"Task {task_info[0]}/{task_info[2]} generated an exception: {exc}")

        # 3. Aggregate the flat results into a structured format
        results_by_instance = {}
        for instance_name, solver_name, gap, solve_time in flat_results:
            if instance_name not in results_by_instance:
                results_by_instance[instance_name] = {'instance_name': instance_name}

            results_by_instance[instance_name][solver_name] = gap
            # Example: to also store solve time, uncomment the line below
            # results_by_instance[instance_name][f"{solver_name}_time"] = solve_time

        # 4. Convert the dictionary of results into a list, preserving original instance order
        ordered_results = []
        instance_names_ordered = [name for name, instance_data in self._datasets.items()]
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

        headers = ['instance_name'] + self.solver_modules

        try:
            with open(self.output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers, restval='N/A')
                writer.writeheader()
                writer.writerows(results_data)
            print(f"Successfully wrote results to '{self.output_csv_path}'")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

    # --- Helper methods from original code (unchanged) ---

    def check_feasibility(self, instance, job_sequence):
        if not isinstance(job_sequence, list) or len(job_sequence) != instance['num_jobs']:
            return False
        if set(job_sequence) != set(range(instance['num_jobs'])):
            return False
        return True

    def calculate_makespan_from_sequence(self, instance, job_sequence):
        if not job_sequence or len(job_sequence) != instance['num_jobs']:
            return float('inf')

        num_machines = instance['num_machines']
        completion_times = [0] * num_machines

        for job_id in job_sequence:
            # First machine
            completion_times[0] += instance['processing_times'][job_id][0]
            # Other machines
            for m_id in range(1, num_machines):
                completion_times[m_id] = max(completion_times[m_id], completion_times[m_id - 1]) + \
                                         instance['processing_times'][job_id][m_id]

        return completion_times[-1]

    def plot_solution(self, instance, job_sequence, instance_name="Unknown"):
        # This plotting utility is kept for convenience but is not used during evaluation.
        # Implementation is omitted here for brevity but is identical to the original.
        pass

    def calculate_schedule_from_sequence(self, instance, job_sequence):
        # This utility is kept for convenience but is not used during evaluation.
        # Implementation is omitted here for brevity but is identical to the original.
        pass


if __name__ == '__main__':
    # --- Example Usage ---
    # To run this, you need to have solver files in your Python path.
    # For example, create a file named 'eop_jssp.py' with the following content:
    #
    # import random
    # class FSSPSolver:
    #     def __init__(self, num_jobs, num_machines, processing_times):
    #         self.num_jobs = num_jobs
    #     def solve(self):
    #         # Returns a simple random permutation of jobs
    #         job_sequence = list(range(self.num_jobs))
    #         random.shuffle(job_sequence)
    #         return job_sequence

    # Define the list of solver modules you want to evaluate.
    # These should be importable Python module names (e.g., 'my_solver_file').
    solvers_to_evaluate = [
        'mths_01372_numba',
        #'eop_fssp',  # Assumes a file named eop_jssp.py exists
    ]

    # Instantiate the evaluator with the desired number of threads.
    fssp_evaluator = FSSPEvaluation(
        solver_modules=solvers_to_evaluate,
        num_threads=8,  # Adjust based on your CPU cores
        output_csv_path='fssp_evaluation_results_mths_new2.csv',
        n_instance=16  # Evaluate on all 16 instances
    )

    # Run the parallel evaluation.
    results = fssp_evaluator.evaluate()

    # Optional: Print the first few results for a quick check
    if results:
        print("\n--- Sample of Results ---")
        for i in range(min(5, len(results))):
            print(results[i])
