from __future__ import annotations
import time
import traceback
import pickle
import csv
import concurrent.futures
from typing import Any, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

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


# --- Mocking the base class and templates for standalone execution ---
class Evaluation:
    def __init__(self, **kwargs):
        pass



# --- End of Mock ---


class BPPEvaluation(Evaluation):
    """
    Evaluator for the Bin Packing Problem (BPP).
    It can evaluate a solver on a dataset of BPP instances in parallel
    and write the results to a CSV file.
    """

    def __init__(self,
                 num_threads: int = 4,
                 output_csv_path: str = 'bpp_results.csv',
                 dataset_path: str = "bpp_instances_500_items.pkl",
                 timeout_seconds: int = 30,  # Note: not strictly enforced by ThreadPoolExecutor
                 **kwargs):
        """
        Args:
            num_threads (int): The number of threads to use for parallel evaluation.
            output_csv_path (str): Path to write the final results CSV file.
            dataset_path (str): Path to the BPP instances pickle file.
            timeout_seconds (int): Timeout for each solver run.
        """
        super().__init__(**kwargs)

        self.num_threads = num_threads
        self.output_csv_path = output_csv_path
        self.timeout_seconds = timeout_seconds

        try:
            with open(dataset_path, 'rb') as f:
                self._datasets = pickle.load(f)
            print(f"Loaded {len(self._datasets)} BPP instances from '{dataset_path}'.")
        except FileNotFoundError:
            print(f"\n!!! ERROR: '{dataset_path}' not found. !!!")
            print("Please run a data generation script to create the instance file.")
            self._datasets = []

        print(f"Running evaluation with {self.num_threads} threads.")

    def check_feasibility(self, instance: Dict, packing: List[List[int]]) -> bool:
        """Checks if the BPP solution is feasible."""
        if not isinstance(packing, list) or not all(isinstance(b, list) for b in packing):
            print(f"Feasibility Error (Instance: {instance['name']}): Solution is not a list of lists.")
            return False

        weights = instance['weights']
        capacity = instance['capacity']
        num_items = len(weights)

        # 1. Check if bin capacities are respected
        for i, bin_content in enumerate(packing):
            bin_load = sum(weights[item_idx] for item_idx in bin_content)
            if bin_load > capacity:
                print(
                    f"Feasibility Error (Instance: {instance['name']}): Bin {i} load {bin_load} exceeds capacity {capacity}.")
                return False

        # 2. Check if all items are packed exactly once
        packed_items = [item_idx for bin_content in packing for item_idx in bin_content]
        if len(packed_items) != num_items:
            print(
                f"Feasibility Error (Instance: {instance['name']}): Solution has {len(packed_items)} items, but instance requires {num_items}.")
            return False
        if set(packed_items) != set(range(num_items)):
            print(f"Feasibility Error (Instance: {instance['name']}): Items are either duplicated or missing.")
            return False

        return True

    def _run_single_solve(self, task_args: Tuple) -> Tuple[str, Any, float]:
        """
        Worker function to run one solver on one instance.
        Returns a tuple: (instance_name, gap, execution_time).
        """
        instance, solver_class = task_args
        instance_name = instance['name']
        gap = 'error'

        try:
            bpp_solver = solver_class(
                capacity=instance['capacity'],
                weights=instance['weights']
            )
            solve_start_time = time.time()
            packing = bpp_solver.solve()
            solve_time = time.time() - solve_start_time

            if not self.check_feasibility(instance, packing):
                gap = 'infeasible'
            else:
                num_bins_used = len(packing)
                best_known = instance['best_known']
                gap = (num_bins_used - best_known) / best_known if best_known > 0 else float('inf')


            # Optional: print progress for each task
            print(f"instance_name={instance_name}, num_bins_used={num_bins_used}, gap={gap:.4f}, solve_time={solve_time:.4f}")

            return (instance_name, gap, solve_time)

        except Exception as e:
            print(f"Runtime error on instance {instance_name}: {e}")
            traceback.print_exc()
            return (instance_name, 'runtime_error', 0)

    def evaluate_program(self, program_str: str) -> List[Dict[str, Any]]:
        """
        Dynamically executes the provided program string and evaluates the implemented solver
        on all instances in parallel.
        """
        start_time = time.time()

        # 1. Dynamically load the solver class from the program string
        g = {}
        try:
            exec(program_str, g)
            solver_class = g['BPPSolver']
        except Exception as e:
            print(f"Failed to execute the program string: {e}")
            traceback.print_exc()
            return []

        # 2. Create a list of all tasks to be executed
        tasks = [(instance, solver_class) for instance in self._datasets]

        # 3. Use ThreadPoolExecutor to run tasks in parallel
        flat_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_task = {executor.submit(self._run_single_solve, task): task for task in tasks}

            print(f"Submitting {len(tasks)} tasks to the thread pool...")
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks),
                               desc="Evaluating Solver"):
                try:
                    result = future.result()
                    flat_results.append(result)
                except Exception as exc:
                    task_info = future_to_task[future]
                    print(f"Task for instance {task_info[0]['name']} generated an exception: {exc}")

        # 4. Aggregate results into a list of dictionaries
        results_by_instance = {res[0]: {'instance_name': res[0], 'gap': res[1], 'time': res[2]} for res in flat_results}

        # 5. Convert the dictionary of results into a list, preserving original instance order
        ordered_results = []
        instance_names_ordered = [d['name'] for d in self._datasets]
        for name in instance_names_ordered:
            if name in results_by_instance:
                ordered_results.append(results_by_instance[name])

        # 6. Write results to CSV and return
        self.write_results_to_csv(ordered_results)

        total_time = time.time() - start_time
        print(f"\nEvaluation finished in {total_time:.2f} seconds.")

        return ordered_results

    def write_results_to_csv(self, results_data: List[Dict[str, Any]]):
        """Writes the evaluation results to a CSV file."""
        if not results_data:
            print("No results to write.")
            return

        headers = ['instance_name', 'gap', 'time']
        try:
            with open(self.output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers, restval='N/A')
                writer.writeheader()
                writer.writerows(results_data)
            print(f"Successfully wrote results to '{self.output_csv_path}'")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

    def plot_solution(self, instance: Dict, packing: List[List[int]], instance_name: str = "Unknown"):
        """Plots a single bin packing solution."""
        if not packing:
            print("Cannot plot: Empty packing solution")
            return

        num_bins = len(packing)
        bin_capacity = instance['capacity']
        weights = instance['weights']
        num_items = len(weights)
        fig, ax = plt.subplots(figsize=(max(8, num_bins * 1.5), 6))
        colors = plt.cm.get_cmap('viridis', num_items)
        bin_labels = []

        for i, bin_indices in enumerate(packing):
            current_height = 0
            bin_load = sum(weights[item_idx] for item_idx in bin_indices)
            bin_labels.append(f'Bin {i + 1}\nLoad: {bin_load}')
            for item_idx in bin_indices:
                item_weight = weights[item_idx]
                ax.bar(i, item_weight, bottom=current_height, color=colors(item_idx), edgecolor='black',
                       label=f'Item {item_idx}')
                ax.text(i, current_height + item_weight / 2, f'I{item_idx}\n({item_weight})', ha='center', va='center',
                        color='white', fontsize=8, weight='bold')
                current_height += item_weight

        ax.axhline(y=bin_capacity, color='r', linestyle='--', linewidth=2, label=f'Capacity ({bin_capacity})')
        ax.set_xticks(np.arange(num_bins))
        ax.set_xticklabels(bin_labels)
        ax.set_ylabel("Accumulated Weight")
        ax.set_ylim(0, bin_capacity * 1.1)
        ax.set_title(f"Bin Packing Solution - Instance: {instance_name}\nBins Used: {num_bins}")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend([by_label[f'Capacity ({bin_capacity})']], ['Capacity'], loc='upper right')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from gpt5mini_mh import program
    # Instantiate the evaluator with the desired number of threads.
    bpp_evaluator = BPPEvaluation(
        num_threads=8,  # <-- Set the number of threads here
        dataset_path="bpp_instances_1000_c150_n64_test.pkl",
        output_csv_path='bpp_evaluation_results_1000.csv'
    )

    # The template_program string contains a complete, runnable solver (FFD Heuristic).
    # We pass it directly to evaluate_program to run the parallel evaluation.
    print("\n--- Evaluating the provided template program (FFD Heuristic) in parallel ---")
    results = bpp_evaluator.evaluate_program(program)

    # Optional: Print the first few results for a quick check
    if results:
        print("\n--- Sample of Results ---")
        for i in range(min(5, len(results))):
            print(results[i])

        # Calculate and print the average gap
        valid_gaps = [r['gap'] for r in results if isinstance(r['gap'], (int, float))]
        if valid_gaps:
            avg_gap = np.mean(valid_gaps)
            print(f"\nAverage Gap across all instances: {avg_gap:.4f}")
        else:
            print("\nNo valid gaps were calculated.")

