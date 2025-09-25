from __future__ import annotations
import random
import copy
from typing import Any, List, Dict
import matplotlib.pyplot as plt
import numpy as np
import traceback
from llm4ad.base import Evaluation
from template import template_program, task_description
import pickle


# --- BPP Evaluation Class ---

class BPPEvaluation(Evaluation):
    def __init__(self,
                 timeout_seconds=20,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
        )

        # Define BPP instances directly for demonstration
        # Each instance has a name, capacity, list of weights, and the best-known solution (optimal number of bins)
        try:
            # Make sure to use the new filename here
            with open('bpp_instances_500_items.pkl', 'rb') as f:
                self._datasets = pickle.load(f)
            print(f"Loaded {len(self._datasets)} BPP instances from 'bpp_instances_500_items.pkl' for evaluation.")
        except FileNotFoundError:
            print("\n!!! ERROR: 'bpp_instances_500_items.pkl' not found. !!!")
            print("Please run the data generation script first to create the instance file.")
            self._datasets = []

        self._datasets = self._datasets[:16]

        print(f"Loaded {len(self._datasets)} BPP instances for evaluation")

    def plot_solution(self, instance: Dict, packing: List[List[int]], instance_name: str = "Unknown"):
        """
        Plot the bin packing solution.

        Args:
            instance (Dict): BPP instance with capacity and weights.
            packing (List[List[int]]): The solution, a list of bins with item indices.
            instance_name (str): Name of the instance for the plot title.
        """
        if not packing:
            print("Cannot plot: Empty packing solution")
            return

        num_bins = len(packing)
        bin_capacity = instance['capacity']
        weights = instance['weights']
        num_items = len(weights)

        fig, ax = plt.subplots(figsize=(max(8, num_bins * 1.5), 6))

        # Define colors for items
        colors = plt.cm.get_cmap('viridis', num_items)

        bin_labels = []
        for i, bin_indices in enumerate(packing):
            current_height = 0
            bin_load = sum(weights[item_idx] for item_idx in bin_indices)
            bin_labels.append(f'Bin {i + 1}\nLoad: {bin_load}')

            # Plot each item in the bin as a stacked bar segment
            for item_idx in bin_indices:
                item_weight = weights[item_idx]
                ax.bar(i, item_weight, bottom=current_height, color=colors(item_idx), edgecolor='black',
                       label=f'Item {item_idx}')

                # Add text label for the item
                ax.text(i, current_height + item_weight / 2, f'I{item_idx}\n({item_weight})',
                        ha='center', va='center', color='white', fontsize=8, weight='bold')

                current_height += item_weight

        # Add a line for bin capacity
        ax.axhline(y=bin_capacity, color='r', linestyle='--', linewidth=2, label=f'Capacity ({bin_capacity})')

        ax.set_xticks(np.arange(num_bins))
        ax.set_xticklabels(bin_labels)
        ax.set_ylabel("Accumulated Weight")
        ax.set_ylim(0, bin_capacity * 1.1)
        ax.set_title(f"Bin Packing Solution - Instance: {instance_name}\nBins Used: {num_bins}")

        # Create a unique legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend([by_label['Capacity ({})'.format(bin_capacity)]], ['Capacity'], loc='upper right')

        plt.tight_layout()
        plt.show()

    def calculate_num_bins(self, packing: List[List[int]]) -> int:
        """Calculate the number of bins used."""
        return len(packing)

    def check_feasibility(self, instance: Dict, packing: List[List[int]]) -> bool:
        """
        Check if the packing solution is feasible.

        Args:
            instance (Dict): The BPP instance.
            packing (List[List[int]]): The proposed solution.

        Returns:
            bool: True if the solution is feasible, False otherwise.
        """
        weights = instance['weights']
        capacity = instance['capacity']
        num_items = len(weights)

        # 1. Check if bin capacities are respected
        for bin_content in packing:
            if not isinstance(bin_content, list): return False  # Structure check
            bin_load = sum(weights[item_idx] for item_idx in bin_content)
            if bin_load > capacity:
                print(f"Feasibility Error: Bin load {bin_load} exceeds capacity {capacity}.")
                return False

        # 2. Check if all items are packed exactly once
        packed_items = [item_idx for bin_content in packing for item_idx in bin_content]

        # Check for correct number of items
        if len(packed_items) != num_items:
            print(f"Feasibility Error: Solution has {len(packed_items)} items, but instance has {num_items}.")
            return False

        # Check for duplicates or missing items
        if set(packed_items) != set(range(num_items)):
            print("Feasibility Error: Items are either duplicated or missing.")
            return False

        return True

    def evaluate(self, alg: type) -> float | None:
        """
        Evaluate the BPP algorithm on the test instances.

        Args:
            alg: The BPP solver algorithm class.

        Returns:
            float: The negative of the average relative gap to the best known solution.
                   Returns None if an error or infeasible solution is found.
        """
        try:
            gap_list = []

            for instance in self._datasets:
                # Create a solver instance
                bpp_solver = alg(
                    capacity=instance['capacity'],
                    weights=instance['weights']
                )

                # Solve the problem
                packing = bpp_solver.solve()

                # Check feasibility
                if not self.check_feasibility(instance, packing):
                    print(f"Infeasible solution for instance: {instance['name']}")
                    return None

                # Calculate number of bins used
                num_bins_used = self.calculate_num_bins(packing)

                # Optional: Plot the solution for one instance
                # if instance['name'] == 'simple_2':
                #    self.plot_solution(instance, packing, instance['name'])

                # Calculate gap to best known solution
                best_known = instance['best_known']
                gap = (num_bins_used - best_known) / best_known
                print(gap)
                gap_list.append(gap)

            # Return negative average gap (since lower gap is better, we maximize the negative gap)
            avg_gap = np.mean(gap_list)
            return -avg_gap

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            print("Traceback:")
            traceback.print_exc()
            return None

    def evaluate_program(self, program_str: str, callable_func: callable = None) -> Any | None:
        """
        Dynamically executes the provided program string and evaluates the implemented solver.
        """
        g = {}
        try:
            exec(program_str, g)
            solver_class = g['BPPSolver']
            return self.evaluate(solver_class)
        except Exception as e:
            print(f"Failed to execute or evaluate the program string: {e}")
            traceback.print_exc()
            return None


if __name__ == '__main__':
    # Instantiate the evaluation class
    bpp_evaluator = BPPEvaluation()

    # The template_program string contains a complete, runnable solver implementation.
    # We can pass it directly to evaluate_program.
    print("--- Evaluating the provided template program (FFD Heuristic) ---")
    result = bpp_evaluator.evaluate_program(template_program)

    if result is not None:
        # The result is the negative average gap. A value closer to 0 is better.
        print(f"\nEvaluation finished successfully.")
        print(f"Returned Score (Negative Average Gap): {result:.4f}")
        print(f"Average Gap: {-result:.4f}")
    else:
        print("\nEvaluation failed or returned an infeasible solution.")

    # --- Example of plotting a single solution ---
    print("\n--- Generating and plotting a solution for a single instance ---")
    try:
        # Pick one instance to visualize
        instance_to_plot = bpp_evaluator._datasets[1]

        # Create a solver from the template
        g = {}
        exec(template_program, g)
        SolverClass = g['BPPSolver']

        solver = SolverClass(capacity=instance_to_plot['capacity'], weights=instance_to_plot['weights'])
        solution = solver.solve()

        # Plot the generated solution
        bpp_evaluator.plot_solution(instance_to_plot, solution, instance_to_plot['name'])
    except Exception as e:
        print(f"Failed to generate plot: {e}")

