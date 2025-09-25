import numpy as np
# You must install ortools first: pip install ortools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class FSSPSolver:
    def __init__(self, num_jobs: int, num_machines: int, processing_times: list):
        """
        Initialize the FSSP solver.

        Args:
            num_jobs: Number of jobs in the problem
            num_machines: Number of machines in the problem
            processing_times: List of lists where processing_times[j][m] is the processing time of job j on machine m
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times
        self.all_jobs = range(self.num_jobs)
        self.all_machines = range(self.num_machines)

    def solve(self) -> list:
        """
        Solve the Flow Shop Scheduling Problem (FSSP) using OR-Tools routing metaheuristics.

        This approach models the FSSP as a vehicle routing problem where 'jobs' are 'locations'.
        The goal is to find the sequence (tour) of jobs that minimizes the makespan.

        Returns:
            A list representing the sequence of jobs to be processed.
            For example, [0, 2, 1] means job 0 is processed first, then job 2, then job 1.
            The sequence must include all jobs exactly once.
        """
        # --- Step 1: Create the data model and manager ---
        # The number of "locations" is the number of jobs.
        # We use a single "vehicle" to determine the sequence.
        # The depot (start/end of the tour) is an arbitrary job, let's use 0.
        manager = pywrapcp.RoutingIndexManager(self.num_jobs, 1, 0)

        # --- Step 2: Create the routing model ---
        routing = pywrapcp.RoutingModel(manager)

        # --- Step 3: Define the cost function and dimensions ---
        # The core of this model is to track the completion time on each machine.
        # We use a "Dimension" for each machine. A dimension is a quantity that
        # accumulates along the route, like weight or volume. Here, it's time.

        # For each machine, we define a callback that calculates the completion time
        # of the 'to_node' job, given the completion time of the 'from_node' job.
        for m in self.all_machines:
            # This callback calculates the time for a job on a specific machine.
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)

                # For the very first job in the sequence (from_node is the depot 0),
                # the completion time is just its own processing time.
                # If it's the first machine (m=0), this is simple.
                if from_node == 0 and m == 0:
                    return self.processing_times[to_node][m]

                # For subsequent jobs, we need to consider the completion time
                # of the *previous job on this machine* and the completion time
                # of *this job on the previous machine*.
                # This logic is handled by the dimension's cumul_var + transit relation.
                # The transit part is just the processing time of the current job.
                return self.processing_times[to_node][m]

            transit_callback_index = routing.RegisterTransitCallback(time_callback)

            # Now, create the dimension for this machine.
            # Horizon is a safe upper bound on the makespan.
            horizon = sum(pt for job_pts in self.processing_times for pt in job_pts)
            dimension_name = f'Machine_{m}'

            routing.AddDimension(
                transit_callback_index,
                horizon,  # slack_max: upper bound on idle time
                horizon,  # capacity: upper bound on total time
                False,  # start_cumul_to_zero: No, start time depends on previous machines
                dimension_name
            )

            dimension = routing.GetDimensionOrDie(dimension_name)

            # Add the precedence constraint: a job can't start on machine 'm'
            # until it's finished on machine 'm-1'.
            if m > 0:
                previous_machine_dim = routing.GetDimensionOrDie(f'Machine_{m - 1}')
                for i in self.all_jobs:
                    index = manager.NodeToIndex(i)
                    # C(i, m) >= C(i, m-1) + P(i, m)
                    # which is equivalent to: Start(i, m) >= C(i, m-1)
                    # The dimension's cumul_var represents the completion time.
                    # So, Cumul(i, m) - Transit(i, m) >= Cumul(i, m-1)
                    routing.AddRelation(
                        pywrapcp.ROUTING_GREATER_EQUAL,
                        dimension.CumulVar(index),
                        [previous_machine_dim.CumulVar(index)],
                        -self.processing_times[i][m]
                    )

        # --- Step 4: Define the objective function ---
        # The objective is to minimize the makespan, which is the completion time
        # of the very last job on the very last machine.
        last_machine_dim = routing.GetDimensionOrDie(f'Machine_{self.num_machines - 1}')
        routing.SetGlobalSpanCost(last_machine_dim)

        # --- Step 5: Set search parameters ---
        # Use metaheuristics to find a good solution.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30  # Set a time limit
        search_parameters.log_search = False

        # --- Step 6: Solve the problem ---
        solution = routing.SolveWithParameters(search_parameters)

        # --- Step 7: Extract and return the solution ---
        if solution:
            job_sequence = []
            index = routing.Start(0)
            # The depot (job 0) is the start, so we skip it in the sequence output.
            index = solution.Value(routing.NextVar(index))
            while not routing.IsEnd(index):
                job_sequence.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))

            # The sequence from the routing model doesn't include the starting depot.
            # We need to find where the depot job (0) fits in the sequence.
            # Since our model starts and ends at 0, the full tour is [0, j1, j2, ...].
            # So we just prepend 0 to the sequence.
            final_sequence = [0] + job_sequence

            # Optional: Print the makespan
            makespan = solution.ObjectiveValue()
            print(f'Makespan: {makespan}')

            return final_sequence
        else:
            print('No solution found!')
            return list(range(self.num_jobs))

#
# # Example Usage:
# if __name__ == '__main__':
#     # Example from the OR-Tools documentation
#     processing_times_data = [
#         [8, 6, 1],  # Job 0
#         [3, 5, 7],  # Job 1
#         [9, 2, 4],  # Job 2
#         [5, 8, 9],  # Job 3
#     ]
#     num_jobs_example = len(processing_times_data)
#     num_machines_example = len(processing_times_data[0])
#
#     # Create the solver instance
#     fssp_solver = FSSPSolver(
#         num_jobs=num_jobs_example,
#         num_machines=num_machines_example,
#         processing_times=processing_times_data
#     )
#
#     # Solve the problem
#     sequence = fssp_solver.solve()
#
#     # Print the result
#     print(f"Job sequence found: {sequence}")
#     # Expected makespan is 33. A common sequence is [1, 0, 3, 2].
#     # The solver might find another optimal sequence depending on the run.
