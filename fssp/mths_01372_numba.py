#
# ALGORITHM Adaptive Cooperative Substructure Search (ACSS)
#



import numpy as np
import random
import time
from copy import deepcopy
from numba import njit
import warnings

# Numba may issue a warning if it can't parallelize a loop as much as it hoped.
# This is often fine, so we can ignore it for cleaner output.
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


# ==============================================================================
# Numba-jitted standalone functions for performance-critical calculations
# ==============================================================================

@njit(cache=True)
def _numba_makespan(sequence, processing_times, num_machines):
    """
    Compute makespan for a given permutation sequence.
    This is a Numba-jitted version for high performance.
    """
    if len(sequence) == 0:
        return 0.0

    # completion times on machines for last scheduled job
    C = np.zeros(num_machines)
    for job_idx in sequence:
        prev_finish_time = 0.0
        for m in range(num_machines):
            # Start time is the max of machine availability and previous operation completion
            start_time = max(C[m], prev_finish_time)
            finish_time = start_time + processing_times[job_idx, m]
            C[m] = finish_time
            prev_finish_time = finish_time

    return C[-1]


@njit(cache=True)
def _numba_best_insertion_improvement(seq, processing_times, num_jobs, num_machines):
    """
    Numba-jitted version of the best insertion local search.
    Returns improved sequence and new cost if improvement found, else (original sequence, original cost).
    """
    n = len(seq)
    if n <= 1:
        return seq, -1.0  # Sentinel for no improvement

    current_cost = _numba_makespan(seq, processing_times, num_machines)
    best_cost = current_cost
    best_seq = seq.copy()  # Start with the current sequence as the best

    # consider removing each position i and inserting at each position j
    for i in range(n):
        job = seq[i]

        # Create a temporary sequence with the job removed
        temp_seq = np.empty(n - 1, dtype=np.int64)
        k = 0
        for idx in range(n):
            if idx != i:
                temp_seq[k] = seq[idx]
                k += 1

        # Try inserting the job at all possible positions
        for j in range(n):
            if i == j:
                continue

            # Create the candidate sequence
            cand = np.empty(n, dtype=np.int64)
            cand[:j] = temp_seq[:j]
            cand[j] = job
            cand[j + 1:] = temp_seq[j:]

            cost = _numba_makespan(cand, processing_times, num_machines)

            if cost < best_cost - 1e-12:
                best_cost = cost
                best_seq = cand.copy()  # Must copy the new best sequence

    if best_cost < current_cost - 1e-12:
        return best_seq, best_cost

    return seq, current_cost  # Return original if no improvement


@njit(cache=True)
def _numba_best_swap_improvement(seq, processing_times, num_jobs, num_machines):
    """
    Numba-jitted version of the best swap local search.
    Returns improved sequence and new cost if improvement found, else (original sequence, original cost).
    """
    n = len(seq)
    if n <= 1:
        return seq, -1.0  # Sentinel for no improvement

    current_cost = _numba_makespan(seq, processing_times, num_machines)
    best_cost = current_cost
    best_seq = seq.copy()

    cand = seq.copy()
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Perform swap on the candidate
            cand[i], cand[j] = cand[j], cand[i]
            cost = _numba_makespan(cand, processing_times, num_machines)

            if cost < best_cost - 1e-12:
                best_cost = cost
                best_seq = cand.copy()

            # Swap back to reset for the next iteration
            cand[i], cand[j] = cand[j], cand[i]

    if best_cost < current_cost - 1e-12:
        return best_seq, best_cost

    return seq, current_cost  # Return original if no improvement


# ==============================================================================
# Original Class Structure - Modified to call Numba functions
# ==============================================================================

class FSSPSolver:
    def __init__(self, num_jobs: int, num_machines: int, processing_times: list,
                 time_limit: float = 60.0, max_iters: int = 1000000, random_seed: int = 2025):
        """
        Initialize the FSSP solver.

        Args:
            num_jobs: Number of jobs in the problem
            num_machines: Number of machines in the problem
            processing_times: List of lists where processing_times[j][m] is the processing time of job j on machine m
            time_limit: optional time limit (seconds) for iterative improvements
            max_iters: maximum number of perturbation iterations
            random_seed: optional random seed for reproducibility
        """
        self.num_jobs = int(num_jobs)
        self.num_machines = int(num_machines)
        # Convert to NumPy array for Numba compatibility and performance
        self.processing_times = np.array(processing_times, dtype=np.float64)
        self.time_limit = float(time_limit)
        self.max_iters = int(max_iters)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _makespan(self, sequence):
        """
        Compute makespan for a given permutation sequence (list or array of job indices).
        This method now acts as a wrapper around the high-performance Numba version.
        """
        # Ensure sequence is a NumPy array for Numba function
        seq_arr = np.array(sequence, dtype=np.int64)
        return _numba_makespan(seq_arr, self.processing_times, self.num_machines)

    def _neh_initial(self):
        """
        NEH heuristic: sort by total processing time (desc), then sequential insertion at best position.
        This function is not jitted as it's called only once and involves list manipulations
        that are less suitable for Numba. It still benefits from the fast `_makespan`.
        """
        n = self.num_jobs
        if n == 0:
            return []
        # Sums are faster with NumPy array
        totals = [(np.sum(self.processing_times[j]), j) for j in range(n)]
        totals.sort(reverse=True)  # descending by total processing time
        sorted_jobs = [j for _, j in totals]

        seq = []
        for job in sorted_jobs:
            best_seq = None
            best_cost = float('inf')
            # try all insertion positions
            for pos in range(len(seq) + 1):
                cand = seq[:pos] + [job] + seq[pos:]
                cost = self._makespan(cand)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = cand
            seq = best_seq
        return seq

    def _best_insertion_improvement(self, seq):
        """
        Wrapper for the Numba-jitted best insertion move function.
        Returns improved sequence and new cost if improvement found, else (None, None).
        """
        current_cost = self._makespan(seq)
        seq_arr = np.array(seq, dtype=np.int64)

        improved_seq_arr, new_cost = _numba_best_insertion_improvement(
            seq_arr, self.processing_times, self.num_jobs, self.num_machines
        )

        if new_cost < current_cost - 1e-12:
            return list(improved_seq_arr), new_cost

        return None, None

    def _best_swap_improvement(self, seq):
        """
        Wrapper for the Numba-jitted best pairwise swap improvement function.
        """
        current_cost = self._makespan(seq)
        seq_arr = np.array(seq, dtype=np.int64)

        improved_seq_arr, new_cost = _numba_best_swap_improvement(
            seq_arr, self.processing_times, self.num_jobs, self.num_machines
        )

        if new_cost < current_cost - 1e-12:
            return list(improved_seq_arr), new_cost

        return None, None

    def _local_search(self, seq, max_no_improve=50):
        """
        Apply local search combining insertion and swap (best-improvement) until no improvement.
        This function's logic remains the same, but it now calls the faster Numba-backed methods.
        """
        best_seq = seq[:]
        best_cost = self._makespan(best_seq)
        no_improve = 0
        while no_improve < max_no_improve:
            improved = False
            # try best insertion
            cand_seq, cand_cost = self._best_insertion_improvement(best_seq)
            if cand_seq is not None and cand_cost < best_cost - 1e-12:
                best_seq = cand_seq
                best_cost = cand_cost
                improved = True
                no_improve = 0
                continue  # continue with insertion improvements
            # try swap
            cand_seq, cand_cost = self._best_swap_improvement(best_seq)
            if cand_seq is not None and cand_cost < best_cost - 1e-12:
                best_seq = cand_seq
                best_cost = cand_cost
                improved = True
                no_improve = 0
                continue
            if not improved:
                no_improve += 1
                break
        return best_seq, best_cost

    def _perturb(self, seq, strength=2):
        """
        Perturbation by performing 'strength' random relocations/swaps.
        (No change needed here)
        """
        n = len(seq)
        if n <= 1 or strength <= 0:
            return seq[:]
        s = seq[:]
        for _ in range(strength):
            if random.random() < 0.5:
                # random swap
                i, j = random.sample(range(n), 2)
                s[i], s[j] = s[j], s[i]
            else:
                # random insertion
                i, j = random.sample(range(n), 2)
                job = s.pop(i)
                s.insert(j, job)
        return s

    def solve(self) -> list:
        """
        Solve the Flow Shop Scheduling Problem (FSSP).
        (No change needed here, it benefits from the faster components it calls)
        """
        n = self.num_jobs
        if n == 0:
            return []
        # Initial solution by NEH heuristic
        best_seq = self._neh_initial()
        best_cost = self._makespan(best_seq)

        # Improve with local search
        best_seq, best_cost = self._local_search(best_seq, max_no_improve=10)

        # Iterated Local Search with perturbations guided by improvements
        start_time = time.time()
        it = 0
        stagnation = 0
        while it < self.max_iters and (time.time() - start_time) < self.time_limit and stagnation < 100000:
            it += 1
            # adaptive perturbation strength
            strength = 1 + (it % max(1, n // 5))
            candidate = self._perturb(best_seq, strength=strength)
            candidate, cand_cost = self._local_search(candidate, max_no_improve=10)
            if cand_cost < best_cost - 1e-12:
                best_seq = candidate
                best_cost = cand_cost
                stagnation = 0
            else:
                stagnation += 1

        # Final polishing
        best_seq, best_cost = self._local_search(best_seq, max_no_improve=50)

        # Ensure it's a permutation of all jobs
        if sorted(best_seq) != list(range(n)):
            # fallback to simple identity
            return list(range(n))
        return best_seq

