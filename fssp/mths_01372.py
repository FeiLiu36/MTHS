# Improved in knowledge transfer
import numpy as np
import random
import time
from copy import deepcopy

class FSSPSolver:
    def __init__(self, num_jobs: int, num_machines: int, processing_times: list,
                 time_limit: float = 5.0, max_iters: int = 200, random_seed: int = None):
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
        self.processing_times = processing_times
        self.time_limit = float(time_limit)
        self.max_iters = int(max_iters)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _makespan(self, sequence):
        """
        Compute makespan for a given permutation sequence (list of job indices).
        Uses the classical dynamic update over machines.
        """
        if not sequence:
            return 0
        M = self.num_machines
        # completion times on machines for last scheduled job
        C = [0] * M
        for job in sequence:
            prev = 0
            for m in range(M):
                start = C[m] if C[m] > prev else prev
                finish = start + self.processing_times[job][m]
                C[m] = finish
                prev = finish
        return C[-1]

    def _neh_initial(self):
        """
        NEH heuristic: sort by total processing time (desc), then sequential insertion at best position.
        """
        n = self.num_jobs
        if n == 0:
            return []
        totals = [(sum(self.processing_times[j]), j) for j in range(n)]
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
        Try best insertion move: remove a job and insert it in best position (excluding current).
        Returns improved sequence and new cost if improvement found, else (None, None).
        """
        n = len(seq)
        if n <= 1:
            return None, None
        current_cost = self._makespan(seq)
        best_cost = current_cost
        best_seq = None
        # consider removing each position i and inserting at each position j != i
        for i in range(n):
            job = seq[i]
            seq_removed = seq[:i] + seq[i+1:]
            # try insert positions 0..n-1
            for j in range(n):
                cand = seq_removed[:j] + [job] + seq_removed[j:]
                if j == i:
                    continue
                cost = self._makespan(cand)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best_seq = cand
        if best_seq is not None:
            return best_seq, best_cost
        return None, None

    def _best_swap_improvement(self, seq):
        """
        Try best pairwise swap improvement. Returns improved sequence and cost if found.
        """
        n = len(seq)
        if n <= 1:
            return None, None
        current_cost = self._makespan(seq)
        best_cost = current_cost
        best_seq = None
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                cost = self._makespan(cand)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best_seq = cand
        if best_seq is not None:
            return best_seq, best_cost
        return None, None

    def _local_search(self, seq, max_no_improve=50):
        """
        Apply local search combining insertion and swap (best-improvement) until no improvement.
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

        Returns:
            A list representing the sequence of jobs to be processed.
        """
        n = self.num_jobs
        if n == 0:
            return []
        # Initial solution by NEH heuristic
        best_seq = self._neh_initial()
        best_cost = self._makespan(best_seq)

        # Improve with local search
        best_seq, best_cost = self._local_search(best_seq, max_no_improve=50)

        # Iterated Local Search with perturbations guided by improvements
        start_time = time.time()
        it = 0
        stagnation = 0
        while it < self.max_iters and (time.time() - start_time) < self.time_limit and stagnation < 50:
            it += 1
            # adaptive perturbation strength
            strength = 1 + (it % max(1, n//5))
            candidate = self._perturb(best_seq, strength=strength)
            candidate, cand_cost = self._local_search(candidate, max_no_improve=30)
            if cand_cost < best_cost - 1e-12:
                best_seq = candidate
                best_cost = cand_cost
                stagnation = 0
            else:
                stagnation += 1

        # Final polishing
        best_seq, best_cost = self._local_search(best_seq, max_no_improve=100)

        # Ensure it's a permutation of all jobs
        if sorted(best_seq) != list(range(n)):
            # fallback to simple identity
            return list(range(n))
        return best_seq

