#
# ALGORITHM Adaptive Cooperative Substructure Search (ACSS)
#


# Improved in knowledge transfer
import numpy as np
import random
import time
from collections import Counter, defaultdict
from copy import deepcopy

class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix

        # Tunable parameters (can be adjusted)
        self.population_size = 8
        self.max_iters = 1000
        self.time_limit = 10.0  # seconds
        self.random_seed = None
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    # ---------------- Helper functions ----------------
    def _tour_length(self, tour):
        # tour: list of node indices length n
        n = len(tour)
        if n == 0:
            return 0.0
        dist = self.distance_matrix
        total = 0.0
        for i in range(n):
            a = tour[i]
            b = tour[(i+1) % n]
            total += dist[a, b]
        return float(total)

    def _nearest_neighbor(self, start=0):
        n = len(self.coordinates)
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)
        cur = start
        dist = self.distance_matrix
        while unvisited:
            # choose nearest
            nxt = min(unvisited, key=lambda v: dist[cur, v])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    def _random_tour(self):
        n = len(self.coordinates)
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    def _compute_edge_counts(self, tours_list, top_k=None):
        counts = Counter()
        for tour in tours_list:
            n = len(tour)
            if n == 0:
                continue
            for i in range(n):
                a = tour[i]
                b = tour[(i+1) % n]
                counts[(a, b)] += 1
        if top_k:
            return counts.most_common(top_k)
        return counts

    def _apply_2opt_tour(self, tour, memory_edges=None, max_iter=1000, penalty_alpha=1e-4):
        """
        2-opt for full tour (cycle). Uses first-improvement strategy and optional memory penalty to discourage breaking frequent edges.
        """
        if tour is None:
            return tour
        n = len(tour)
        if n <= 3:
            return tour[:]
        best = tour[:]
        dist = self.distance_matrix
        mem_get = memory_edges.get if memory_edges is not None else (lambda k, default=0: default)
        alpha = float(penalty_alpha)

        def total_length(lst):
            return self._tour_length(lst)

        best_len = total_length(best)
        it = 0
        while it < max_iter:
            it += 1
            improved = False
            # iterate over i,j for possible 2-opt moves
            for i in range(0, n-1):
                a = best[i]
                b = best[(i+1) % n]
                dist_ab = dist[a, b]
                rem_ab = mem_get((a, b), 0)
                row_a = dist[a]
                row_b = dist[b]
                # j must be at least i+2 and not equal to i (skip adjacent), and avoid full reversal when i==0 and j==n-1
                for j in range(i+2, n):
                    if i == 0 and j == n-1:
                        continue
                    c = best[j]
                    d = best[(j+1) % n]
                    # compute delta
                    delta = float(row_a[c] + row_b[d] - dist_ab - dist[c, d])
                    rem_cd = mem_get((c, d), 0)
                    if delta + alpha * (rem_ab + rem_cd) < -1e-9:
                        # perform reversal of segment (i+1 .. j) inclusive
                        new_best = best[:i+1] + list(reversed(best[i+1:j+1])) + best[j+1:]
                        best = new_best
                        best_len += delta
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def _cheapest_insertion_position(self, tour, node):
        # returns (pos, inc_cost)
        n = len(tour)
        dist = self.distance_matrix
        if n == 0:
            return 0, 0.0
        best_inc = float('inf')
        best_pos = 0
        for pos in range(n):
            prev = tour[pos]
            nxt = tour[(pos+1) % n]
            inc = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]
            if inc < best_inc:
                best_inc = inc
                best_pos = pos+1  # insert after prev => at index pos+1
        # also consider inserting at beginning (which is equivalent to pos = n if we keep list)
        # Already handled by pos==n-1 wrap to insert after last element.
        return best_pos % (n+1), best_inc

    def _perturb(self, tour, strength=1):
        """
        Remove 'k' random nodes and reinsert by cheapest insertion (diversify).
        """
        n = len(tour)
        if n <= 3:
            return tour[:]
        k = min(max(1, int(strength)), n-1)
        removed = set(random.sample(tour, k))
        remaining = [v for v in tour if v not in removed]
        orphans = [v for v in tour if v in removed]
        random.shuffle(orphans)
        # reinsert each orphan by cheapest insertion
        for node in orphans:
            if not remaining:
                remaining = [node]
                continue
            # try all insertion positions 0..len(remaining)
            best_inc = float('inf')
            best_pos = 0
            m = len(remaining)
            dist = self.distance_matrix
            for pos in range(m):
                prev = remaining[pos]
                nxt = remaining[(pos+1) % m]
                inc = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos + 1
            # insert at best_pos (mod m+1)
            if best_pos >= 0 and best_pos <= m:
                remaining.insert(best_pos, node)
            else:
                remaining.append(node)
        return remaining

    def _recombine_with_memory(self, tour, memory_edges, max_attempts=50):
        """
        Try to enforce frequently occurring directed edges by moving nodes so that (a,b) appear adjacent a->b.
        """
        if not memory_edges:
            return tour
        tour = tour[:]
        n = len(tour)
        pos_map = {v: i for i, v in enumerate(tour)}
        # sort edges by frequency desc
        edges_sorted = sorted(memory_edges.items(), key=lambda x: -x[1])
        attempts = 0
        for (a, b), freq in edges_sorted:
            if attempts >= max_attempts:
                break
            if a not in pos_map or b not in pos_map:
                continue
            pa = pos_map[a]
            pb = pos_map[b]
            # already adjacent a->b?
            if (pa + 1) % n == pb:
                continue
            # Move b to position after a
            # remove b at pb
            tour.pop(pb)
            # adjust pa if pb < pa
            if pb < pa:
                pa -= 1
            insert_pos = (pa + 1)
            tour.insert(insert_pos, b)
            # rebuild map
            pos_map = {v: i for i, v in enumerate(tour)}
            attempts += 1
        return tour

    # ---------------- Main solve method ----------------
    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP).

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        n = len(self.coordinates)
        if n <= 1:
            return np.arange(n)
        if n == 2:
            return np.array([0,1])

        start_time = time.time()

        # Build initial population
        population = []
        population_costs = []
        for i in range(self.population_size):
            if i % 2 == 0:
                start = random.randrange(n)
                tour = self._nearest_neighbor(start=start)
            else:
                tour = self._random_tour()
            # small random perturb
            if random.random() < 0.5:
                tour = self._perturb(tour, strength=random.randint(1, max(1, n//10)))
            tour_len = self._tour_length(tour)
            population.append(tour)
            population_costs.append(tour_len)

        # cooperative memory: edge frequencies from top solutions
        def update_memory(pop):
            sols = [p for p in pop]
            return self._compute_edge_counts(sols)

        cooperative_memory = update_memory(population)

        best_idx = int(np.argmin(population_costs))
        best_tour = population[best_idx][:]
        best_cost = population_costs[best_idx]

        iter_count = 0
        while iter_count < self.max_iters and (time.time() - start_time) < self.time_limit:
            iter_count += 1

            # selection: bias to good but keep diversity
            if random.random() < 0.6:
                sorted_idx = sorted(range(len(population)), key=lambda i: population_costs[i])
                sel_candidates = sorted_idx[:max(1, len(population)//2)]
                chosen = random.choice(sel_candidates)
            else:
                chosen = random.randrange(len(population))
            current = deepcopy(population[chosen])
            current_cost = population_costs[chosen]

            # Intensify: guided 2-opt
            memory_edges = cooperative_memory
            candidate = self._apply_2opt_tour(current, memory_edges=memory_edges, max_iter=200, penalty_alpha=1e-4)

            # Diversify: perturb with adaptive strength
            strength = 1 + int(iter_count ** 0.5) % max(1, n//10)
            if random.random() < 0.5:
                pert_strength = random.randint(1, max(1, strength))
            else:
                pert_strength = random.randint(1, max(1, n//6))
            candidate = self._perturb(candidate, strength=pert_strength)

            # Recombine with memory
            if random.random() < 0.6:
                candidate = self._recombine_with_memory(candidate, cooperative_memory)

            # Final local improvement
            candidate = self._apply_2opt_tour(candidate, memory_edges=cooperative_memory, max_iter=500, penalty_alpha=1e-5)
            cand_cost = self._tour_length(candidate)

            replaced = False
            # Replacement: if better than chosen, replace; otherwise occasional replace worst
            if cand_cost + 1e-9 < current_cost:
                population[chosen] = candidate
                population_costs[chosen] = cand_cost
                replaced = True
            else:
                if random.random() < 0.05:
                    worst_idx = int(np.argmax(population_costs))
                    population[worst_idx] = candidate
                    population_costs[worst_idx] = cand_cost
                    replaced = True

            # Update cooperative memory using top solutions
            sorted_idx = sorted(range(len(population)), key=lambda i: population_costs[i])
            top_k = [population[i] for i in sorted_idx[:min(4, len(population))]]
            cooperative_memory = self._compute_edge_counts(top_k)

            # Update best
            cur_best_idx = int(np.argmin(population_costs))
            cur_best_cost = population_costs[cur_best_idx]
            if cur_best_cost + 1e-9 < best_cost:
                best_cost = cur_best_cost
                best_tour = population[cur_best_idx][:]

        # Post-processing / polishing on best solution
        polished = self._apply_2opt_tour(best_tour, memory_edges=cooperative_memory, max_iter=2000, penalty_alpha=1e-6)
        polished_cost = self._tour_length(polished)
        if polished_cost + 1e-9 < best_cost:
            best_tour = polished
            best_cost = polished_cost

        # Ensure output is a numpy array permutation of 0..n-1
        final = np.array(best_tour, dtype=int)
        return final

