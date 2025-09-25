
# --- BPP Task Description and Template Program ---

task_description = """
You are given a set of items, each with a specific weight, and a number of identical bins, each with a fixed capacity.
The goal is to pack all items into the minimum number of bins possible, such that the sum of the weights of the items in each bin does not exceed the bin's capacity.
"""


template_program = """

import time
import random
import math
from collections import defaultdict
import numpy as np

class BPPSolver:
    def __init__(self, capacity: int, weights: list[int | float], *,
                 max_iters: int = 100000, pool_size: int = 12, time_limit: float = 60.0, random_seed: int | None = None):
        \"\"\"
        Initialize the BPP solver.

        Args:
            capacity (int): The capacity of each bin.
            weights (list[int | float]): A list of item weights.
            max_iters (int): Maximum number of main-loop iterations.
            pool_size (int): Number of candidate solutions to maintain.
            time_limit (float): Soft time limit in seconds to stop early.
            random_seed (int|None): Seed for reproducibility.
        \"\"\"
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.capacity = float(capacity)
        self.weights = [float(w) for w in weights]
        self.num_items = len(weights)
        self.max_iters = max_iters
        self.pool_size = max(4, pool_size)
        self.time_limit = time_limit

        # Adaptive controller
        self.operator_names = ['ffd', 'bfd', 'move', 'swap', 'repack', 'random_reassign']
        self.op_stats = {op: {'tries': 0, 'suc': 0} for op in self.operator_names}

        # Bookkeeping
        self.best_solution = None
        self.best_bins = math.inf

    # -------------------------
    # Utility functions
    # -------------------------
    def _bin_loads(self, solution):
        return [sum(self.weights[i] for i in b) for b in solution]

    def _is_feasible(self, solution):
        return all(load <= self.capacity + 1e-9 for load in self._bin_loads(solution))

    def _evaluate(self, solution):
        # primary metric: number of bins, secondary: total overflow penalty if infeasible
        loads = self._bin_loads(solution)
        over = sum(max(0.0, l - self.capacity) for l in loads)
        return len(solution), over

    def _normalize_solution(self, solution):
        # remove empty bins and sort items inside bins; maintain determinism
        s = [list(b) for b in solution if len(b) > 0]
        for b in s:
            b.sort()
        return s

    def _copy_solution(self, sol):
        return [list(b) for b in sol]

    # -------------------------
    # Constructors (initial pool)
    # -------------------------
    def _ffd(self, items_order=None):
        if items_order is None:
            items = sorted(range(self.num_items), key=lambda i: self.weights[i], reverse=True)
        else:
            items = list(items_order)
        bins = []
        loads = []
        for it in items:
            placed = False
            for j in range(len(bins)):
                if loads[j] + self.weights[it] <= self.capacity + 1e-9:
                    bins[j].append(it)
                    loads[j] += self.weights[it]
                    placed = True
                    break
            if not placed:
                bins.append([it])
                loads.append(self.weights[it])
        return self._normalize_solution(bins)

    def _bfd(self, items_order=None):
        if items_order is None:
            items = sorted(range(self.num_items), key=lambda i: self.weights[i], reverse=True)
        else:
            items = list(items_order)
        bins = []
        loads = []
        for it in items:
            best_j = -1
            best_remain = float('inf')
            for j in range(len(bins)):
                if loads[j] + self.weights[it] <= self.capacity + 1e-9:
                    remain = self.capacity - (loads[j] + self.weights[it])
                    if remain < best_remain:
                        best_remain = remain
                        best_j = j
            if best_j >= 0:
                bins[best_j].append(it)
                loads[best_j] += self.weights[it]
            else:
                bins.append([it])
                loads.append(self.weights[it])
        return self._normalize_solution(bins)

    def _random_greedy(self):
        items = list(range(self.num_items))
        random.shuffle(items)
        return self._ffd(items_order=items)

    def _initial_pool(self):
        pool = []
        # deterministic strong heuristics
        pool.append(self._ffd())
        pool.append(self._bfd())
        # some randomized variants
        for _ in range(self.pool_size - 4):
            if random.random() < 0.5:
                pool.append(self._random_greedy())
            else:
                items = sorted(range(self.num_items), key=lambda i: (random.random(), -self.weights[i]))
                pool.append(self._ffd(items_order=items))
        # small local improvements
        for i in range(len(pool)):
            pool[i] = self._local_improve(pool[i], max_no_improve=10)
        # unique and limit
        uniq = []
        seen = set()
        for s in pool:
            key = tuple(sorted(tuple(sorted(b)) for b in s))
            if key not in seen:
                uniq.append(s)
                seen.add(key)
            if len(uniq) >= self.pool_size:
                break
        return uniq

    # -------------------------
    # Local operators
    # -------------------------
    def _select_operator(self):
        # probability proportional to success rate (Laplace-smoothed)
        weights = []
        for op in self.operator_names:
            stat = self.op_stats[op]
            score = (stat['suc'] + 1.0) / (stat['tries'] + 1.0)
            weights.append(score)
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(self.operator_names, probs, k=1)[0]

    def _op_mark(self, op, success: bool):
        st = self.op_stats[op]
        st['tries'] += 1
        if success:
            st['suc'] += 1

    def _move_item(self, sol, allow_over=False):
        # try move one item from a heavier bin to another to reduce bins or fix overflow
        s = self._copy_solution(sol)
        loads = self._bin_loads(s)
        if not s:
            return s
        # select donor bin (largest load) and try to move one of its items
        donor_idx = max(range(len(s)), key=lambda j: loads[j])
        donor = s[donor_idx]
        if not donor:
            return s
        # try items largest first
        items_sorted = sorted(donor, key=lambda i: self.weights[i], reverse=True)
        for it in items_sorted:
            # try put into some other bin
            placed = False
            for j in range(len(s)):
                if j == donor_idx: continue
                if loads[j] + self.weights[it] <= self.capacity + (0.0 if not allow_over else self.capacity * 0.05):
                    s[donor_idx].remove(it)
                    s[j].append(it)
                    return self._normalize_solution(s)
            # else try opening new bin (worse)
        return s

    def _swap_items(self, sol):
        # swap two items across bins to try to fit better
        s = self._copy_solution(sol)
        loads = self._bin_loads(s)
        nb = len(s)
        if nb < 2:
            return s
        # pick two random bins (pref heavier)
        a, b = sorted(random.sample(range(nb), 2), key=lambda x: -loads[x])
        best_gain = 0
        best_pair = None
        for ia in s[a]:
            for ib in s[b]:
                new_la = loads[a] - self.weights[ia] + self.weights[ib]
                new_lb = loads[b] - self.weights[ib] + self.weights[ia]
                # both should be feasible
                if new_la <= self.capacity + 1e-9 and new_lb <= self.capacity + 1e-9:
                    # prefer pairs that reduce max load or produce more even distribution
                    gain = (loads[a] + loads[b]) - (new_la + new_lb)
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (ia, ib)
        if best_pair:
            ia, ib = best_pair
            s[a].remove(ia); s[a].append(ib)
            s[b].remove(ib); s[b].append(ia)
        return self._normalize_solution(s)

    def _repack_local(self, sol):
        # select a subset of bins, unpack their items and repack with FFD
        s = self._copy_solution(sol)
        nb = len(s)
        if nb <= 1:
            return s
        k = max(1, min(4, int(math.ceil(nb * 0.25))))
        chosen = set(random.sample(range(nb), k))
        items = []
        for i in sorted(chosen, reverse=True):
            items.extend(s[i])
            del s[i]
        # repack items with FFD into existing bins first then new
        loads = self._bin_loads(s)
        # sort items by weight desc
        items = sorted(items, key=lambda x: self.weights[x], reverse=True)
        for it in items:
            placed = False
            for j in range(len(s)):
                if loads[j] + self.weights[it] <= self.capacity + 1e-9:
                    s[j].append(it)
                    loads[j] += self.weights[it]
                    placed = True
                    break
            if not placed:
                s.append([it])
                loads.append(self.weights[it])
        return self._normalize_solution(s)

    def _random_reassign(self, sol):
        # randomly move small subset of items to random bins / new bin then repair
        s = self._copy_solution(sol)
        nb = len(s)
        if self.num_items == 0:
            return s
        k = max(1, int(0.05 * self.num_items))
        items = random.sample(range(self.num_items), k)
        # remove these items
        for it in items:
            for b in s:
                if it in b:
                    b.remove(it)
                    break
        # try place them randomly
        loads = self._bin_loads(s)
        for it in items:
            placed = False
            order = list(range(len(s)))
            random.shuffle(order)
            for j in order:
                if loads[j] + self.weights[it] <= self.capacity + 1e-9:
                    s[j].append(it)
                    loads[j] += self.weights[it]
                    placed = True
                    break
            if not placed:
                s.append([it])
                loads.append(self.weights[it])
        return self._normalize_solution(s)

    # -------------------------
    # Negotiation / Repair
    # -------------------------
    def _repair_overfilled(self, sol, allow_slack_ratio=0.0):
        # Greedy repair: move items out of overfull bins to bins with space or create new bins.
        # allow_slack_ratio permits temporary slight overflow; then we try to remove.
        s = self._copy_solution(sol)
        loads = self._bin_loads(s)
        slack = allow_slack_ratio * self.capacity
        # loop until no overfull or stable
        iterations = 0
        while True:
            iterations += 1
            changed = False
            loads = self._bin_loads(s)
            overfilled = [i for i, l in enumerate(loads) if l > self.capacity + 1e-9]
            if not overfilled or iterations > 5 + len(overfilled):
                break
            # handle largest overfull first
            overfilled.sort(key=lambda x: loads[x] - self.capacity, reverse=True)
            for idx in overfilled:
                # try moving largest items out
                items_sorted = sorted(s[idx], key=lambda it: self.weights[it], reverse=True)
                moved_any = False
                for it in items_sorted:
                    # find target bin with space minimizing leftover
                    best_j = -1
                    best_rem = float('inf')
                    for j in range(len(s)):
                        if j == idx:
                            continue
                        rem = self.capacity - (loads[j] + self.weights[it])
                        if rem >= -1e-9 and rem < best_rem:
                            best_rem = rem
                            best_j = j
                    if best_j >= 0:
                        s[idx].remove(it)
                        s[best_j].append(it)
                        changed = True
                        moved_any = True
                        loads[idx] -= self.weights[it]
                        loads[best_j] += self.weights[it]
                        if loads[idx] <= self.capacity + slack + 1e-9:
                            break
                if not moved_any:
                    # try creating a new bin for a large item
                    it = items_sorted[0]
                    s[idx].remove(it)
                    s.append([it])
                    changed = True
                    loads = self._bin_loads(s)
                # continue to next overfilled
            if not changed:
                break
        # finally, cleanup empty bins
        s = self._normalize_solution(s)
        return s

    # -------------------------
    # Memetic recombination
    # -------------------------
    def _memetic_combine(self, a, b):
        # Take some bins from parent a and the rest try to fill remaining items using parent b ordering then FFD
        A = [list(bin) for bin in a]
        B = [list(bin) for bin in b]
        na = len(A)
        if na == 0:
            return self._ffd()
        take_k = random.randint(1, max(1, min(na, int(math.ceil(na * 0.5)))))
        indices = list(range(na))
        random.shuffle(indices)
        take = set(indices[:take_k])
        new_bins = []
        taken_items = set()
        for i in range(na):
            if i in take:
                new_bins.append(list(A[i]))
                taken_items.update(A[i])
        # remaining items from both parents
        remaining = [i for i in range(self.num_items) if i not in taken_items]
        # preference ordering: items that appear together in B are placed earlier
        order = []
        for bin in B:
            for it in bin:
                if it in remaining and it not in order:
                    order.append(it)
        # append any left
        for it in remaining:
            if it not in order:
                order.append(it)
        # pack remaining into existing new_bins first then FFD new bins
        loads = [sum(self.weights[i] for i in b) for b in new_bins]
        for it in sorted(order, key=lambda x: self.weights[x], reverse=True):
            placed = False
            best_j = -1
            best_rem = float('inf')
            for j in range(len(new_bins)):
                if loads[j] + self.weights[it] <= self.capacity + 1e-9:
                    rem = self.capacity - (loads[j] + self.weights[it])
                    if rem < best_rem:
                        best_rem = rem
                        best_j = j
            if best_j >= 0:
                new_bins[best_j].append(it)
                loads[best_j] += self.weights[it]
            else:
                new_bins.append([it])
                loads.append(self.weights[it])
        return self._normalize_solution(new_bins)

    # -------------------------
    # Post-processing local search
    # -------------------------
    def _local_improve(self, sol, max_no_improve=50):
        s = self._copy_solution(sol)
        best_bins, _ = self._evaluate(s)
        no_improve = 0
        while no_improve < max_no_improve:
            # try remove a bin by moving its items
            nb = len(s)
            improved = False
            if nb <= 1:
                break
            # try bins with small loads first
            loads = self._bin_loads(s)
            cand = sorted(range(nb), key=lambda j: loads[j])
            for idx in cand:
                items = list(s[idx])
                feasible = True
                tmp_loads = loads[:]
                tmp_loads[idx] = 0.0
                # check if we can place all items into other bins
                for it in sorted(items, key=lambda x: self.weights[x], reverse=True):
                    placed = False
                    for j in range(nb):
                        if j == idx: continue
                        if tmp_loads[j] + self.weights[it] <= self.capacity + 1e-9:
                            tmp_loads[j] += self.weights[it]
                            placed = True
                            break
                    if not placed:
                        feasible = False
                        break
                if feasible:
                    # commit
                    for it in items:
                        for j in range(nb):
                            if j == idx: continue
                            if sum(self.weights[i] for i in s[j]) + self.weights[it] <= self.capacity + 1e-9:
                                s[j].append(it)
                                break
                    s[idx] = []
                    s = self._normalize_solution(s)
                    improved = True
                    break
            if improved:
                newbins, _ = self._evaluate(s)
                if newbins < best_bins:
                    best_bins = newbins
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        return s

    # -------------------------
    # Pool management
    # -------------------------
    def _pool_update(self, pool, candidate):
        # pool: list of solutions
        # keep top self.pool_size solutions by num_bins (feasible preferred) and diversity
        cand_norm = self._normalize_solution(candidate)
        key = tuple(sorted(tuple(sorted(b)) for b in cand_norm))
        # avoid duplicates
        for s in pool:
            if tuple(sorted(tuple(sorted(b)) for b in s)) == key:
                return pool
        pool.append(cand_norm)
        # sort: feasible then fewer bins then lower overflow
        pool.sort(key=lambda s: (self._evaluate(s)[1] > 1e-9, self._evaluate(s)[0], self._evaluate(s)[1]))
        # keep unique up to pool_size
        unique = []
        seen = set()
        for s in pool:
            k = tuple(sorted(tuple(sorted(b)) for b in s))
            if k not in seen:
                seen.add(k)
                unique.append(s)
            if len(unique) >= self.pool_size:
                break
        return unique

    # -------------------------
    # Strategic moves
    # -------------------------
    def _diversify(self, pool):
        # insert a few randomized solutions to escape stagnation
        for _ in range(2):
            s = self._random_greedy()
            pool = self._pool_update(pool, s)
        return pool

    # -------------------------
    # Main solver
    # -------------------------
    def solve(self) -> list[list[int]]:
        \"\"\"
        Solve the Bin Packing Problem.

        Returns:
            A list of lists, where each inner list represents a bin and contains the
            original indices of the items packed into it.
        \"\"\"
        start_time = time.time()
        pool = self._initial_pool()

        # initialize best
        for s in pool:
            bins_count, over = self._evaluate(s)
            if over <= 1e-9 and bins_count < self.best_bins:
                self.best_bins = bins_count
                self.best_solution = s

        stagnation = 0
        last_best = self.best_bins if self.best_solution is not None else math.inf

        for it in range(self.max_iters):
            print(it)
            if time.time() - start_time > self.time_limit:
                break

            # selection: pick a few candidates favoring good solutions and diversity
            k = max(2, min(len(pool), max(2, int(math.sqrt(len(pool)*2)))))
            # probability inverse to bins (fewer bins -> higher prob)
            qualities = [1.0 / (self._evaluate(s)[0] + 0.1) for s in pool]
            total_q = sum(qualities)
            probs = [q / total_q for q in qualities]
            selected_indices = np.random.choice(len(pool), size=k, replace=False, p=probs).tolist()

            new_candidates = []
            for idx in selected_indices:
                parent = pool[idx]
                op = self._select_operator()
                variant = None
                success_flag = False
                if op == 'ffd':
                    # repack entire instance guided by parent's bin ordering
                    order = []
                    for b in parent:
                        order.extend(sorted(b, key=lambda x: -self.weights[x]))
                    variant = self._ffd(items_order=order)
                elif op == 'bfd':
                    order = []
                    for b in parent:
                        order.extend(sorted(b, key=lambda x: -self.weights[x]))
                    variant = self._bfd(items_order=order)
                elif op == 'move':
                    variant = self._move_item(parent, allow_over=True)
                elif op == 'swap':
                    variant = self._swap_items(parent)
                elif op == 'repack':
                    variant = self._repack_local(parent)
                elif op == 'random_reassign':
                    variant = self._random_reassign(parent)
                else:
                    variant = self._random_greedy()

                # negotiation / repair: allow small slack temporarily
                repaired = self._repair_overfilled(variant, allow_slack_ratio=0.05)
                # memetic recombination occasionally
                if random.random() < 0.35:
                    partner = pool[random.randrange(len(pool))]
                    child = self._memetic_combine(repaired, partner)
                    repaired = self._repair_overfilled(child, allow_slack_ratio=0.02)

                # final polishing
                repaired = self._local_improve(repaired, max_no_improve=6)

                # evaluate
                bins_count, over = self._evaluate(repaired)
                success_flag = (over <= 1e-9 and bins_count <= self._evaluate(parent)[0])
                self._op_mark(op, success_flag)

                # update best
                if over <= 1e-9 and bins_count < self.best_bins:
                    self.best_bins = bins_count
                    self.best_solution = repaired
                    stagnation = 0
                new_candidates.append(repaired)

            # update pool with new candidates
            for c in new_candidates:
                pool = self._pool_update(pool, c)

            # strategic moves based on stagnation
            if self.best_solution is None:
                stagnation += 1
            else:
                if self.best_bins == last_best:
                    stagnation += 1
                else:
                    stagnation = 0
                    last_best = self.best_bins

            if stagnation > 25 and random.random() < 0.6:
                pool = self._diversify(pool)
                stagnation = 0

            # slight chance of heavy random restart if very stagnated
            if stagnation > 80:
                pool = self._initial_pool()
                stagnation = 0

        # Post-processing: concentrated polishing attempt to remove bins
        if self.best_solution is not None:
            polished = self._local_improve(self.best_solution, max_no_improve=200)
            if self._evaluate(polished)[0] <= self._evaluate(self.best_solution)[0] and self._is_feasible(polished):
                self.best_solution = polished

        # If still no feasible found (shouldn't happen), return a basic FFD
        if self.best_solution is None:
            self.best_solution = self._ffd()

        # Final cleanup
        final = self._normalize_solution(self.best_solution)
        return final
       
"""