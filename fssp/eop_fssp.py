
import numpy as np
import random
import time
from copy import deepcopy

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
        self.processing_times = np.array(processing_times, dtype=float)
        # Precompute job totals and other lightweight stats
        self.job_totals = np.sum(self.processing_times, axis=1)
        # random seed for reproducibility could be set externally if desired
        random.seed()
        np.random.seed()

    # --- Utility: makespan computation for a permutation ---
    def makespan(self, perm):
        """
        Alternative makespan computation using a left-run accumulator.
        For each job we scan machines left-to-right maintaining:
          left = completion time on previous machine for current job (C[i,k-1])
        and update prev[k] in-place with:
          prev[k] = max(prev[k], left) + p[job,k]
        This reduces temporaries and branch work compared to the original.
        Returns final completion time (float).
        """
        m = self.num_machines
        if m == 0:
            return 0.0

        proc = self.processing_times  # local ref
        prev = [0.0] * m              # completion times for previous job on each machine

        for job in perm:
            # convert row to a Python list for fast indexed access
            row = proc[job].tolist()
            left = 0.0
            # update machines left-to-right
            for k in range(m):
                pk = row[k]
                vk = prev[k]
                if vk < left:
                    s = left + pk
                else:
                    s = vk + pk
                prev[k] = s
                left = s

        return float(prev[-1])

    # --- NEH heuristic for a strong initial solution ---
    def neh_init(self):
        jobs_sorted = sorted(range(self.num_jobs), key=lambda j: -self.job_totals[j])
        seq = []
        for j in jobs_sorted:
            best_seq = None
            best_cost = float('inf')
            # try inserting j at all positions
            for pos in range(len(seq)+1):
                cand = seq[:pos] + [j] + seq[pos:]
                cost = self.makespan(cand)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = cand
            seq = best_seq
        return seq, best_cost

    # --- simple local refinement: best-insertion improvement (bounded searches) ---
    def local_refine(self, perm, budget_iters=100):
        n = len(perm)
        best = perm[:]
        best_cost = self.makespan(best)
        improved = True
        iters = 0
        # Try single-job relocations (insertion) until budget or no improvement
        while improved and iters < budget_iters:
            improved = False
            iters += 1
            # iterate jobs in random order for diversification
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                job = best.pop(i)
                best_local_cost = float('inf')
                best_pos = 0
                # try insert at all positions
                for pos in range(n):
                    candidate = best[:pos] + [job] + best[pos:]
                    cost = self.makespan(candidate)
                    if cost < best_local_cost:
                        best_local_cost = cost
                        best_pos = pos
                # reinsert job
                best = best[:best_pos] + [job] + best[best_pos:]
                if best_local_cost + 1e-12 < best_cost:
                    best_cost = best_local_cost
                    improved = True
                    # small immediate break to re-evaluate neighbors
                    break
                # otherwise continue
            # end for
        return best, best_cost

    # --- permutation feature encoding for surrogate ---
    def perm_features(self, perm):
        # produce a fixed-length numeric feature vector for a permutation
        # Feature: for each job j, position index normalized times job_total
        pos = np.empty(self.num_jobs, dtype=float)
        for idx, job in enumerate(perm):
            pos[job] = idx / max(1, (self.num_jobs - 1))
        feat = pos * self.job_totals  # emphasizes heavy jobs' positions
        return feat  # shape (num_jobs,)

    # --- lightweight ridge regression surrogate ---
    class RidgeSurrogate:
        def __init__(self, dim, alpha=1e-3):
            self.alpha = alpha
            self.dim = dim
            self.coef = np.zeros(dim, dtype=float)
            self.intercept = 0.0
            self.trained = False

        def fit(self, X, y):
            # X shape (samples, dim), y shape (samples,)
            if len(y) == 0:
                self.trained = False
                return
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            # Solve (X^T X + alpha I) w = X^T y
            xtx = X.T.dot(X)
            reg = self.alpha * np.eye(self.dim)
            try:
                w = np.linalg.solve(xtx + reg, X.T.dot(y))
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(xtx + reg, X.T.dot(y), rcond=None)
            self.coef = w
            # compute intercept via mean residual
            pred = X.dot(self.coef)
            self.intercept = np.mean(y - pred)
            self.trained = True

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            if not self.trained:
                # fallback: use mean of job_totals scaled by position sum
                return np.full((X.shape[0],), 1e6)
            return X.dot(self.coef) + self.intercept

    # --- generate cascaded recombinants from bases ---
    def generate_cascaded(self, bases, num_offspring=40, window_scales=(2,4,8)):
        # bases: list of base permutations (lists)
        n = self.num_jobs
        offspring = []
        for _ in range(num_offspring):
            # choose one primary base
            base = random.choice(bases)
            # choose up to 2 other parents
            parents = [base] + random.sample(bases, k=min(len(bases)-1, 2))
            # create offspring start as base copy
            child = base[:]
            # hierarchical overlapping windows: for each scale choose a random start
            for scale in window_scales:
                w = min(scale, n)
                if w <= 1:
                    continue
                start = random.randint(0, n - w)
                # pick a parent to source this window
                src_parent = random.choice(parents)
                segment = src_parent[start:start + w]
                # Replace that window in child while preserving relative order of remaining elements
                # Remove segment elements from child
                remaining = [job for job in child if job not in segment]
                # Insert segment at the same start position (clamped)
                insert_pos = start
                if insert_pos > len(remaining):
                    insert_pos = len(remaining)
                child = remaining[:insert_pos] + segment + remaining[insert_pos:]
            # small shuffle with low probability inside child to increase diversity
            if random.random() < 0.1:
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                child[a], child[b] = child[b], child[a]
            # ensure valid permutation
            if len(set(child)) == n:
                offspring.append(child)
        # deduplicate
        unique = []
        seen = set()
        for c in offspring:
            key = tuple(c)
            if key not in seen:
                unique.append(c)
                seen.add(key)
        return unique

    # --- small perturbation: shuffle a large block of best ---
    def perturb(self, perm, intensity=0.3):
        n = len(perm)
        k = max(2, int(n * intensity))
        a = random.randint(0, n - k)
        block = perm[a:a+k]
        random.shuffle(block)
        pert = perm[:a] + block + perm[a+k:]
        # possibly also perform a random swap somewhere
        if random.random() < 0.5:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            pert[i], pert[j] = pert[j], pert[i]
        return pert

    # --- main solve method implementing the flow of pseudocode ---
    def solve(self) -> list:
        # Parameters (tunable)
        max_iters = max(60, 6 * self.num_jobs)  # iteration budget
        pop_init = min(40, 5 * self.num_jobs)
        archive_size = max(50, pop_init * 2)
        candidate_pool = 60
        triage_top = 12
        local_budget = 80
        stall_threshold = 12
        window_scales = [2, 4, max(4, self.num_jobs // 6)]
        start_time = time.time()
        time_limit = 5.0 + 0.05 * (self.num_jobs * self.num_machines)  # soft time budget (seconds)

        # --- Initialization Phase ---
        archive = {}  # map tuple(perm) -> makespan
        evaluated_X = []
        evaluated_y = []

        # NEH start
        try:
            neh_seq, neh_cost = self.neh_init()
            archive[tuple(neh_seq)] = neh_cost
            evaluated_X.append(self.perm_features(neh_seq))
            evaluated_y.append(neh_cost)
        except Exception:
            pass

        # Add some random and greedy initial permutations
        base_perm = list(range(self.num_jobs))
        # greedy by ascending totals and descending totals
        greedy_desc = sorted(range(self.num_jobs), key=lambda j: -self.job_totals[j])
        greedy_asc = sorted(range(self.num_jobs), key=lambda j: self.job_totals[j])
        for seq in [greedy_desc, greedy_asc, base_perm]:
            key = tuple(seq)
            if key not in archive:
                archive[key] = self.makespan(seq)
                evaluated_X.append(self.perm_features(seq))
                evaluated_y.append(archive[key])

        # random variations
        while len(archive) < pop_init:
            perm = list(range(self.num_jobs))
            random.shuffle(perm)
            key = tuple(perm)
            if key not in archive:
                archive[key] = self.makespan(perm)
                evaluated_X.append(self.perm_features(perm))
                evaluated_y.append(archive[key])

        # surrogate
        surrogate = FSSPSolver.RidgeSurrogate(self.num_jobs, alpha=1e-3)
        surrogate.fit(np.array(evaluated_X), np.array(evaluated_y))

        # adaptive acceptance parameters
        accept_worse_prob = 0.12
        best_key = min(archive.keys(), key=lambda k: archive[k])
        best_cost = archive[best_key]
        best_perm = list(best_key)

        no_improve = 0

        # Main loop
        for it in range(max_iters):
            # time check
            if time.time() - start_time > time_limit:
                break

            # select bases - pick top quality and diverse ones
            sorted_archive = sorted(archive.items(), key=lambda kv: kv[1])
            # select few best plus some diverse ones
            top_k = max(3, min(len(sorted_archive), 6))
            bases = [list(kv[0]) for kv in sorted_archive[:top_k]]
            # add some diverse top entries
            for kv in sorted_archive[top_k: min(len(sorted_archive), top_k + 10)]:
                if len(bases) >= 10:
                    break
                seq = list(kv[0])
                # ensure diversity by Hamming-like difference
                if all(sum(a != b for a, b in zip(seq, base)) > self.num_jobs // 4 for base in bases):
                    bases.append(seq)

            # generate cascaded recombinants
            offspring = self.generate_cascaded(bases, num_offspring=candidate_pool, window_scales=window_scales)

            # compute surrogate features and predictions
            feats = np.array([self.perm_features(p) for p in offspring])
            preds = surrogate.predict(feats)
            # rank by predicted makespan ascending
            order = np.argsort(preds)
            candidates = [offspring[i] for i in order[:min(triage_top, len(offspring))]]

            # evaluate top candidates truly
            improved_any = False
            evaluated_new = []
            for cand in candidates:
                cost = self.makespan(cand)
                evaluated_new.append((cand, cost))
                # local refinement on evaluated recombinants
                refined, refined_cost = self.local_refine(cand, budget_iters=local_budget // 6)
                if refined_cost + 1e-12 < cost:
                    cand = refined
                    cost = refined_cost
                # decide acceptance into archive
                key = tuple(cand)
                existing = archive.get(key)
                accept = False
                if existing is None or cost < existing - 1e-12:
                    accept = True
                else:
                    # probabilistic exploration acceptance
                    if random.random() < accept_worse_prob:
                        accept = True
                if accept:
                    archive[key] = cost
                    evaluated_new.append((cand, cost))
                    if cost + 1e-12 < best_cost:
                        best_cost = cost
                        best_perm = cand[:]
                        no_improve = 0
                        improved_any = True
            # update evaluated dataset and surrogate
            for cand, cost in evaluated_new:
                evaluated_X.append(self.perm_features(cand))
                evaluated_y.append(cost)
            # keep dataset bounded
            max_hist = max(200, 6 * self.num_jobs)
            if len(evaluated_y) > max_hist:
                evaluated_X = evaluated_X[-max_hist:]
                evaluated_y = evaluated_y[-max_hist:]
            surrogate.fit(np.array(evaluated_X), np.array(evaluated_y))

            # maintain archive size and diversity: keep best by cost but preserve some diverse
            items = sorted(archive.items(), key=lambda kv: kv[1])
            new_archive = {}
            # keep best half
            keep = min(archive_size // 2, len(items))
            for k, v in items[:keep]:
                new_archive[k] = v
            # fill remaining slots with diverse selections
            i = keep
            while len(new_archive) < archive_size and i < len(items):
                k, v = items[i]
                seq = list(k)
                # ensure some diversity: accept only if different enough from existing
                diffs = []
                for exk in new_archive.keys():
                    exseq = list(exk)
                    diffs.append(sum(a != b for a, b in zip(seq, exseq)))
                if not diffs or max(diffs) >= max(1, self.num_jobs // 6):
                    new_archive[k] = v
                else:
                    # occasionally include to maintain quality
                    if random.random() < 0.08:
                        new_archive[k] = v
                i += 1
            archive = new_archive

            # adapt acceptance probability slightly based on progress
            if improved_any:
                accept_worse_prob = max(0.02, accept_worse_prob * 0.9)
                no_improve = 0
            else:
                accept_worse_prob = min(0.25, accept_worse_prob * 1.06)
                no_improve += 1

            # targeted large-scale perturbation if stalled
            if no_improve >= stall_threshold:
                # perturb current best and add to archive after refinement
                pert = self.perturb(best_perm, intensity=0.4)
                pert_refined, pert_cost = self.local_refine(pert, budget_iters=local_budget)
                key = tuple(pert_refined)
                archive[key] = pert_cost
                evaluated_X.append(self.perm_features(pert_refined))
                evaluated_y.append(pert_cost)
                surrogate.fit(np.array(evaluated_X), np.array(evaluated_y))
                no_improve = 0

        # Post-Processing: intensify polishing on top candidates
        final_items = sorted(archive.items(), key=lambda kv: kv[1])
        top_candidates = [list(kv[0]) for kv in final_items[:min(6, len(final_items))]]
        for cand in top_candidates:
            refined, refined_cost = self.local_refine(cand, budget_iters=200)
            if refined_cost + 1e-12 < best_cost:
                best_cost = refined_cost
                best_perm = refined

        # final sanity: ensure permutation length correct
        if len(best_perm) != self.num_jobs or len(set(best_perm)) != self.num_jobs:
            # fallback to simple sequence
            best_perm = list(range(self.num_jobs))

        return best_perm

