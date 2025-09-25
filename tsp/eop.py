import numpy as np
import random
import time
from collections import defaultdict, deque

class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
        """
        self.coordinates = np.asarray(coordinates)
        self.distance_matrix = np.asarray(distance_matrix)
        self.n = len(self.coordinates)

    # ---------- Helper utilities ----------
    def tour_length(self, tour):
        n = self.n
        dist = 0.0
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            dist += self.distance_matrix[a, b]
        return dist

    def edge_key(self, a, b):
        return (a, b) if a <= b else (b, a)

    def get_edges_set(self, tour):
        n = len(tour)
        edges = set()
        for i in range(n):
            a = int(tour[i])
            b = int(tour[(i + 1) % n])
            edges.add(self.edge_key(a, b))
        return edges

    def nearest_neighbor(self, start):
        n = self.n
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)
        cur = start
        while unvisited:
            next_city = min(unvisited, key=lambda j: self.distance_matrix[cur, j])
            tour.append(next_city)
            unvisited.remove(next_city)
            cur = next_city
        return tour

    def cheapest_insertion(self):
        n = self.n
        nodes = list(range(n))
        random.shuffle(nodes)
        tour = nodes[:3]
        tour = list(dict.fromkeys(tour))  # ensure unique
        if len(tour) < 3:
            for v in nodes:
                if v not in tour:
                    tour.append(v)
                if len(tour) == 3:
                    break
        remaining = [v for v in nodes if v not in tour]
        while remaining:
            best_cost = float('inf')
            best_pos = None
            best_v = None
            for v in remaining:
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i + 1) % len(tour)]
                    cost = (self.distance_matrix[a, v] + self.distance_matrix[v, b] - self.distance_matrix[a, b])
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1
                        best_v = v
            tour.insert(best_pos, best_v)
            remaining.remove(best_v)
        return tour

    def two_opt(self, tour, max_iter=100, fixed_edges=None):
        """
        Best-improvement 2-opt local search.

        This variation scans all valid 2-opt exchanges each iteration and applies the
        single best-improving swap (largest decrease in tour length). It tends to
        produce better local optima than a first-improvement strategy at the cost
        of doing more work per iteration.

        Preserves the same signature and respects fixed_edges (edges that cannot
        be removed). Returns a new tour list (a copy of the input).
        """
        if fixed_edges is None:
            fixed_edges = set()

        # Normalize fixed edges to sorted tuples for O(1) membership checks
        fixed_local = set()
        for (u, v) in fixed_edges:
            fixed_local.add((u, v) if u <= v else (v, u))

        tour = list(tour)
        n = len(tour)
        if n < 4 or max_iter <= 0:
            return tour

        dist = self.distance_matrix  # local alias
        eps = 1e-9

        # Main loop: perform up to max_iter best-improvement 2-opt moves
        for _ in range(max_iter):
            best_delta = -eps
            best_i = best_j = None

            # Scan all valid (i, j) pairs for 2-opt
            # i indexes a, b = tour[i], tour[i+1]
            for i in range(0, n - 2):
                a = tour[i]
                b = tour[i + 1]
                key_ab = (a, b) if a <= b else (b, a)
                if key_ab in fixed_local:
                    continue
                dab = dist[a, b]

                # j indexes c = tour[j], d = tour[(j+1) % n]; ensure at least one node between b and c
                for j in range(i + 2, n):
                    # Skip the trivial wrap-around that would recreate the same edge
                    if j == n - 1 and i == 0:
                        continue

                    c = tour[j]
                    d = tour[(j + 1) % n]
                    key_cd = (c, d) if c <= d else (d, c)
                    if key_cd in fixed_local:
                        continue

                    # compute delta = new_edges - removed_edges
                    delta = (dist[a, c] + dist[b, d]) - (dab + dist[c, d])

                    # track the best (most negative) delta
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j

            # If we found an improving exchange, apply the best one
            if best_i is not None:
                i = best_i
                j = best_j
                # perform 2-opt: reverse segment between i+1 .. j (inclusive)
                tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                # continue to next iteration to try further improvements
                continue
            else:
                # no improving move found -> local optimum
                break

        return tour

    def randomized_two_opt(self, tour, fixed_edges=None, attempts=50):
        best = list(tour)
        best_len = self.tour_length(best)
        for _ in range(attempts):
            i = random.randrange(0, self.n)
            j = random.randrange(0, self.n)
            if i == j:
                continue
            a, b = min(i, j), max(i, j)
            candidate = list(tour)
            candidate[a + 1:b + 1] = reversed(candidate[a + 1:b + 1])
            candidate = self.two_opt(candidate, max_iter=20, fixed_edges=fixed_edges)
            l = self.tour_length(candidate)
            if l < best_len:
                best_len = l
                best = candidate
        return best

    # ---------- Multiscale decomposition ----------
    def make_patches(self, scale_k):
        """
        Partition nodes into scale_k patches using 1D split on principal axis (alternating).
        Returns list of lists of node indices.
        """
        nodes = list(range(self.n))
        coords = self.coordinates
        # pick axis by variance
        varx = np.var(coords[:, 0])
        vary = np.var(coords[:, 1])
        axis = 0 if varx >= vary else 1
        sorted_nodes = sorted(nodes, key=lambda i: coords[i, axis])
        patches = []
        sz = max(1, len(sorted_nodes) // scale_k)
        for i in range(scale_k):
            start = i * sz
            end = (i + 1) * sz if i < scale_k - 1 else len(sorted_nodes)
            patch = sorted_nodes[start:end]
            if patch:
                patches.append(patch)
        return patches

    # ---------- Main algorithm (AM-CEM inspired) ----------
    def solve(self) -> np.ndarray:
        n = self.n
        # quick trivial cases
        if n <= 2:
            return np.arange(n)

        # Parameters (adaptive)
        ensemble_size = min(12, max(6, n // 4))
        max_iters = max(100, 6 * n)
        stagnation_limit = max(20, n // 2)
        start_time = time.time()
        time_limit = max(1.0, min(10.0, 0.1 * n))  # prefer a small time cap

        # Build initial ensemble of candidate tours
        ensemble = []
        # a few nearest neighbor with different starts
        seeds = list(range(n))
        random.shuffle(seeds)
        for s in seeds[:ensemble_size // 2]:
            t = self.nearest_neighbor(s)
            t = self.two_opt(t, max_iter=50)
            ensemble.append(t)
        # some cheapest insertion and random swaps
        while len(ensemble) < ensemble_size:
            t = self.cheapest_insertion()
            # random perturbation
            for _ in range(3):
                i = random.randrange(n)
                j = random.randrange(n)
                if i >= j:
                    continue
                t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
            t = self.two_opt(t, max_iter=40)
            ensemble.append(t)

        # multiscale definitions (number of patches at each scale)
        max_scale = min(16, max(2, int(np.ceil(n / 4))))
        scales = []
        s = 2
        while s <= max_scale:
            scales.append(s)
            s *= 2
        if scales[-1] != max_scale:
            scales.append(max_scale)

        # Consensus recorder: edge -> count
        def compute_consensus(ens):
            counter = defaultdict(int)
            for t in ens:
                edges = self.get_edges_set(t)
                for e in edges:
                    counter[e] += 1
            return counter

        consensus = compute_consensus(ensemble)
        best_tour = min(ensemble, key=self.tour_length)
        best_length = self.tour_length(best_tour)

        # Elasticity controller (maps node -> elasticity score)
        elasticity = np.zeros(n, dtype=float)
        # History adviser: score per scale, last improvement iteration
        scale_scores = {sc: 1.0 for sc in scales}
        last_improve_iter = 0

        iter_no_improve = 0

        for iteration in range(max_iters):
            # time-based termination
            if time.time() - start_time > time_limit:
                break

            # pick scale(s) guided by history
            total_score = sum(scale_scores.values())
            rnd = random.random() * total_score
            acc = 0.0
            chosen_scale = scales[0]
            for sc, sc_score in scale_scores.items():
                acc += sc_score
                if rnd <= acc:
                    chosen_scale = sc
                    break

            # create patches and overlapping elastic patches
            patches = self.make_patches(chosen_scale)
            if len(patches) == 0:
                patches = [list(range(n))]
            # choose a patch at random, and maybe overlap with neighbor
            p_idx = random.randrange(len(patches))
            patch = set(patches[p_idx])
            if random.random() < 0.5 and len(patches) > 1:
                # overlap with adjacent patch if exists
                neighbor_idx = min(len(patches) - 1, p_idx + (1 if p_idx + 1 < len(patches) else -1))
                patch = patch.union(patches[neighbor_idx])

            patch = list(patch)
            if len(patch) <= 3:
                # nothing much to do, continue to global improvements
                pass

            # Cooperative multiscale merges and recombinations among candidates:
            # extract segments for patch from ensemble, recombine, and try to improve
            # Build a pool of segments (subtours) as orderings of patch nodes extracted from ensemble
            segment_pool = []
            for t in ensemble:
                # get order of patch nodes as they appear in tour t
                pos = {t[i]: i for i in range(n)}
                ordered = sorted(patch, key=lambda x: pos[x])
                segment_pool.append(tuple(ordered))

            # Try crossovers: for a subset of ensemble entries, replace patch portion with another segment and repair
            new_candidates = []
            for _ in range(min(len(ensemble), 6)):
                base = random.choice(ensemble)
                donor = list(random.choice(segment_pool))
                # form new candidate by replacing the base's order of patch nodes with donor order
                base_copy = [x for x in base if x not in patch]
                # find insertion point: place donor in place by nearest neighbor heuristic among endpoints
                # simple approach: insert donor between two nodes whose midpoint is closest
                # find best position to insert donor list
                best_insert = None
                best_cost_increase = float('inf')
                for i in range(len(base_copy)):
                    a = base_copy[i]
                    b = base_copy[(i + 1) % len(base_copy)]
                    # cost to insert donor between a and b: a->donor[0] + donor[-1]->b - a->b
                    cost = (self.distance_matrix[a, donor[0]] + self.distance_matrix[donor[-1], b] - self.distance_matrix[a, b])
                    if cost < best_cost_increase:
                        best_cost_increase = cost
                        best_insert = i + 1
                if best_insert is None:
                    new_t = base
                else:
                    new_t = base_copy[:best_insert] + donor + base_copy[best_insert:]
                new_t = self.two_opt(new_t, max_iter=30)
                new_candidates.append(new_t)

            # Add new candidates to ensemble and keep best ensemble_size
            ensemble.extend(new_candidates)
            ensemble = sorted(ensemble, key=self.tour_length)[:ensemble_size]

            # Extract consensus backbone from recurring high-quality connections
            consensus = compute_consensus(ensemble)
            # threshold for backbone
            threshold = max(2, int(0.5 * len(ensemble)))
            backbone_edges = {e for e, cnt in consensus.items() if cnt >= threshold}
            # designate low-confidence regions as elastic zones
            elastic_nodes = set()
            for e, cnt in consensus.items():
                if cnt < max(1, 0.25 * len(ensemble)):
                    elastic_nodes.add(e[0])
                    elastic_nodes.add(e[1])
                    elasticity[e[0]] += 0.1
                    elasticity[e[1]] += 0.1

            # Assemble full-route proposals by stitching the consensus backbone with elastic connectors:
            # Strategy: for each ensemble member, perform two_opt but disallow breaking backbone edges
            fixed_edges = set(backbone_edges)
            proposals = []
            for t in ensemble:
                t_copy = list(t)
                t_copy = self.two_opt(t_copy, max_iter=50, fixed_edges=fixed_edges)
                proposals.append(t_copy)
            ensemble = sorted(proposals, key=self.tour_length)[:ensemble_size]

            # Cooperative exchange of promising segments among the ensemble:
            improved_flag = False
            for a_idx in range(len(ensemble)):
                for b_idx in range(a_idx + 1, len(ensemble)):
                    a = list(ensemble[a_idx])
                    b = list(ensemble[b_idx])
                    # choose random segment in a that is elastic-heavy
                    seg_len = max(2, int(0.1 * n))
                    i = random.randrange(0, n)
                    j = (i + seg_len) % n
                    if i < j:
                        seg_a = a[i:j + 1]
                    else:
                        seg_a = a[i:] + a[:j + 1]
                    # attempt to insert seg_a into b in best place
                    b_candidate = [x for x in b if x not in seg_a]
                    # find best insertion place
                    best_place = None
                    best_inc = float('inf')
                    for pos in range(len(b_candidate)):
                        a1 = b_candidate[pos]
                        b1 = b_candidate[(pos + 1) % len(b_candidate)]
                        inc = (self.distance_matrix[a1, seg_a[0]] + self.distance_matrix[seg_a[-1], b1] - self.distance_matrix[a1, b1])
                        if inc < best_inc:
                            best_inc = inc
                            best_place = pos + 1
                    b_new = b_candidate[:best_place] + seg_a + b_candidate[best_place:]
                    b_new = self.two_opt(b_new, max_iter=20, fixed_edges=fixed_edges)
                    if self.tour_length(b_new) + 1e-9 < self.tour_length(b):
                        ensemble[b_idx] = b_new
                        improved_flag = True

            # If progress stalls, generate history-guided multiscale perturbations
            current_best = min(ensemble, key=self.tour_length)
            current_best_len = self.tour_length(current_best)
            if current_best_len + 1e-9 < best_length:
                best_length = current_best_len
                best_tour = current_best
                last_improve_iter = iteration
                iter_no_improve = 0
                # reinforce chosen scale
                scale_scores[chosen_scale] = scale_scores.get(chosen_scale, 1.0) + 1.0
            else:
                iter_no_improve += 1
                # decay scale score slightly
                scale_scores[chosen_scale] = max(0.1, scale_scores.get(chosen_scale, 1.0) * 0.995)

            # If too many iterations without improvement -> perturb
            if iter_no_improve >= stagnation_limit:
                iter_no_improve = 0
                # perform perturbations: random reversals, relax/tighten consensus
                for idx in range(len(ensemble)):
                    t = list(ensemble[idx])
                    a = random.randrange(n)
                    b = random.randrange(n)
                    if a != b:
                        i, j = min(a, b), max(a, b)
                        t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                        t = self.two_opt(t, max_iter=40)
                        ensemble[idx] = t
                # relax consensus slightly
                for k in scale_scores:
                    scale_scores[k] = max(0.2, scale_scores[k] * 0.9)
                # random reseed some candidates
                for _ in range(2):
                    t = list(self.nearest_neighbor(random.randrange(n)))
                    t = self.two_opt(t, max_iter=50)
                    ensemble.append(t)
                ensemble = sorted(ensemble, key=self.tour_length)[:ensemble_size]

        # Post-processing: cross-scale polishing pass (global 2-opt with backbone protection sometimes)
        # Recompute consensus and possibly do one final heavy 2-opt on best tour
        final_fixed = set(e for e, cnt in compute_consensus(ensemble).items() if cnt >= max(1, int(0.6 * len(ensemble))))
        polished = list(best_tour)
        polished = self.two_opt(polished, max_iter=500, fixed_edges=final_fixed)
        polished = self.two_opt(polished, max_iter=500)  # allow breaking if beneficial
        best_tour = polished
        best_length = self.tour_length(best_tour)

        return np.asarray(best_tour, dtype=int)

