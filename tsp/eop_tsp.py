# our updated program here
import numpy as np
import math
import random
import time
from typing import List, Tuple

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
        self.n = len(coordinates)
        if self.distance_matrix.shape != (self.n, self.n):
            raise ValueError("distance_matrix must be shape (n,n)")
        # RNG
        self.rng = np.random.default_rng()
        # Topology guide: adjacency frequency matrix
        self.adj_freq = np.zeros((self.n, self.n), dtype=float)
        # Fragment repository: list of dicts {'nodes': tuple, 'left': int, 'right': int, 'score': float}
        self.fragment_repo = []
        # Parameters (tunable)
        self.max_frag_repo = max(2000, 10 * self.n)
        self.seed_pool_size = min(2 * self.n, 200)
        self.best_pool_size = min(50, self.seed_pool_size)
        self.start_time = None

    # --- helper methods ---

    def tour_length(self, tour: np.ndarray) -> float:
        # tour is numpy array of ints
        n = len(tour)
        if n == 0:
            return 0.0
        nxt = np.roll(tour, -1)
        return float(self.distance_matrix[tour, nxt].sum())

    def nearest_neighbor(self, start: int) -> np.ndarray:
        n = self.n
        visited = np.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True
        current = start
        for _ in range(n - 1):
            # choose nearest unvisited
            dist_row = self.distance_matrix[current]
            # mask visited
            dist_row_masked = np.where(visited, np.inf, dist_row)
            nxt = int(np.argmin(dist_row_masked))
            tour.append(nxt)
            visited[nxt] = True
            current = nxt
        return np.array(tour, dtype=int)

    def two_opt(self, tour: np.ndarray, max_iters: int = 1000) -> np.ndarray:
        """
        2-opt using nearest-neighbor candidate lists and best-improvement among those candidates.
        Keeps the same signature and return type as the original function but typically inspects
        only a small set of promising j-candidates per i (accelerates large instances).
        """
        n = len(tour)
        if n <= 3:
            return tour.copy()
        tour = tour.copy().astype(int)
        dist = self.distance_matrix
        it = 0
        improved = True
        tol = 1e-12

        # Build nearest-neighbor candidate lists: for each node, keep up to k nearest neighbors
        # (exclude the node itself). Using argsort here is simple and effective for moderate n.
        k = min(20, n - 1)  # tuneable: number of neighbors to consider
        # argsort full matrix row-wise; take first k nearest excluding self
        # For numerical ties or symmetric matrices this is fine.
        nn_idx = np.argsort(dist, axis=1)[:, 1 : (k + 1)]  # shape (n, k)

        # Main loop: attempt improvements up to max_iters
        while improved and it < max_iters:
            it += 1
            improved = False
            # position mapping: node -> index in current tour
            pos = np.empty(n, dtype=int)
            pos[tour] = np.arange(n, dtype=int)

            # randomize the order of i's to avoid pathological patterns
            order = np.arange(0, n - 1)  # we don't consider i == n-1 because edge (n-1,0) handled via wrap
            np.random.shuffle(order)

            for i in order:
                a = tour[i]
                b = tour[i + 1]
                # candidate neighbor nodes for 'a'
                candidates = nn_idx[a]  # node labels
                # map candidate nodes to their positions in the tour
                j_pos = pos[candidates]

                # We only consider j > i to avoid duplicate swap checks (the other direction will be checked when i is smaller)
                mask = j_pos > i
                # Exclude trivial (adjacent) swaps which do nothing: j == i or j == i+1
                mask &= (j_pos != i + 1)
                # Exclude (i==0, j==n-1) as that reconnects same edge
                if i == 0:
                    mask &= (j_pos != n - 1)
                if not np.any(mask):
                    continue

                filtered_candidates = candidates[mask]
                filtered_j = j_pos[mask]

                # nodes c and d for the delta formula
                c_nodes = filtered_candidates
                d_nodes = tour[(filtered_j + 1) % n]

                # vectorized delta computation for all filtered candidates
                # delta = dist[a,c] + dist[b,d] - dist[a,b] - dist[c,d]
                # Use fancy indexing to compute arrays efficiently
                delta = dist[a, c_nodes] + dist[b, d_nodes] - dist[a, b] - dist[c_nodes, d_nodes]

                # find best (most negative) delta among candidates
                min_idx = np.argmin(delta)
                min_delta = float(delta[min_idx])
                if min_delta < -tol:
                    j = int(filtered_j[min_idx])
                    # perform 2-opt: reverse segment (i+1 .. j)
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    improved = True
                    break  # restart from updated tour (first/best improvement per outer iteration)

            # loop continues if improved
        return tour

    def ordered_crossover(self, p1: np.ndarray, p2: np.ndarray, seg_len: int = None) -> np.ndarray:
        # OX crossover: take a segment from p1 and fill with order of p2
        n = len(p1)
        if seg_len is None:
            seg_len = max(1, n // 6)
        i = self.rng.integers(0, n)
        j = (i + seg_len) % n
        if i < j:
            seg = p1[i:j + 1]
            seg_set = set(seg.tolist())
            child = []
            # add segment in position
            for k in range(i):
                child.append(None)
            child.extend(seg.tolist())
            for k in range(j + 1, n):
                child.append(None)
            # fill from p2 in order
            pos = 0
            for x in p2:
                if x in seg_set:
                    continue
                # find next None position
                while pos < n and child[pos] is not None:
                    pos += 1
                if pos < n:
                    child[pos] = int(x)
                    pos += 1
            # safety assert
            if any(c is None for c in child):
                # fallback to simple fill
                remaining = [x for x in p2 if x not in seg_set]
                child = remaining[:i] + seg.tolist() + remaining[i:]
            return np.array(child, dtype=int)
        else:
            # wrap-around case: take seg across end
            seg = np.concatenate((p1[i:], p1[:j + 1]))
            seg_set = set(seg.tolist())
            child = [None] * n
            # place segment starting at i
            pos = i
            for val in seg:
                child[pos] = int(val)
                pos = (pos + 1) % n
            # fill remaining with p2
            pos = 0
            for x in p2:
                if x in seg_set:
                    continue
                while child[pos] is not None:
                    pos += 1
                child[pos] = int(x)
            return np.array(child, dtype=int)

    def random_segment_crossover(self, p1: np.ndarray, p2: np.ndarray, num_segments: int = 2) -> np.ndarray:
        # take multiple disjoint segments from p1 and place them into p2's order
        n = len(p1)
        seg_positions = []
        lengths = []
        taken = np.zeros(n, dtype=bool)
        for _ in range(num_segments):
            l = self.rng.integers(1, max(2, n // 8) + 1)
            start = self.rng.integers(0, n)
            seg = [(start + k) % n for k in range(l)]
            # if too much overlap, try another
            if taken[seg].any():
                continue
            seg_positions.append((start, l))
            lengths.append(l)
            taken[seg] = True
        # collect nodes for segments in order of appearance in p1
        segments_nodes = []
        for start, l in seg_positions:
            nodes = [int(p1[(start + k) % n]) for k in range(l)]
            segments_nodes.append(nodes)
        # Build child by starting from p2 order and inserting segments in their original relative order
        child = [x for x in p2 if all(x not in seg for seg in segments_nodes)]
        # Now decide insertion positions: insert each segment at position nearest to its first node in child (by adjacency)
        for seg in segments_nodes:
            insert_idx = 0
            # find best insertion by matching left adjacency frequency
            best_score = -1.0
            for idx in range(len(child) + 1):
                left = child[idx - 1] if idx > 0 else child[-1]
                right = child[idx] if idx < len(child) else child[0]
                # score connection left->seg[0] + seg[-1]->right using adj_freq
                score = self.adj_freq[left, seg[0]] + self.adj_freq[seg[-1], right]
                if score > best_score:
                    best_score = score
                    insert_idx = idx
            child[insert_idx:insert_idx] = seg
        return np.array(child, dtype=int)

    def extract_fragments_from_tour(self, tour: np.ndarray, min_len: int = 2, max_len: int = None, sample: int = 10):
        n = len(tour)
        if max_len is None:
            max_len = max(2, n // 6)
        frags = []
        # sample random segments
        for _ in range(sample):
            L = int(self.rng.integers(min_len, max_len + 1))
            i = int(self.rng.integers(0, n))
            nodes = tuple(int(tour[(i + k) % n]) for k in range(L))
            left = int(tour[(i - 1) % n])
            right = int(tour[(i + L) % n])
            frags.append((nodes, left, right))
        return frags

    def fragment_score(self, nodes: Tuple[int, ...], left: int, right: int) -> float:
        # score based on adjacency frequencies internal + interface
        score = 0.0
        af = self.adj_freq
        # internal edges
        for a, b in zip(nodes, nodes[1:]):
            score += af[a, b]
        # interface edges
        score += af[left, nodes[0]] + af[nodes[-1], right]
        # normalize by length
        return score / max(1, len(nodes))

    def update_adj_freq_from_tour(self, tour: np.ndarray, weight: float = 1.0):
        n = len(tour)
        for a, b in zip(tour, np.roll(tour, -1)):
            self.adj_freq[int(a), int(b)] += weight

    def rebuild_fragment_repo(self, seed_tours: List[np.ndarray]):
        # clear and repopulate from given tours
        self.fragment_repo = []
        for tour in seed_tours:
            frags = self.extract_fragments_from_tour(tour, min_len=2, max_len=max(2, self.n // 6), sample= min(40, 6*self.n//10))
            for nodes, left, right in frags:
                score = self.fragment_score(nodes, left, right)
                self.fragment_repo.append({'nodes': nodes, 'left': left, 'right': right, 'score': score})
        # keep top fragments
        self.fragment_repo.sort(key=lambda x: -x['score'])
        if len(self.fragment_repo) > self.max_frag_repo:
            self.fragment_repo = self.fragment_repo[:self.max_frag_repo]

    def sample_fragments(self, k: int = 5):
        # sample top and random fragments
        if not self.fragment_repo:
            return []
        top_k = max(1, min(len(self.fragment_repo), k//2))
        sampled = []
        sampled.extend(self.fragment_repo[:top_k])
        if len(self.fragment_repo) > top_k:
            rand_count = k - top_k
            idxs = self.rng.choice(len(self.fragment_repo), size=rand_count, replace=False)
            sampled.extend([self.fragment_repo[i] for i in idxs])
        return sampled

    def macro_restructure(self, tour: np.ndarray, intensity: float = 0.2) -> np.ndarray:
        # perform a macro change: scramble a segment and reinsert
        n = len(tour)
        L = max(2, int(n * intensity))
        i = int(self.rng.integers(0, n))
        seg = [int(tour[(i + k) % n]) for k in range(L)]
        remainder = [int(x) for x in tour if x not in seg]
        # shuffle seg
        self.rng.shuffle(seg)
        # choose insertion point
        pos = int(self.rng.integers(0, len(remainder) + 1))
        new = remainder[:pos] + seg + remainder[pos:]
        return np.array(new, dtype=int)

    def intensify_local(self, tour: np.ndarray, rounds: int = 5) -> np.ndarray:
        best = tour.copy()
        best_len = self.tour_length(best)
        for _ in range(rounds):
            cand = self.two_opt(best.copy(), max_iters=2000)
            cand_len = self.tour_length(cand)
            if cand_len < best_len - 1e-12:
                best = cand
                best_len = cand_len
            # try small macro restructure and local improve
            cand2 = self.macro_restructure(best, intensity=0.05)
            cand2 = self.two_opt(cand2, max_iters=500)
            cand2_len = self.tour_length(cand2)
            if cand2_len < best_len - 1e-12:
                best = cand2
                best_len = cand2_len
        return best

    def compute_initial_seeds(self) -> List[np.ndarray]:
        seeds = []
        n = self.n
        # some deterministic greedy seeds
        starts = list(range(min(n, 10)))
        for s in starts:
            tour = self.nearest_neighbor(s)
            tour = self.two_opt(tour, max_iters=1000)
            seeds.append(tour)
        # random nearest neighbor starts and random permutations
        more = max(0, self.seed_pool_size - len(seeds))
        for _ in range(more):
            if self.rng.random() < 0.6:
                start = int(self.rng.integers(0, n))
                tour = self.nearest_neighbor(start)
            else:
                tour = np.array(self.rng.permutation(n), dtype=int)
            tour = self.two_opt(tour, max_iters=500)
            seeds.append(tour)
        # unique by tuple
        unique = {}
        for t in seeds:
            key = tuple(int(x) for x in t)
            unique[key] = t
        seeds = list(unique.values())
        # sort by length
        seeds.sort(key=self.tour_length)
        return seeds

    def choose_parents(self, pool: List[np.ndarray], best_pool_fraction: float = 0.6):
        # pick two parents mixing best and random
        k_best = max(1, int(len(pool) * best_pool_fraction))
        parents = []
        # choose one from top k_best and one random
        p1 = pool[int(self.rng.integers(0, k_best))]
        p2 = pool[int(self.rng.integers(0, len(pool)))]
        return p1, p2

    # --- main solve method ---

    def solve(self) -> np.ndarray:
        n = self.n
        if n == 0:
            return np.array([], dtype=int)
        self.start_time = time.time()

        # Initial seeds
        seed_tours = self.compute_initial_seeds()
        # initialize adjacency guide from seeds
        self.adj_freq.fill(0.0)
        for t in seed_tours:
            self.update_adj_freq_from_tour(t, weight=1.0)
        # Build fragment repo
        self.rebuild_fragment_repo(seed_tours)

        # candidate pool: maintain a list of good tours (sorted by length)
        pool = seed_tours.copy()
        pool.sort(key=self.tour_length)
        if len(pool) == 0:
            pool = [np.array(self.rng.permutation(n), dtype=int)]

        best = pool[0].copy()
        best_cost = self.tour_length(best)
        current = best.copy()
        current_cost = best_cost

        # annealing parameters
        avg_edge = self.distance_matrix.mean()
        T0 = avg_edge * max(1.0, n * 0.02)
        T = T0
        cooling = 0.995

        # iterations
        max_iter = min(10000, 1000 + 10 * n)
        stagnation = 0
        last_improvement_iter = 0

        for it in range(max_iter):
            # termination check by time (optional): avoid very long runs
            if time.time() - self.start_time > 30.0 + 0.02 * n:
                break

            # select parents
            p1, p2 = self.choose_parents(pool, best_pool_fraction=0.6)
            # pick resolution/fragment sizes
            if self.rng.random() < 0.6:
                seg_len = int(max(1, min(n // 4, max(2, self.rng.integers(2, max(3, n // 8) + 1)))))
                child = self.ordered_crossover(p1, p2, seg_len=seg_len)
            else:
                num_segs = 1 + int(self.rng.integers(0, max(1, n // 50)))
                child = self.random_segment_crossover(p1, p2, num_segments=num_segs)

            # occasionally apply macro restructure
            if self.rng.random() < 0.08:
                child = self.macro_restructure(child, intensity=0.08)

            # local refinement (multi-scale): try small and larger 2-opt
            child = self.two_opt(child, max_iters=500)
            if self.rng.random() < 0.2:
                child = self.two_opt(child, max_iters=2000)

            child_cost = self.tour_length(child)

            # acceptance via fusion-annealing
            delta = child_cost - current_cost
            accept = False
            if child_cost < current_cost - 1e-12:
                accept = True
            else:
                if self.rng.random() < math.exp(-max(0.0, delta) / max(1e-9, T)):
                    accept = True

            if accept:
                current = child
                current_cost = child_cost

            # update global best
            if child_cost < best_cost - 1e-12:
                best = child.copy()
                best_cost = child_cost
                last_improvement_iter = it
                stagnation = 0
            else:
                stagnation += 1

            # update topology guide and fragment repo with accepted child (or sometimes with child regardless)
            weight = 1.0 if accept else 0.5
            self.update_adj_freq_from_tour(child, weight=weight)
            # add fragments
            frags = self.extract_fragments_from_tour(child, min_len=2, max_len=max(2, n // 6), sample=8)
            for nodes, left, right in frags:
                score = self.fragment_score(nodes, left, right)
                self.fragment_repo.append({'nodes': nodes, 'left': left, 'right': right, 'score': score})
            # prune repository
            if len(self.fragment_repo) > 2 * self.max_frag_repo:
                self.fragment_repo.sort(key=lambda x: -x['score'])
                self.fragment_repo = self.fragment_repo[:self.max_frag_repo]

            # maintain pool: occasionally insert child into pool
            if child_cost < self.tour_length(pool[-1]) or self.rng.random() < 0.05:
                pool.append(child.copy())
                pool.sort(key=self.tour_length)
                # cap pool
                if len(pool) > self.seed_pool_size:
                    pool = pool[:self.seed_pool_size]

            # occasional strategic re-seeding to escape stagnation
            if stagnation > max(50, n // 2) and self.rng.random() < 0.5:
                # generate a new random seed improved by 2-opt and insert
                new_seed = np.array(self.rng.permutation(n), dtype=int)
                new_seed = self.two_opt(new_seed, max_iters=800)
                pool.append(new_seed)
                pool.sort(key=self.tour_length)
                if len(pool) > self.seed_pool_size:
                    pool = pool[:self.seed_pool_size]
                # rebuild fragment repo from pool occasionally
                if self.rng.random() < 0.3:
                    self.rebuild_fragment_repo(pool[:max(5, len(pool)//4)])
                stagnation = 0

            # cooling
            T *= cooling
            if T < 1e-6:
                T = T0

        # Post-processing intensification around best structures
        best = self.intensify_local(best, rounds=8)
        best = self.two_opt(best, max_iters=5000)
        # final sanity: ensure permutation of all nodes
        if set(best.tolist()) != set(range(n)):
            # fallback: produce best-known ordering from pool or random
            if pool:
                cand = pool[0]
                if set(cand.tolist()) == set(range(n)):
                    best = cand.copy()
                else:
                    best = np.array(self.rng.permutation(n), dtype=int)
            else:
                best = np.array(self.rng.permutation(n), dtype=int)

        return np.array(best, dtype=int)

