'''
{
        "algorithm": "ALGORITHM Consensus-driven Critic Ensemble (CCE)\n\n    /* PURPOSE: A task-agnostic cooperative metaheuristic that solves routing and scheduling problems by iteratively building consensus sketches from a diverse ensemble, applying a pool of focused critics to refine hybrid proposals, and projecting candidates back to feasibility. */\n    \n    INPUT: abstract problem instance (entities, relations, costs/times, capacity/processing constraints where applicable), objective evaluator, feasibility projector, termination control\n    OUTPUT: a feasible solution representation (permutation or set of routes/schedule) and its objective value\n    \n    /* Initialization Phase */\n    Generate a diverse ensemble of feasible candidate solutions using multiple lightweight constructive strategies\n    Instantiate a diverse set of critics (local improvement operators and feasibility restorers) and an archive of high-quality fragments\n    Configure a consensus engine that can extract recurring structural features from the ensemble and an acceptance rule balancing improvement and diversity\n    \n    /* Main Processing Loop */\n    WHILE termination criteria not satisfied DO\n        Extract a consensus sketch representing common structural features across the ensemble (a soft template, not a full solution)\n        Create hybrid proposals by combining ensemble members guided by the consensus sketch and by sampling complementary fragments from the archive\n        For each hybrid proposal, invoke a committee of critics to generate focused variant edits and to suggest feasibility projections where constraints are violated\n        Evaluate edited proposals with the objective; accept proposals into the ensemble according to the acceptance rule that rewards objective gain and structural novelty\n        Update critic influence by reinforcing critics that produced accepted improvements and attenuating ineffective ones; refresh the archive with newly discovered high-value fragments\n        Occasionally trigger a restructuring step that replaces low-performing ensemble members with recombinations seeded from the archive and the current consensus to escape entrenched patterns\n    END WHILE\n    \n    /* Post-Processing Phase */\n    Apply a concentrated critic cascade to the best ensemble candidate for final polishing and a feasibility pass\n    RETURN the polished feasible solution and its objective value",
        "task_scores": [
            -0.0008615034904700632,
            -0.05343107200318503,
            -0.1345133896671264
        ],
        "task_programs": [
            "# our updated program here\nimport numpy as np\nimport time\nfrom typing import List, Tuple, Callable, Dict\n\nclass TSPSolver:\n    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, rng_seed: int = 0):\n        \"\"\"\n        Initialize the TSP solver.\n\n        Args:\n            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.\n            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.\n            rng_seed: optional random seed for reproducibility.\n        \"\"\"\n        self.coordinates = coordinates\n        self.distance_matrix = distance_matrix\n        self.n = len(coordinates)\n        self.rng = np.random.RandomState(rng_seed)\n\n    # ---------------- Utility functions ----------------\n    def tour_length(self, tour: np.ndarray) -> float:\n        idx = tour\n        d = self.distance_matrix\n        return float(np.sum(d[idx, np.roll(idx, -1)]))\n\n    def _apply_2opt_once(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:\n        \"\"\"\n        Vectorized best-improvement single 2-opt pass with a memory-aware fallback.\n\n        Tries to find the single best 2-opt move (i, j) that reduces tour length.\n        For moderate-sized problems it builds row-wise gains in a vectorized manner.\n        For very large n it falls back to a memory-friendly per-i loop that still\n        uses NumPy operations for each row.\n\n        Returns (new_tour, new_length). If no improving move exists, returns the\n        original tour and its length.\n        \"\"\"\n        n = int(self.n)\n        dmat = self.distance_matrix\n        idx = tour\n\n        # current tour length (ensure float)\n        cur_len = float(np.sum(dmat[idx, np.roll(idx, -1)]))\n\n        # nothing to do for tiny tours\n        if n < 4:\n            return tour, cur_len\n\n        # i ranges 0 .. n-3 (we consider breaking edge (i,i+1) and reconnecting to (j,j+1))\n        i_vals = np.arange(0, n - 2, dtype=int)\n        m = i_vals.size\n        a = idx[i_vals]           # node at position i\n        b = idx[i_vals + 1]       # node at position i+1\n        dab = dmat[a, b]          # distance for edge (a,b) per i\n\n        # memory threshold (number of matrix cells). Tweak if needed.\n        MAX_CELLS = 8_000_000\n\n        # small tolerance for \"no improvement\" (to account for floating rounding)\n        NO_IMPROV_TOL = -1e-12\n\n        if m * n <= MAX_CELLS:\n            # Fully vectorized path (efficient for moderate n)\n            # Build grids using broadcasting where possible to avoid explicit repeats\n            j_range = np.arange(n, dtype=int)\n            # i_grid shape (m,1), j_grid shape (1,n) broadcastable to (m,n)\n            i_grid = i_vals[:, None]\n            j_grid = j_range[None, :]\n\n            # j is valid when j >= i+2 and j < n - (i==0)\n            j_max = (n - (i_vals == 0))[:, None]  # shape (m,1)\n            valid_mask = (j_grid >= (i_grid + 2)) & (j_grid < j_max)\n\n            # corresponding tour nodes c = tour[j], d = tour[j+1]\n            # broadcasting idx over rows\n            c_nodes = idx[j_grid]                 # shape (m, n)\n            d_nodes = idx[(j_grid + 1) % n]       # shape (m, n)\n\n            # compute distance components (each yields shape (m, n))\n            # Use advanced indexing with broadcasting: a[:,None] has shape (m,1)\n            d_a_c = dmat[a[:, None], c_nodes]\n            d_b_d = dmat[b[:, None], d_nodes]\n            d_c_d = dmat[c_nodes, d_nodes]\n\n            # gains = (a->c + b->d) - (a->b + c->d)\n            gains = d_a_c + d_b_d - (dab[:, None] + d_c_d)\n\n            # invalidate infeasible positions\n            gains[~valid_mask] = np.inf\n\n            # find global best gain\n            # use unravel_index for clarity\n            flat_min_idx = int(np.argmin(gains))\n            best_i_row, best_j_col = np.unravel_index(flat_min_idx, gains.shape)\n            min_gain = float(gains[best_i_row, best_j_col])\n\n            # no improving move found\n            if min_gain >= NO_IMPROV_TOL:\n                return tour, cur_len\n\n            # Map to actual i, j positions in the tour\n            best_i = int(i_vals[best_i_row])\n            best_j = int(best_j_col)\n        else:\n            # Memory-conserving row-wise path: iterate over i but compute arrays per-row with NumPy\n            best_gain = 0.0\n            best_i = -1\n            best_j = -1\n\n            # Pre-extract idx for speed\n            for ii in range(m):\n                i = int(i_vals[ii])\n                j_start = i + 2\n                j_end = n - (1 if i == 0 else 0)  # exclusive upper bound\n                if j_start >= j_end:\n                    continue\n\n                js = np.arange(j_start, j_end, dtype=int)\n                c = idx[js]\n                d = idx[(js + 1) % n]\n\n                # vectorized computation for this row\n                da_c = dmat[a[ii], c]\n                db_d = dmat[b[ii], d]\n                dc_d = dmat[c, d]\n                gains_row = da_c + db_d - (dab[ii] + dc_d)\n\n                # find best in this row\n                row_min_pos = int(np.argmin(gains_row))\n                row_min_gain = float(gains_row[row_min_pos])\n\n                if row_min_gain < best_gain:\n                    best_gain = row_min_gain\n                    best_i = i  # actual i value (not row index)\n                    best_j = int(js[row_min_pos])\n\n            if best_i == -1:\n                return tour, cur_len\n\n            min_gain = float(best_gain)\n\n        # Apply 2-opt: best_i is actual tour index i, best_j is actual tour index j\n        i = int(best_i)\n        j = int(best_j)\n\n        # Defensive check (shouldn't happen because of masks)\n        if not (0 <= i < n and 0 <= j < n and (i + 1) <= j):\n            return tour, cur_len\n\n        # Construct new tour by reversing segment between i+1 .. j (inclusive)\n        new = np.empty_like(idx)\n        if i + 1 <= j:\n            # copy prefix up to i\n            if i + 1 > 0:\n                new[: i + 1] = idx[: i + 1]\n            # reversed middle segment\n            new[i + 1 : j + 1] = idx[i + 1 : j + 1][::-1]\n            # suffix after j\n            if j + 1 < n:\n                new[j + 1 :] = idx[j + 1 :]\n        else:\n            # defensive fallback\n            return tour, cur_len\n\n        new_len = cur_len + min_gain\n        return new, float(new_len)\n\n    def two_opt_local_search(self, tour: np.ndarray, max_iters: int = 100) -> Tuple[np.ndarray, float]:\n        \"\"\"\n        Apply repeated best 2-opt improvements until no improvement or max_iters reached.\n        \"\"\"\n        current = tour.copy()\n        cur_len = self.tour_length(current)\n        it = 0\n        while it < max_iters:\n            new, new_len = self._apply_2opt_once(current)\n            if new_len + 1e-12 < cur_len:\n                current = new\n                cur_len = new_len\n                it += 1\n            else:\n                break\n        return current, cur_len\n\n    def swap_operator(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:\n        \"\"\"Random swap of two nodes followed by 2-opt local quick-improve.\"\"\"\n        n = self.n\n        i, j = self.rng.choice(n, size=2, replace=False)\n        new = tour.copy()\n        new[i], new[j] = new[j], new[i]\n        new, new_len = self.two_opt_local_search(new, max_iters=5)\n        return new, new_len\n\n    def reinsertion_operator(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:\n        \"\"\"Remove a block of size 1-3 and insert at random position (or-opt).\"\"\"\n        n = self.n\n        k = self.rng.randint(1, min(4, n))\n        i = self.rng.randint(0, n)\n        block = []\n        for x in range(k):\n            block.append(tour[(i + x) % n])\n        block = np.array(block, dtype=int)\n        remaining = []\n        for t in tour:\n            if not np.any(t == block):\n                remaining.append(t)\n        remaining = np.array(remaining, dtype=int)\n        insert_pos = self.rng.randint(0, len(remaining) + 1)\n        new = np.concatenate([remaining[:insert_pos], block, remaining[insert_pos:]])\n        new_len = self.tour_length(new)\n        new, new_len = self.two_opt_local_search(new, max_iters=5)\n        return new, new_len\n\n    def double_bridge(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:\n        \"\"\"Perturbation via double-bridge move to escape local minima.\"\"\"\n        n = self.n\n        if n < 8:\n            return self.swap_operator(tour)\n        # select four cut points\n        pts = sorted(self.rng.choice(range(1, n), size=4, replace=False))\n        a, b, c, d = pts\n        p = tour\n        new = np.concatenate([p[:a], p[c:d], p[b:c], p[a:b], p[d:]])\n        new_len = self.tour_length(new)\n        return new, new_len\n\n    # ---------------- Ensemble and archive initialization ----------------\n    def _nearest_neighbor_tour(self, start: int = 0) -> np.ndarray:\n        n = self.n\n        d = self.distance_matrix\n        unvisited = set(range(n))\n        curr = start\n        tour = [curr]\n        unvisited.remove(curr)\n        while unvisited:\n            # choose nearest unvisited\n            rem = np.array(list(unvisited))\n            next_idx = rem[np.argmin(d[curr, rem])]\n            tour.append(int(next_idx))\n            unvisited.remove(int(next_idx))\n            curr = int(next_idx)\n        return np.array(tour, dtype=int)\n\n    def init_ensemble(self, ensemble_size: int) -> List[Dict]:\n        \"\"\"\n        Generate a diverse ensemble of feasible solutions.\n        Each ensemble member is a dict: {'tour': np.ndarray, 'cost': float, 'age': int, 'score': float}\n        \"\"\"\n        ensemble = []\n        n = self.n\n        # seeds: nearest neighbor from several starts, greedy heuristics, random permutations\n        starts = list(range(min(n, 10)))\n        for s in starts:\n            tour = self._nearest_neighbor_tour(start=s)\n            tour, cost = self.two_opt_local_search(tour, max_iters=20)\n            ensemble.append({'tour': tour, 'cost': cost, 'age': 0, 'score': -cost})\n        # add random perturbed tours\n        while len(ensemble) < ensemble_size:\n            tour = np.arange(n)\n            self.rng.shuffle(tour)\n            tour, cost = self.two_opt_local_search(tour, max_iters=10)\n            ensemble.append({'tour': tour, 'cost': cost, 'age': 0, 'score': -cost})\n        # sort by cost\n        ensemble.sort(key=lambda x: x['cost'])\n        return ensemble\n\n    # ---------------- Consensus sketch and hybrid construction ----------------\n    def consensus_sketch(self, ensemble: List[Dict]) -> np.ndarray:\n        \"\"\"\n        Build an edge frequency matrix (soft template).\n        Returns a matrix P of shape (n, n) with values proportional to frequency of directed edges.\n        \"\"\"\n        n = self.n\n        P = np.zeros((n, n), dtype=float)\n        for member in ensemble:\n            tour = member['tour']\n            for i in range(n):\n                a = tour[i]\n                b = tour[(i + 1) % n]\n                P[a, b] += 1.0\n        # normalize\n        total = np.sum(P)\n        if total > 0:\n            P /= total\n        # smooth to avoid zeros\n        P += 1e-6\n        return P\n\n    def build_hybrid_from_sketch(self, P: np.ndarray, archive: List[Tuple[np.ndarray, float]], blend_prob: float = 0.6) -> np.ndarray:\n        \"\"\"\n        Construct a hybrid tour guided by consensus P and optionally seeded with fragments from archive.\n        \"\"\"\n        n = self.n\n        # attempt to place high-quality archived fragments first with some probability\n        used = np.zeros(n, dtype=bool)\n        tour = []\n        if archive and self.rng.rand() < 0.5:\n            # pick a fragment proportional to fragment quality\n            frags, scores = zip(*archive)\n            scores = np.array(scores)\n            probs = (np.max(scores) - scores + 1e-6)\n            probs /= probs.sum()\n            idx = self.rng.choice(len(frags), p=probs)\n            frag = frags[idx]\n            # insert fragment into tour\n            for node in frag:\n                if not used[node]:\n                    tour.append(node)\n                    used[node] = True\n        # pick random start among unvisited\n        if not tour:\n            start = self.rng.randint(0, n)\n            tour.append(start)\n            used[start] = True\n        # greedy build using P and distance heuristics\n        while len(tour) < n:\n            curr = tour[-1]\n            candidates = np.where(~used)[0]\n            # score combining P (consensus) and inverse distance\n            pvals = P[curr, candidates]\n            dvals = self.distance_matrix[curr, candidates]\n            invd = 1.0 / (dvals + 1e-6)\n            score = pvals * 2.0 + invd  # weight consensus more\n            # add a small randomization\n            score = score ** (1.0) + 1e-4 * self.rng.rand(len(score))\n            # sample proportional to score to maintain diversity\n            probs = score / score.sum()\n            chosen = self.rng.choice(candidates, p=probs)\n            tour.append(int(chosen))\n            used[int(chosen)] = True\n        return np.array(tour, dtype=int)\n\n    # ---------------- Archive management ----------------\n    def update_archive(self, archive: List[Tuple[np.ndarray, float]], fragment: np.ndarray, cost: float, max_size: int = 50):\n        \"\"\"\n        Maintain archive of high-quality fragments (subsequences).\n        Store fragments as tuples (array, cost).\n        \"\"\"\n        # store shorter fragments to encourage recombination\n        frag_len = len(fragment)\n        if frag_len < 2:\n            return\n        # keep unique by tuple\n        key = tuple(fragment.tolist())\n        for f, c in archive:\n            if tuple(f.tolist()) == key:\n                # update if better\n                return\n        archive.append((fragment.copy(), cost))\n        # trim archive by best (lower cost considered better)\n        archive.sort(key=lambda x: x[1])\n        if len(archive) > max_size:\n            del archive[max_size:]\n\n    # ---------------- Critics committee ----------------\n    def get_critics(self) -> List[Callable[[np.ndarray], Tuple[np.ndarray, float]]]:\n        \"\"\"\n        Return the set of critic operators.\n        \"\"\"\n        return [\n            self.two_opt_local_search,\n            self.swap_operator,\n            self.reinsertion_operator,\n            self.double_bridge\n        ]\n\n    # ---------------- Main solver ----------------\n    def solve(self,\n              max_iters: int = None,\n              ensemble_size: int = None,\n              time_limit: float = None) -> np.ndarray:\n        \"\"\"\n        Solve the Traveling Salesman Problem (TSP) using a Consensus-driven Critic Ensemble metaheuristic.\n\n        Returns:\n            A numpy array of shape (n,) containing a permutation of integers\n            [0, 1, ..., n-1] representing the order in which the cities are visited.\n        \"\"\"\n        n = self.n\n        if n <= 1:\n            return np.arange(n, dtype=int)\n        # configure parameters heuristically\n        if ensemble_size is None:\n            ensemble_size = min(max(8, n // 2), 40)\n        if max_iters is None:\n            max_iters = 200 + 20 * int(np.sqrt(n))\n        if time_limit is None:\n            time_limit = 30.0  # seconds default cap\n        start_time = time.time()\n\n        # initialize ensemble and archive\n        ensemble = self.init_ensemble(ensemble_size)\n        archive: List[Tuple[np.ndarray, float]] = []\n        # populate archive with fragments from initial ensemble\n        for member in ensemble:\n            tour = member['tour']\n            cost = member['cost']\n            # extract several random fragments\n            for _ in range(3):\n                a = self.rng.randint(0, n)\n                b = self.rng.randint(0, n)\n                if a == b:\n                    continue\n                if a < b:\n                    frag = tour[a:b+1]\n                else:\n                    frag = np.concatenate([tour[a:], tour[:b+1]])\n                self.update_archive(archive, frag, cost)\n\n        critics = self.get_critics()\n        critic_weights = np.ones(len(critics), dtype=float)\n        critic_success = np.zeros(len(critics), dtype=int)\n        # acceptance parameters\n        diversity_bias = 0.3  # weight on novelty\n        restructure_period = max(10, max_iters // 10)\n        iter_count = 0\n\n        # main loop\n        while iter_count < max_iters and (time.time() - start_time) < time_limit:\n            iter_count += 1\n            # age increment\n            for member in ensemble:\n                member['age'] += 1\n\n            # consensus sketch\n            P = self.consensus_sketch(ensemble)\n\n            # create hybrid proposals\n            proposals = []\n            num_hybrids = min(len(ensemble), max(2, ensemble_size // 2))\n            for _ in range(num_hybrids):\n                hybrid = self.build_hybrid_from_sketch(P, archive)\n                proposals.append(hybrid)\n\n            # for each hybrid, apply a committee of critics (sample critics probabilistically)\n            for hybrid in proposals:\n                # initial candidate\n                base_tour = hybrid\n                base_cost = self.tour_length(base_tour)\n                candidate_pool = []\n                # apply committee: pick a small committee of critics by weight\n                probs = critic_weights / critic_weights.sum()\n                committee_indices = self.rng.choice(len(critics), size=min(3, len(critics)), replace=False, p=probs)\n                for ci in committee_indices:\n                    critic = critics[ci]\n                    try:\n                        new_tour, new_cost = critic(base_tour.copy())\n                    except Exception:\n                        continue\n                    candidate_pool.append((ci, new_tour, new_cost))\n                # also consider applying pairwise critic compositions occasionally\n                if self.rng.rand() < 0.2 and len(candidate_pool) >= 2:\n                    # take best two and combine with small perturbation\n                    candidate_pool.sort(key=lambda x: x[2])\n                    _, t1, _ = candidate_pool[0]\n                    _, t2, _ = candidate_pool[1]\n                    # splice t2 fragment into t1\n                    a = self.rng.randint(0, n)\n                    b = self.rng.randint(0, n)\n                    if a != b:\n                        if a < b:\n                            frag = t2[a:b+1]\n                        else:\n                            frag = np.concatenate([t2[a:], t2[:b+1]])\n                        # insert frag into t1\n                        remaining = [x for x in t1 if x not in frag]\n                        insert_pos = self.rng.randint(0, len(remaining)+1)\n                        new = np.array(remaining[:insert_pos] + list(frag) + remaining[insert_pos:], dtype=int)\n                        new_cost = self.tour_length(new)\n                        candidate_pool.append((None, new, new_cost))\n\n                # evaluate and accept according to acceptance rule\n                # acceptance favoring improvement and novelty\n                for ci, cand_tour, cand_cost in candidate_pool:\n                    # measure novelty: average edge overlap with ensemble\n                    # compute fraction of edges shared with best ensemble member\n                    best_member = ensemble[0]['tour']\n                    # edge set for best member\n                    best_edges = set()\n                    for i in range(n):\n                        a = best_member[i]\n                        b = best_member[(i+1) % n]\n                        best_edges.add((a,b))\n                    cand_edges = 0\n                    for i in range(n):\n                        a = cand_tour[i]\n                        b = cand_tour[(i+1) % n]\n                        if (a,b) in best_edges:\n                            cand_edges += 1\n                    overlap = cand_edges / n\n                    novelty = 1.0 - overlap\n                    # compute acceptance score\n                    improvement = ensemble[-1]['cost'] - cand_cost  # relative to worst\n                    # we define objective score\n                    score = 0.7 * (improvement) + diversity_bias * novelty * (ensemble[-1]['cost'] * 0.1)\n                    # accept if score positive or probabilistic based on small improvements\n                    if score > 1e-9 or (self.rng.rand() < 0.01 and cand_cost < ensemble[0]['cost'] * 1.05):\n                        # replace worst or a random high-age member\n                        # find candidate to replace: worst or oldest among worse ones\n                        worst_idx = max(range(len(ensemble)), key=lambda i: ensemble[i]['cost'])\n                        # ensure we do not always replace best\n                        if cand_cost < ensemble[worst_idx]['cost'] or self.rng.rand() < 0.5:\n                            ensemble[worst_idx] = {'tour': cand_tour.copy(), 'cost': cand_cost, 'age': 0, 'score': -cand_cost}\n                            # update archive with fragments from candidate\n                            # extract a few fragments\n                            for _ in range(3):\n                                a = self.rng.randint(0, n)\n                                b = self.rng.randint(0, n)\n                                if a == b:\n                                    continue\n                                if a < b:\n                                    frag = cand_tour[a:b+1]\n                                else:\n                                    frag = np.concatenate([cand_tour[a:], cand_tour[:b+1]])\n                                self.update_archive(archive, frag, cand_cost)\n                            # update critic success\n                            if ci is not None:\n                                critic_success[ci] += 1\n                                critic_weights[ci] = 1.0 + critic_success[ci]\n                            # sort ensemble\n                            ensemble.sort(key=lambda x: x['cost'])\n            # attenuate ineffective critics periodically\n            if iter_count % 10 == 0:\n                # reduce weight of critics that rarely succeed\n                total_success = critic_success.sum() + 1e-6\n                for i in range(len(critic_weights)):\n                    # proportional to recent success\n                    critic_weights[i] = 1.0 + critic_success[i] / (1.0 + total_success)\n                # small random jitter\n                critic_weights += 1e-3 * self.rng.rand(len(critic_weights))\n\n            # occasional restructuring to escape entrenched patterns\n            if iter_count % restructure_period == 0 and iter_count > 0:\n                # replace a few worst members by recombinations seeded from top and archive\n                num_replace = max(1, ensemble_size // 8)\n                for _ in range(num_replace):\n                    worst_idx = max(range(len(ensemble)), key=lambda i: ensemble[i]['cost'])\n                    # build recombination from best and an archive fragment\n                    top = ensemble[0]['tour']\n                    if archive and self.rng.rand() < 0.8:\n                        frag, _ = archive[self.rng.randint(0, len(archive))]\n                        # splice frag into top\n                        remaining = [x for x in top if x not in frag]\n                        insert_pos = self.rng.randint(0, len(remaining)+1)\n                        new = np.array(remaining[:insert_pos] + list(frag) + remaining[insert_pos:], dtype=int)\n                    else:\n                        # blend top and a random ensemble member\n                        partner = ensemble[self.rng.randint(0, len(ensemble))]['tour']\n                        cut = self.rng.randint(1, n-1)\n                        part = partner[:cut]\n                        remaining = [x for x in top if x not in part]\n                        new = np.array(list(part) + remaining, dtype=int)\n                    new, new_cost = self.two_opt_local_search(new, max_iters=10)\n                    ensemble[worst_idx] = {'tour': new, 'cost': new_cost, 'age': 0, 'score': -new_cost}\n                ensemble.sort(key=lambda x: x['cost'])\n\n        # Post-processing: concentrated critic cascade on best candidate\n        best = ensemble[0]\n        best_tour = best['tour']\n        best_cost = best['cost']\n        # apply multiple intensification passes\n        for _ in range(3):\n            best_tour, best_cost = self.two_opt_local_search(best_tour, max_iters=50)\n            # attempt a few reinsertion and swap improvements\n            for _ in range(10):\n                new, new_cost = self.reinsertion_operator(best_tour)\n                if new_cost + 1e-12 < best_cost:\n                    best_tour, best_cost = new, new_cost\n            # occasional double-bridge + local search to try escape shallow local minima\n            new, new_cost = self.double_bridge(best_tour)\n            new, new_cost = self.two_opt_local_search(new, max_iters=50)\n            if new_cost + 1e-12 < best_cost:\n                best_tour, best_cost = new, new_cost\n\n        # final feasibility pass (identity for TSP) and return\n        return best_tour.copy()",
            "# our updated program here\nimport numpy as np\nimport random\nimport time\nfrom copy import deepcopy\nfrom collections import defaultdict, Counter\n\nclass CVRPSolver:\n    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: list, vehicle_capacity: int,\n                 ensemble_size: int = 8, max_iters: int = 500, time_limit: float = 10.0, seed: int = None):\n        \"\"\"\n        Initialize the CVRP solver.\n\n        Args:\n            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each node, including the depot.\n            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between nodes.\n            demands: List of integers representing the demand of each node (first node is typically the depot with zero demand).\n            vehicle_capacity: Integer representing the maximum capacity of each vehicle.\n            ensemble_size: Number of ensemble members to maintain.\n            max_iters: Maximum number of main loop iterations.\n            time_limit: Max seconds to run (overrides max_iters if reached).\n            seed: random seed for reproducibility (optional).\n        \"\"\"\n        self.coordinates = coordinates\n        self.distance_matrix = distance_matrix\n        self.demands = demands\n        self.vehicle_capacity = vehicle_capacity\n\n        self.n = len(coordinates)\n        self.customers = list(range(1, self.n))\n        self.ensemble_size = max(2, ensemble_size)\n        self.max_iters = max_iters\n        self.time_limit = time_limit\n        self.seed = seed\n        if seed is not None:\n            random.seed(seed)\n            np.random.seed(seed)\n\n        # critics: mapping name -> function\n        self.critics = {}\n        self.critic_weights = {}\n        self._init_critics()\n\n        # archive of fragments: map fragment tuple -> (best_score, last_seen_iter)\n        self.archive = {}\n\n    # ---------------- Helper utilities ----------------\n\n    def _routes_from_solution(self, sol):\n        routes = []\n        cur = []\n        for v in sol:\n            if v == 0:\n                if cur:\n                    routes.append([0] + cur + [0])\n                    cur = []\n                else:\n                    # empty separator; ignore consecutive zeros\n                    continue\n            else:\n                cur.append(v)\n        if cur:\n            routes.append([0] + cur + [0])\n        if not routes:\n            routes = [[0,0]]\n        return routes\n\n    def _solution_from_routes(self, routes):\n        flat = []\n        for r in routes:\n            if len(r) == 0:\n                continue\n            if flat and flat[-1] == 0:\n                # avoid double zeros; ensure route starts with 0\n                pass\n            # ensure route starts and ends with depot\n            if r[0] != 0:\n                r = [0] + r\n            if r[-1] != 0:\n                r = r + [0]\n            flat.extend(r if flat == [] else r)  # might add duplicate 0s but acceptable\n        # remove leading duplicate zeros in sequences like [0,0,...]\n        # compress consecutive zeros to single separators\n        compressed = []\n        prev = None\n        for x in flat:\n            if x == 0 and prev == 0:\n                continue\n            compressed.append(x)\n            prev = x\n        if compressed[0] != 0:\n            compressed = [0] + compressed\n        if compressed[-1] != 0:\n            compressed = compressed + [0]\n        return compressed\n\n    def _objective(self, sol):\n        # sum distances along flat solution\n        total = 0.0\n        prev = sol[0]\n        for v in sol[1:]:\n            total += self.distance_matrix[prev, v]\n            prev = v\n        return total\n\n    def _route_demand(self, route):\n        # route is list including depot ends\n        return sum(self.demands[v] for v in route if v != 0)\n\n    def _feasibility_projector(self, sol):\n        # Ensure each route respects capacity by splitting overloaded routes greedily.\n        routes = self._routes_from_solution(sol)\n        new_routes = []\n        for r in routes:\n            if self._route_demand(r) <= self.vehicle_capacity:\n                new_routes.append(r)\n                continue\n            # greedily split r into feasible chunks preserving order\n            cur = [0]\n            cur_load = 0\n            for v in r[1:-1]:\n                d = self.demands[v]\n                if cur_load + d > self.vehicle_capacity:\n                    cur.append(0)\n                    new_routes.append(cur)\n                    cur = [0]\n                    cur_load = 0\n                cur.append(v)\n                cur_load += d\n            cur.append(0)\n            new_routes.append(cur)\n        # flatten and ensure everyone visited once: cover missed nodes by insertion if something wrong\n        flat = []\n        for rr in new_routes:\n            flat.extend(rr)\n        # fix duplicates or missing: simple repair using unvisited list\n        visited = [x for x in flat if x != 0]\n        missing = [c for c in self.customers if c not in visited]\n        # append missing greedily\n        if missing:\n            # append as new routes using greedy fill\n            bucket = []\n            load = 0\n            for v in missing:\n                if load + self.demands[v] > self.vehicle_capacity:\n                    bucket = [0] + bucket + [0]\n                    flat.extend(bucket)\n                    bucket = []\n                    load = 0\n                bucket.append(v)\n                load += self.demands[v]\n            if bucket:\n                flat.extend([0] + bucket + [0])\n        # remove accidental duplicates beyond one visit: ensure single visit by naive filtering pref: keep first occurrence\n        seen = set()\n        final = []\n        for x in flat:\n            if x == 0:\n                final.append(0)\n                continue\n            if x not in seen:\n                final.append(x)\n                seen.add(x)\n            # if duplicate encountered, skip\n        # ensure all visited; if still missing due to duplicates removed, append them\n        if any(c not in seen for c in self.customers):\n            for c in self.customers:\n                if c not in seen:\n                    final.extend([0, c, 0])\n                    seen.add(c)\n        # compress adjacent zeros\n        compressed = []\n        prev = None\n        for x in final:\n            if x == 0 and prev == 0:\n                continue\n            compressed.append(x)\n            prev = x\n        if compressed[0] != 0:\n            compressed = [0] + compressed\n        if compressed[-1] != 0:\n            compressed = compressed + [0]\n        return compressed\n\n    # ---------------- Initial ensemble generation ----------------\n\n    def _initial_greedy_nn(self):\n        unvisited = set(self.customers)\n        routes = []\n        while unvisited:\n            cur = 0\n            route = [0]\n            load = 0\n            while True:\n                # find nearest unvisited that fits\n                candidates = [v for v in unvisited if load + self.demands[v] <= self.vehicle_capacity]\n                if not candidates:\n                    break\n                nextv = min(candidates, key=lambda v: self.distance_matrix[cur, v])\n                route.append(nextv)\n                load += self.demands[nextv]\n                unvisited.remove(nextv)\n                cur = nextv\n            route.append(0)\n            routes.append(route)\n        return self._solution_from_routes(routes)\n\n    def _initial_randomized(self):\n        perm = self.customers[:]\n        random.shuffle(perm)\n        routes = []\n        cur = [0]\n        load = 0\n        for v in perm:\n            if load + self.demands[v] > self.vehicle_capacity:\n                cur.append(0)\n                routes.append(cur)\n                cur = [0]\n                load = 0\n            cur.append(v)\n            load += self.demands[v]\n        cur.append(0)\n        routes.append(cur)\n        return self._solution_from_routes(routes)\n\n    def _initial_savings(self):\n        # Clarke-Wright savings heuristic (simple implementation)\n        n = self.n\n        savings = []\n        depot = 0\n        for i in range(1, n):\n            for j in range(i+1, n):\n                s = self.distance_matrix[depot, i] + self.distance_matrix[depot, j] - self.distance_matrix[i, j]\n                savings.append(((i, j), s))\n        savings.sort(key=lambda x: -x[1])\n        # start with each customer in its own route\n        routes = {i: [0, i, 0] for i in self.customers}\n        route_load = {i: self.demands[i] for i in self.customers}\n        # mapping from node to route id\n        route_of = {i: i for i in self.customers}\n        for (i, j), _ in savings:\n            ri = route_of[i]\n            rj = route_of[j]\n            if ri == rj:\n                continue\n            if route_load[ri] + route_load[rj] > self.vehicle_capacity:\n                continue\n            r_i = routes[ri]\n            r_j = routes[rj]\n            # check if i at end of its route and j at start (so concatenation keeps depot separators)\n            if r_i[-2] == i and r_j[1] == j:\n                new = r_i[:-1] + r_j[1:]\n            elif r_j[-2] == j and r_i[1] == i:\n                new = r_j[:-1] + r_i[1:]\n            else:\n                continue\n            # merge\n            new_id = ri\n            routes[new_id] = new\n            route_load[new_id] = route_load[ri] + route_load[rj]\n            # reassign nodes of rj to ri\n            for node in r_j:\n                if node != 0:\n                    route_of[node] = new_id\n            # remove rj\n            del routes[rj]\n            del route_load[rj]\n        return self._solution_from_routes(list(routes.values()))\n\n    # ---------------- Critics (local operators) ----------------\n\n    def _init_critics(self):\n        # register critic functions and initial weights\n        self.critics = {\n            '2opt_intra': self._critic_2opt_intra,\n            'relocate': self._critic_relocate,\n            'swap': self._critic_swap,\n            'merge_split': self._critic_merge_split\n        }\n        self.critic_weights = {k: 1.0 for k in self.critics.keys()}\n\n    def _critic_2opt_intra(self, sol):\n        \"\"\"\n        Vectorized best-improvement single 2-opt pass on a randomly chosen route (if possible).\n        This replaces the prior simple random reversal with a more informed best-improvement scan,\n        while preserving the original semantics: operate only on inner nodes (do not touch depot positions).\n        Returns a flat solution (list) after applying at most one best 2-opt reversal on the selected route.\n        \"\"\"\n        routes = self._routes_from_solution(sol)\n        routes2 = deepcopy(routes)\n        # choose candidate routes that have at least 2 inner nodes (route length >= 5 -> [0,a,b,0] is minimal)\n        idx_candidates = [i for i, r in enumerate(routes2) if len(r) > 4]\n        if not idx_candidates:\n            return sol\n        idx_choice = random.choice(idx_candidates)\n        r = routes2[idx_choice]\n        # Work with numpy array for vectorized computations\n        arr = np.ascontiguousarray(np.asarray(r, dtype=np.int64).ravel())\n        m = arr.size  # length of the route including depots at positions 0 and m-1\n        dmat = self.distance_matrix\n\n        # Tolerance for numerical comparisons\n        tol = 1e-12\n\n        # Current route length (sum over consecutive node pairs)\n        cur_len = float(np.sum(dmat[arr, np.roll(arr, -1)]))\n\n        best_gain = 0.0  # gains are (new - old); negative is improvement\n        best_i = -1\n        best_j = -1\n\n        # Precompute rolled indices to get next node after any position\n        arr_next = np.roll(arr, -1)\n\n        # Iterate i only over inner positions (avoid depot at pos 0). This matches original boundaries:\n        # a in [1 .. m-3], b in [a+1 .. m-2]\n        # So i from 1 to m-3 inclusive\n        if m < 5:\n            # nothing to do\n            return sol\n\n        for i in range(1, m - 2):\n            a = int(arr[i])\n            b = int(arr[i + 1])\n            dab = dmat[a, b]\n\n            j_start = i + 1\n            j_end_exclusive = m - 1  # ensure b (=idx j) <= m-2, so exclusive upper bound is m-1\n            if j_start >= j_end_exclusive:\n                continue\n\n            # Preload rows\n            row_a = dmat[a]\n            row_b = dmat[b]\n\n            # Candidate j positions and corresponding c and d nodes\n            j_candidates = np.arange(j_start, j_end_exclusive)\n            c_nodes = arr[j_candidates]\n            d_nodes = arr_next[j_candidates]  # node after c in the route\n\n            # Compute vectorized gains for all candidate j:\n            # gain = (a->c) + (b->d) - (a->b) - (c->d)\n            gains = row_a[c_nodes] + row_b[d_nodes] - (dab + dmat[c_nodes, d_nodes])\n\n            local_min_gain = float(np.min(gains))\n            # If best for this i cannot beat global best, skip\n            if local_min_gain >= best_gain - tol:\n                continue\n\n            # Get first minimal gain position for this i\n            local_pos = int(np.argmin(gains))\n            actual_gain = float(gains[local_pos])\n            if actual_gain < best_gain - tol:\n                best_gain = actual_gain\n                best_i = i\n                best_j = int(j_candidates[local_pos])\n\n        # If improvement found (best_gain negative), apply reversal on slice (i+1 .. j) inclusive\n        if best_gain < -tol and best_i >= 0 and best_j >= 0:\n            i = best_i\n            j = best_j\n            new_arr = arr.copy()\n            # reverse inner segment\n            if i + 1 <= j:\n                new_arr[i + 1 : j + 1] = new_arr[i + 1 : j + 1][::-1]\n            # Replace route in routes2 and return flattened solution\n            routes2[idx_choice] = list(new_arr)\n            new_solution = self._solution_from_routes(routes2)\n            # If any route becomes infeasible, return projector suggestion\n            if any(self._route_demand(rt) > self.vehicle_capacity for rt in routes2):\n                return self._feasibility_projector(new_solution)\n            return new_solution\n\n        # No improvement found, return original solution\n        return sol\n\n    def _critic_relocate(self, sol):\n        # Move one customer from one route to another position in same or different route\n        routes = self._routes_from_solution(sol)\n        if len(routes) <= 1:\n            return sol\n        routes2 = deepcopy(routes)\n        # pick source route with at least one customer\n        src_idx = random.randrange(len(routes2))\n        if len(routes2[src_idx]) <= 2:\n            return sol\n        node_idx = random.randint(1, len(routes2[src_idx]) - 2)\n        node = routes2[src_idx].pop(node_idx)\n        # choose target route (could be same)\n        tgt_idx = random.randrange(len(routes2))\n        insert_pos = random.randint(1, len(routes2[tgt_idx]) - 1)\n        routes2[tgt_idx].insert(insert_pos, node)\n        # cleanup empty routes\n        cleaned = [r for r in routes2 if len(r) > 2]\n        new = self._solution_from_routes(cleaned)\n        if any(self._route_demand(rt) > self.vehicle_capacity for rt in cleaned):\n            return self._feasibility_projector(new)\n        return new\n\n    def _critic_swap(self, sol):\n        # Swap two customers across routes or within route\n        routes = self._routes_from_solution(sol)\n        routes2 = deepcopy(routes)\n        # collect customer positions\n        positions = []\n        for i, r in enumerate(routes2):\n            for j in range(1, len(r)-1):\n                positions.append((i, j))\n        if len(positions) < 2:\n            return sol\n        (i1, j1), (i2, j2) = random.sample(positions, 2)\n        routes2[i1][j1], routes2[i2][j2] = routes2[i2][j2], routes2[i1][j1]\n        # check capacities\n        if any(self._route_demand(rt) > self.vehicle_capacity for rt in routes2):\n            return self._feasibility_projector(self._solution_from_routes(routes2))\n        return self._solution_from_routes(routes2)\n\n    def _critic_merge_split(self, sol):\n        # Try merging two small routes and re-splitting them better\n        routes = self._routes_from_solution(sol)\n        if len(routes) < 2:\n            return sol\n        # pick two routes at random\n        i, j = random.sample(range(len(routes)), 2)\n        merged = routes[i][:-1] + routes[j][1:]\n        # split merged greedily\n        new_routes = []\n        cur = [0]\n        load = 0\n        for v in merged[1:-1]:\n            if load + self.demands[v] > self.vehicle_capacity:\n                cur.append(0)\n                new_routes.append(cur)\n                cur = [0]\n                load = 0\n            cur.append(v)\n            load += self.demands[v]\n        cur.append(0)\n        new_routes.append(cur)\n        # compose final set replacing i and j with new_routes\n        final_routes = [r for k, r in enumerate(routes) if k not in (i, j)]\n        final_routes.extend(new_routes)\n        new = self._solution_from_routes(final_routes)\n        return new\n\n    # ---------------- Archive management ----------------\n\n    def _extract_fragments(self, sol):\n        # fragments are sequences inside routes excluding depot ends\n        frags = []\n        routes = self._routes_from_solution(sol)\n        for r in routes:\n            inner = r[1:-1]\n            if not inner:\n                continue\n            # all contiguous subsegments of length 1..len\n            L = len(inner)\n            for a in range(0, L):\n                for b in range(a, min(L, a+5)):  # limit fragment length to 5 for efficiency\n                    frag = tuple(inner[a:b+1])\n                    if frag:\n                        frags.append(frag)\n        return frags\n\n    def _archive_add(self, frag, score, iter_idx):\n        prev = self.archive.get(frag)\n        if prev is None or score < prev[0]:\n            self.archive[frag] = (score, iter_idx)\n\n    def _archive_sample(self):\n        # sample fragments biased by quality (lower score better)\n        if not self.archive:\n            return None\n        items = list(self.archive.items())\n        frags = [k for k, v in items]\n        scores = np.array([v[0] for k, v in items], dtype=float)\n        # convert to weights: smaller score -> larger weight\n        weights = (scores.max() - scores + 1e-3)\n        probs = weights / weights.sum()\n        return frags[np.random.choice(len(frags), p=probs)]\n\n    # ---------------- Consensus sketch ----------------\n\n    def _consensus_sketch(self, ensemble):\n        # ensemble: list of (sol, score)\n        edge_counter = Counter()\n        for sol, _ in ensemble:\n            prev = sol[0]\n            for v in sol[1:]:\n                edge_counter[(prev, v)] += 1\n                prev = v\n        threshold = max(1, int(0.5 * len(ensemble)))\n        favored = set([e for e, cnt in edge_counter.items() if cnt >= threshold])\n        return favored\n\n    # ---------------- Hybrid construction ----------------\n\n    def _hybrid_from_parents(self, parent_a, parent_b, consensus):\n        # Build solution preferring consensus edges and parent adjacency\n        # Start from depot, greedily pick next node among remaining preferring edges (cur,next) in consensus,\n        # else choose next according to parent adjacency, else nearest neighbor.\n        remaining = set(self.customers)\n        # map adjacency from parents\n        adj = defaultdict(list)\n        for p in (parent_a, parent_b):\n            prev = p[0]\n            for v in p[1:]:\n                if prev != 0 and v != 0:\n                    adj[prev].append(v)\n                prev = v\n        routes = []\n        while remaining:\n            cur = 0\n            route = [0]\n            load = 0\n            while True:\n                candidates = [v for v in remaining if load + self.demands[v] <= self.vehicle_capacity]\n                if not candidates:\n                    break\n                # prefer candidate that forms a favored edge (cur, v)\n                favored = [v for v in candidates if (cur, v) in consensus]\n                chosen = None\n                if favored:\n                    chosen = random.choice(favored)\n                else:\n                    # then prefer adjacency from parents\n                    adjs = [v for v in candidates if any(v == w for w in adj.get(cur, []))]\n                    if adjs:\n                        chosen = random.choice(adjs)\n                    else:\n                        # sample nearest among candidates\n                        chosen = min(candidates, key=lambda v: self.distance_matrix[cur, v])\n                route.append(chosen)\n                remaining.remove(chosen)\n                load += self.demands[chosen]\n                cur = chosen\n            route.append(0)\n            routes.append(route)\n        # occasionally inject an archive fragment: try to merge a sampled fragment into a random route\n        frag = self._archive_sample()\n        if frag is not None and len(frag) > 0:\n            # choose route where insertion is feasible\n            for _ in range(3):\n                ri = random.randrange(len(routes))\n                r = routes[ri]\n                frag_load = sum(self.demands[v] for v in frag)\n                if self._route_demand(r) + frag_load <= self.vehicle_capacity:\n                    # insert fragment at random inner position\n                    pos = random.randint(1, len(r)-1)\n                    r2 = r[:pos] + list(frag) + r[pos:]\n                    routes[ri] = r2\n                    break\n        return self._solution_from_routes(routes)\n\n    # ---------------- Acceptance rule and novelty ----------------\n\n    def _novelty_score(self, sol, consensus):\n        # number of edges not in consensus (higher means more novel)\n        cnt = 0\n        prev = sol[0]\n        for v in sol[1:]:\n            if (prev, v) not in consensus:\n                cnt += 1\n            prev = v\n        return cnt\n\n    # ---------------- Main solver ----------------\n\n    def solve(self) -> list:\n        start_time = time.time()\n\n        # build initial ensemble\n        ensemble = []\n        candidates = []\n        try:\n            candidates.append(self._initial_greedy_nn())\n        except Exception:\n            pass\n        try:\n            candidates.append(self._initial_savings())\n        except Exception:\n            pass\n        # fill rest with randomized\n        while len(candidates) < self.ensemble_size:\n            candidates.append(self._initial_randomized())\n        # evaluate and ensure feasibility\n        for sol in candidates[:self.ensemble_size]:\n            solp = self._feasibility_projector(sol)\n            score = self._objective(solp)\n            ensemble.append((solp, score))\n            # populate archive fragments\n            frags = self._extract_fragments(solp)\n            for f in frags:\n                self._archive_add(f, score, 0)\n\n        # keep ensemble sorted by score ascending\n        ensemble.sort(key=lambda x: x[1])\n\n        iter_idx = 1\n        restructure_interval = max(10, int(0.05 * self.max_iters))\n        while iter_idx <= self.max_iters and (time.time() - start_time) < self.time_limit:\n            # consensus\n            consensus = self._consensus_sketch(ensemble)\n\n            # produce hybrid proposals\n            proposals = []\n            for _ in range(self.ensemble_size):\n                # pick parents (biased towards better)\n                ranks = list(range(len(ensemble)))\n                weights = np.array([1.0/(1+i) for i in ranks])\n                probs = weights / weights.sum()\n                idxs = np.random.choice(len(ensemble), size=2, replace=False, p=probs)\n                p_a = ensemble[idxs[0]][0]\n                p_b = ensemble[idxs[1]][0]\n                hybrid = self._hybrid_from_parents(p_a, p_b, consensus)\n                proposals.append(hybrid)\n\n            # for each proposal, invoke committee of critics (sample critics by weight)\n            for prop in proposals:\n                committee = []\n                names = list(self.critics.keys())\n                w = np.array([self.critic_weights[n] for n in names], dtype=float)\n                if w.sum() == 0:\n                    w += 1.0\n                probs = w / w.sum()\n                k = min(3, len(names))\n                chosen = np.random.choice(names, size=k, replace=False, p=probs)\n                edited = prop\n                applied_from = None\n                for cname in chosen:\n                    func = self.critics[cname]\n                    candidate = func(edited)\n                    # feasibility projector ensures feasibility\n                    candidate = self._feasibility_projector(candidate)\n                    # accept immediate edits if they improve over edited\n                    if self._objective(candidate) <= self._objective(edited):\n                        edited = candidate\n                        applied_from = cname\n                score_edited = self._objective(edited)\n                # acceptance rule: if better than worst in ensemble OR novel enough\n                worst_score = ensemble[-1][1]\n                novelty = self._novelty_score(edited, consensus)\n                accept = False\n                # accept if improvement to best of ensemble or improves worst\n                if score_edited < worst_score:\n                    accept = True\n                else:\n                    # accept with small probability if novel\n                    if novelty > int(0.2 * self.n) and random.random() < 0.3:\n                        accept = True\n                if accept:\n                    # replace worst if edited better than worst, else replace random low performer\n                    if score_edited < worst_score:\n                        ensemble[-1] = (edited, score_edited)\n                    else:\n                        # replace a random one among bottom half to maintain diversity\n                        idx_replace = random.randint(len(ensemble)//2, len(ensemble)-1)\n                        ensemble[idx_replace] = (edited, score_edited)\n                    # update critic weights: reinforce applied critic(s)\n                    if applied_from is not None:\n                        self.critic_weights[applied_from] = self.critic_weights.get(applied_from, 1.0) + 0.5\n                    # update archive with fragments\n                    frags = self._extract_fragments(edited)\n                    for f in frags:\n                        self._archive_add(f, score_edited, iter_idx)\n                else:\n                    # attenuate critics that were tried\n                    for cname in chosen:\n                        self.critic_weights[cname] = max(0.1, self.critic_weights.get(cname, 1.0) * 0.98)\n\n            # sort ensemble\n            ensemble.sort(key=lambda x: x[1])\n\n            # occasionally restructure: replace some low performers with recombinations seeded from archive/consensus\n            if iter_idx % restructure_interval == 0:\n                num_replace = max(1, int(0.2 * len(ensemble)))\n                for _ in range(num_replace):\n                    frag = self._archive_sample()\n                    # combine best with fragment to create new candidate\n                    base = ensemble[0][0]\n                    if frag is None:\n                        newc = self._initial_randomized()\n                    else:\n                        # insert fragment into base at feasible point\n                        routes = self._routes_from_solution(base)\n                        routes2 = deepcopy(routes)\n                        inserted = False\n                        random.shuffle(routes2)\n                        for r in routes2:\n                            if self._route_demand(r) + sum(self.demands[v] for v in frag) <= self.vehicle_capacity:\n                                pos = random.randint(1, len(r)-1)\n                                r[pos:pos] = list(frag)\n                                inserted = True\n                                break\n                        if not inserted:\n                            newc = self._hybrid_from_parents(ensemble[0][0], ensemble[-1][0], consensus)\n                        else:\n                            newc = self._solution_from_routes(routes2)\n                    newc = self._feasibility_projector(newc)\n                    newscore = self._objective(newc)\n                    # replace a low performer\n                    ensemble[-1] = (newc, newscore)\n                    ensemble.sort(key=lambda x: x[1])\n\n            # small normalization of critic weights to avoid explosion\n            total_w = sum(self.critic_weights.values())\n            if total_w > 0:\n                for k in self.critic_weights:\n                    self.critic_weights[k] /= total_w / len(self.critic_weights)\n\n            iter_idx += 1\n\n        # Post-processing: apply concentrated critic cascade to best candidate\n        best_sol, best_score = ensemble[0]\n        polished = deepcopy(best_sol)\n        # cascade: try many local improvements\n        for _ in range(200):\n            # attempt best-improving single critic application\n            improved = False\n            for cname, func in self.critics.items():\n                cand = func(polished)\n                cand = self._feasibility_projector(cand)\n                if self._objective(cand) < self._objective(polished) - 1e-8:\n                    polished = cand\n                    improved = True\n                    break\n            if not improved:\n                break\n        polished = self._feasibility_projector(polished)\n        # final ensure format and return\n        final_solution = polished\n        # ensure every customer visited exactly once\n        visited = [x for x in final_solution if x != 0]\n        missing = [c for c in self.customers if c not in visited]\n        if missing:\n            # add missing as separate routes\n            for m in missing:\n                final_solution.extend([0, m, 0])\n        # compress zeros\n        compressed = []\n        prev = None\n        for x in final_solution:\n            if x == 0 and prev == 0:\n                continue\n            compressed.append(x)\n            prev = x\n        if compressed[0] != 0:\n            compressed = [0] + compressed\n        if compressed[-1] != 0:\n            compressed = compressed + [0]\n        return compressed",
            "# our updated program here\nimport numpy as np\nimport time\nimport random\nfrom collections import deque, Counter\n\nclass FSSPSolver:\n    def __init__(self, num_jobs: int, num_machines: int, processing_times: list):\n        \"\"\"\n        Initialize the FSSP solver.\n\n        Args:\n            num_jobs: Number of jobs in the problem\n            num_machines: Number of machines in the problem\n            processing_times: List of lists where processing_times[j][m] is the processing time of job j on machine m\n        \"\"\"\n        self.num_jobs = num_jobs\n        self.num_machines = num_machines\n        self.processing_times = np.array(processing_times, dtype=float)\n\n    def solve(self) -> list:\n        \"\"\"\n        Solve the Flow Shop Scheduling Problem (FSSP).\n\n        Returns:\n            A list representing the sequence of jobs to be processed.\n        \"\"\"\n        # Basic parameters (adaptive to instance size)\n        n = self.num_jobs\n        max_time = 1.5 + 0.02 * n  # seconds budget heuristic\n        max_iters = 300 + 10 * n\n        ensemble_size = min(20, max(5, n // 2))\n        archive_capacity = 200\n        random_seed = None\n\n        rand = np.random.RandomState() if random_seed is None else np.random.RandomState(random_seed)\n\n        # Helper: compute makespan of a permutation (flow shop)\n        def makespan(seq):\n            \"\"\"\n            Compute the flow-shop makespan for a given job sequence `seq`.\n\n            Expectations:\n            - This function is intended to be used as a method and accesses:\n                self.num_machines (int), self.num_jobs (int), self.processing_times\n              where processing_times is an array-like of shape (num_jobs, num_machines).\n            - `seq` is any iterable of job indices (ints). It may be a sized sequence or\n              a one-shot iterator.\n\n            Features / improvements:\n            - Robust handling of arbitrary iterables/iterators (doesn't assume seq is a list).\n            - Validates job indices and processing-times shape.\n            - Fast path for single-machine case.\n            - Local variable aliasing to minimize attribute lookups.\n            - Caches a tuple-of-tuples representation of processing times on self\n              for fast repeated access.\n            \"\"\"\n            from itertools import chain\n\n            self_obj = self\n            # Read and coerce counts once\n            try:\n                m = int(self_obj.num_machines)\n                nj = int(self_obj.num_jobs)\n            except Exception as exc:\n                raise AttributeError(\"self must provide num_machines and num_jobs as integers\") from exc\n\n            if m <= 0 or nj <= 0:\n                return 0.0\n\n            # Validate or (re)build a tuple-of-tuples cache for processing times\n            pt_cache = getattr(self_obj, \"_pt_tup_cache\", None)\n            valid_cache = False\n            if isinstance(pt_cache, tuple):\n                # Quick structural check; guard against malformed cache\n                try:\n                    if len(pt_cache) == nj and all(isinstance(row, tuple) and len(row) == m for row in pt_cache):\n                        valid_cache = True\n                except Exception:\n                    valid_cache = False\n\n            if not valid_cache:\n                # Try to obtain a row-wise iterable representation; support numpy, lists, etc.\n                raw = None\n                proc = getattr(self_obj, \"processing_times\", None)\n                if proc is None:\n                    raise AttributeError(\"self must provide processing_times\")\n\n                # Prefer .tolist() for numpy-like objects, fall back to iterating\n                try:\n                    raw = proc.tolist()\n                except Exception:\n                    try:\n                        raw = list(proc)\n                    except Exception as exc:\n                        raise TypeError(\"Unable to interpret processing_times as a 2D sequence\") from exc\n\n                if len(raw) != nj:\n                    raise ValueError(f\"processing_times has unexpected number of jobs: {len(raw)} != {nj}\")\n\n                # Convert to tuple-of-tuples of floats (ensures immutability & fast indexing)\n                try:\n                    pt_cache = tuple(tuple(float(x) for x in row) for row in raw)\n                except Exception as exc:\n                    raise TypeError(\"processing_times must be convertible to floats\") from exc\n\n                # store cache\n                try:\n                    self_obj._pt_tup_cache = pt_cache\n                except Exception:\n                    # If we cannot set attribute, continue without caching\n                    pass\n\n            pt = pt_cache  # local alias for speed\n\n            # Normalize seq to an iterator and handle emptiness robustly\n            try:\n                if hasattr(seq, \"__len__\"):\n                    if len(seq) == 0:\n                        return 0.0\n                    iterable = iter(seq)\n                else:\n                    it = iter(seq)\n                    try:\n                        first = next(it)\n                    except StopIteration:\n                        return 0.0\n                    iterable = chain((first,), it)\n            except TypeError:\n                raise TypeError(\"seq must be an iterable of job indices\")\n\n            # Fast path: single machine (no precedence constraints across machines)\n            if m == 1:\n                total = 0.0\n                for job in iterable:\n                    try:\n                        j = int(job)\n                    except Exception:\n                        raise TypeError(f\"job index {job!r} is not convertible to int\")\n                    if j < 0 or j >= nj:\n                        raise IndexError(f\"job index out of range: {j}\")\n                    total += pt[j][0]\n                return float(total)\n\n            # General case: maintain per-machine completion times (mc[i] = completion time of machine i)\n            mc = [0.0] * m\n            # Precompute a range for machines 1..m-1\n            rng = range(1, m)\n\n            for job in iterable:\n                try:\n                    j = int(job)\n                except Exception:\n                    raise TypeError(f\"job index {job!r} is not convertible to int\")\n                if j < 0 or j >= nj:\n                    raise IndexError(f\"job index out of range: {j}\")\n\n                job_times = pt[j]\n\n                # Machine 0: no precedence from previous machines on this job\n                prev = mc[0] + job_times[0]\n                mc[0] = prev\n\n                # Remaining machines: each machine i can start only after both\n                # - machine i finished its previous job (mc[i])\n                # - this job finished on machine i-1 (prev)\n                for i in rng:\n                    # start = max(mc[i], prev) ; completion = start + job_times[i]\n                    # use local variables to reduce attribute lookups\n                    cur_mc = mc[i]\n                    if cur_mc > prev:\n                        prev = cur_mc + job_times[i]\n                    else:\n                        prev = prev + job_times[i]\n                    mc[i] = prev\n\n            return float(mc[-1])\n\n        # NEH heuristic (good constructive)\n        def neh_sequence():\n            totals = self.processing_times.sum(axis=1)\n            order = list(np.argsort(-totals))  # descending total processing time\n            seq = []\n            for job in order:\n                best_seq = None\n                best_val = float('inf')\n                for pos in range(len(seq) + 1):\n                    cand = seq[:pos] + [job] + seq[pos:]\n                    val = makespan(cand)\n                    if val < best_val:\n                        best_val = val\n                        best_seq = cand\n                seq = best_seq\n            return seq\n\n        # Simple constructive heuristics\n        def heuristic_shortest_total():\n            order = list(np.argsort(self.processing_times.sum(axis=1)))\n            return order\n\n        def heuristic_longest_total():\n            order = list(np.argsort(-self.processing_times.sum(axis=1)))\n            return order\n\n        def random_shuffle():\n            seq = list(range(n))\n            rand.shuffle(seq)\n            return seq\n\n        def greedy_insertion_by_first_machine():\n            # sort by processing time on machine 0 ascending\n            order = list(np.argsort(self.processing_times[:, 0]))\n            seq = []\n            for job in order:\n                best_seq = None\n                best_val = float('inf')\n                for pos in range(len(seq) + 1):\n                    cand = seq[:pos] + [job] + seq[pos:]\n                    val = makespan(cand)\n                    if val < best_val:\n                        best_val = val\n                        best_seq = cand\n                seq = best_seq\n            return seq\n\n        # Critics (local improvement operators). Each returns a list of candidate sequences\n        def critic_swap(seq, limit=10):\n            # random pairwise swaps\n            candidates = []\n            L = len(seq)\n            for _ in range(limit):\n                a, b = rand.randint(0, L), rand.randint(0, L)\n                if a == b:\n                    continue\n                s = seq.copy()\n                s[a], s[b] = s[b], s[a]\n                candidates.append(s)\n            return candidates\n\n        def critic_insert(seq, limit=10):\n            # remove and insert\n            candidates = []\n            L = len(seq)\n            for _ in range(limit):\n                i = rand.randint(0, L)\n                j = rand.randint(0, L)\n                if i == j:\n                    continue\n                s = seq.copy()\n                job = s.pop(i)\n                s.insert(j, job)\n                candidates.append(s)\n            return candidates\n\n        def critic_two_opt(seq, limit=8):\n            candidates = []\n            L = len(seq)\n            for _ in range(limit):\n                i = rand.randint(0, L - 1)\n                j = rand.randint(i + 1, L)\n                s = seq.copy()\n                s[i:j] = reversed(s[i:j])\n                candidates.append(s)\n            return candidates\n\n        def critic_block_move(seq, limit=6):\n            # cut a block and insert elsewhere\n            candidates = []\n            L = len(seq)\n            for _ in range(limit):\n                i = rand.randint(0, max(0, L - 1))\n                j = rand.randint(i + 1, L)\n                block = seq[i:j]\n                s = seq[:i] + seq[j:]\n                k = rand.randint(0, len(s) + 1)\n                s = s[:k] + block + s[k:]\n                if s != seq:\n                    candidates.append(s)\n            return candidates\n\n        critic_pool = {\n            'swap': critic_swap,\n            'insert': critic_insert,\n            'two_opt': critic_two_opt,\n            'block_move': critic_block_move\n        }\n        critic_names = list(critic_pool.keys())\n        critic_influence = {name: 1.0 for name in critic_names}\n\n        # Archive of fragments (subsequences) with scores\n        archive = deque(maxlen=archive_capacity)\n\n        # Ensemble initialization: diverse constructive strategies\n        ensemble = []\n        generators = [neh_sequence, heuristic_shortest_total, heuristic_longest_total,\n                      greedy_insertion_by_first_machine, random_shuffle]\n        # create multiple seeds from random shuffles too\n        while len(ensemble) < ensemble_size:\n            for gen in generators:\n                if len(ensemble) >= ensemble_size:\n                    break\n                seq = gen()\n                if seq is None:\n                    continue\n                # ensure permutation\n                if len(seq) != n:\n                    seq = list(range(n))\n                score = makespan(seq)\n                ensemble.append((seq, score))\n        # sort ensemble by score ascending\n        ensemble.sort(key=lambda x: x[1])\n\n        # seed archive with fragments from top ensemble members\n        def add_fragments_from(seq, score):\n            # add substrings of various lengths\n            L = len(seq)\n            for length in range(1, min(6, L) + 1):\n                # sample some starting positions\n                max_start = max(1, L - length + 1)\n                samples = min(4, max_start)\n                for s in range(samples):\n                    start = rand.randint(0, max(0, L - length))\n                    frag = tuple(seq[start:start + length])\n                    archive.append((frag, score))\n\n        for seq, sc in ensemble[:max(5, ensemble_size//2)]:\n            add_fragments_from(seq, sc)\n\n        # Consensus engine: build pairwise precedence matrix and score\n        def compute_consensus(ens):\n            # ens: list of (seq, score)\n            m = np.zeros((n, n), dtype=float)\n            count = 0\n            for seq, _ in ens:\n                pos = np.empty(n, dtype=int)\n                for idx, job in enumerate(seq):\n                    pos[job] = idx\n                for i in range(n):\n                    # jobs that i precedes\n                    # faster: for j in range(n): if pos[i] < pos[j]: m[i,j] += 1\n                    for j in range(n):\n                        if pos[i] < pos[j]:\n                            m[i, j] += 1.0\n                count += 1\n            if count > 0:\n                m /= count\n            # consensus score for each job: how often it precedes others\n            score = m.sum(axis=1)\n            # soft template: ordering by score descending (higher means earlier)\n            order = list(np.argsort(-score))\n            return m, order\n\n        # Hybridization: guided by consensus and archive fragments\n        def hybridize(cons_order, ensemble_local):\n            # create variants by respecting consensus ordering and injecting fragments\n            candidates = []\n            base = cons_order.copy()\n            # 1) precedence-preserving build: greedily pick next job that maintains many consensus pairwise preferences\n            for _ in range(4):\n                remaining = set(range(n))\n                pos_pref = np.zeros(n, dtype=float)\n                for j in range(n):\n                    pos_pref[j] = cons_order.index(j) if j in cons_order else 0\n                seq = []\n                while remaining:\n                    # choose subset of candidates with top pref rank\n                    remaining_list = list(remaining)\n                    scores = [pos_pref[j] for j in remaining_list]\n                    idx = np.argmax(scores)\n                    sel = remaining_list[idx]\n                    # with small prob pick from archive fragment head to diversify\n                    if rand.rand() < 0.25 and archive:\n                        frag, _ = archive[rand.randint(0, len(archive))]\n                        frag = list(frag)\n                        # place fragment respecting remaining\n                        frag = [f for f in frag if f in remaining]\n                        if len(frag) >= 1:\n                            for f in frag:\n                                seq.append(f)\n                                if f in remaining:\n                                    remaining.remove(f)\n                            continue\n                    seq.append(sel)\n                    remaining.remove(sel)\n                candidates.append(seq)\n                # shuffle base slightly for next repetition\n                rand.shuffle(cons_order)\n            # 2) recombine fragments from two random ensemble members\n            for _ in range(6):\n                p1 = ensemble_local[rand.randint(0, len(ensemble_local)-1)][0]\n                p2 = ensemble_local[rand.randint(0, len(ensemble_local)-1)][0]\n                # precedence respecting crossover (order crossover)\n                cut = rand.randint(1, n-1)\n                left = p1[:cut]\n                child = left + [j for j in p2 if j not in left]\n                candidates.append(child)\n            # 3) sample some archive-seeded sequences: pick fragments and fill rest by consensus\n            for _ in range(6):\n                seq = []\n                remaining = set(range(n))\n                if archive and rand.rand() < 0.7:\n                    frag, _ = archive[rand.randint(0, len(archive))]\n                    frag = list(frag)\n                    # place fragment at random position\n                    pos = rand.randint(0, n)\n                    frag = [f for f in frag if f in remaining]\n                    # fill left and right by consensus order\n                    left_pool = [j for j in cons_order if j in remaining and j not in frag]\n                    seq = []\n                    # insert some left part\n                    k = rand.randint(0, len(left_pool))\n                    seq.extend(left_pool[:k])\n                    seq.extend(frag)\n                    seq.extend([j for j in left_pool[k:]])\n                    # if missing jobs, append\n                    missing = [j for j in range(n) if j not in seq]\n                    rand.shuffle(missing)\n                    seq.extend(missing)\n                else:\n                    # random perturbation of consensus\n                    seq = cons_order.copy()\n                    for _ in range(max(1, n//6)):\n                        i = rand.randint(0, n-1)\n                        j = rand.randint(0, n-1)\n                        seq[i], seq[j] = seq[j], seq[i]\n                candidates.append(seq)\n            # ensure uniqueness and valid permutations\n            uniq = []\n            seen = set()\n            for c in candidates:\n                tup = tuple(c)\n                if len(c) == n and tup not in seen:\n                    seen.add(tup)\n                    uniq.append(c)\n            return uniq\n\n        # Acceptance rule: reward objective improvement and novelty\n        def sequence_distance(a, b):\n            # simple Hamming-like: positions equal count\n            pos_a = {job: i for i, job in enumerate(a)}\n            dist = 0\n            for idx, job in enumerate(b):\n                if pos_a.get(job, -1) != idx:\n                    dist += 1\n            return dist\n\n        def accept_into_ensemble(candidate_seq, candidate_score, ensemble_local):\n            # Accept if better than worst or if moderately novel and not too bad\n            worst_score = ensemble_local[-1][1]\n            best_score = ensemble_local[0][1]\n            # compute minimal distance to existing ensemble\n            min_dist = min(sequence_distance(candidate_seq, s) for s, _ in ensemble_local)\n            # acceptance thresholds\n            if candidate_score < worst_score - 1e-9:\n                # accept and replace worst\n                ensemble_local[-1] = (candidate_seq, candidate_score)\n                ensemble_local.sort(key=lambda x: x[1])\n                return True\n            # allow occasional acceptance for diversity if within factor\n            if candidate_score <= best_score * 1.15 and min_dist >= max(1, n//5) and rand.rand() < 0.4:\n                # replace a random low performer (not best)\n                idx = rand.randint(max(1, len(ensemble_local)//2), len(ensemble_local)-1)\n                ensemble_local[idx] = (candidate_seq, candidate_score)\n                ensemble_local.sort(key=lambda x: x[1])\n                return True\n            return False\n\n        # Restructure: replace low-performing ensemble members occasionally\n        def restructure(ensemble_local):\n            # replace half of worst with recombinations seeded from archive and consensus\n            m = len(ensemble_local)\n            replace_count = max(1, m // 3)\n            cons_mtx, cons_order = compute_consensus(ensemble_local)\n            for i in range(replace_count):\n                new_seq_candidates = hybridize(cons_order, ensemble_local)\n                best_cand = None\n                best_val = float('inf')\n                for cand in new_seq_candidates:\n                    v = makespan(cand)\n                    if v < best_val:\n                        best_val = v\n                        best_cand = cand\n                # replace one of worst performers\n                ensemble_local[-1 - i] = (best_cand, best_val)\n            ensemble_local.sort(key=lambda x: x[1])\n\n        # Concentrated critic cascade for polishing final sequence\n        def polish(seq):\n            improved = True\n            current = seq.copy()\n            current_score = makespan(current)\n            no_improve_iters = 0\n            while improved and no_improve_iters < 40:\n                improved = False\n                # try insert improvement exhaustively in randomized order\n                for i in range(n):\n                    for j in range(n):\n                        if i == j:\n                            continue\n                        s = current.copy()\n                        job = s.pop(i)\n                        s.insert(j, job)\n                        val = makespan(s)\n                        if val + 1e-9 < current_score:\n                            current = s\n                            current_score = val\n                            improved = True\n                            break\n                    if improved:\n                        break\n                if not improved:\n                    # try pairwise swaps\n                    for i in range(n-1):\n                        for j in range(i+1, n):\n                            s = current.copy()\n                            s[i], s[j] = s[j], s[i]\n                            val = makespan(s)\n                            if val + 1e-9 < current_score:\n                                current = s\n                                current_score = val\n                                improved = True\n                                break\n                        if improved:\n                            break\n                if not improved:\n                    no_improve_iters += 1\n            return current, current_score\n\n        # Main loop\n        start_time = time.time()\n        it = 0\n        restructure_cooldown = 0\n        while it < max_iters and (time.time() - start_time) < max_time:\n            it += 1\n            # compute consensus\n            cons_mtx, cons_order = compute_consensus(ensemble)\n            # hybrid proposals\n            proposals = hybridize(cons_order, ensemble)\n            # evaluate and apply critics\n            for prop in proposals:\n                base_score = makespan(prop)\n                # apply committee of critics: choose top-K by influence\n                total_infl = sum(critic_influence.values())\n                # sample critics proportional to influence\n                crit_names = critic_names.copy()\n                weights = np.array([critic_influence[n] for n in crit_names], dtype=float)\n                if weights.sum() <= 0:\n                    weights = np.ones_like(weights)\n                probs = weights / weights.sum()\n                chosen = list(np.random.choice(crit_names, size=min(3, len(crit_names)), replace=False, p=probs))\n                # generate variants\n                variants = []\n                for c in chosen:\n                    fn = critic_pool[c]\n                    try:\n                        variants.extend(fn(prop, limit=6))\n                    except TypeError:\n                        variants.extend(fn(prop))\n                # include original prop\n                variants.append(prop)\n                # evaluate variants and choose best\n                best_variant = None\n                best_val = float('inf')\n                best_critic = None\n                for v in variants:\n                    # ensure validity: permutation\n                    if len(v) != n or set(v) != set(range(n)):\n                        # repair: keep order of seen jobs then append missing\n                        seen = []\n                        seen_set = set()\n                        for job in v:\n                            if job not in seen_set and 0 <= job < n:\n                                seen.append(job)\n                                seen_set.add(job)\n                        for job in range(n):\n                            if job not in seen_set:\n                                seen.append(job)\n                        v = seen\n                    val = makespan(v)\n                    if val < best_val:\n                        best_val = val\n                        best_variant = v\n                # acceptance\n                accepted = accept_into_ensemble(best_variant, best_val, ensemble)\n                # update critic influence: reward critics that were chosen if improved\n                if accepted:\n                    for c in chosen:\n                        critic_influence[c] = critic_influence.get(c, 1.0) * 1.08\n                    # refresh archive with fragments from accepted solution\n                    add_fragments_from(best_variant, best_val)\n                else:\n                    for c in chosen:\n                        critic_influence[c] = critic_influence.get(c, 1.0) * 0.97\n                # keep influences bounded\n                for c in critic_influence:\n                    critic_influence[c] = min(max(critic_influence[c], 0.05), 20.0)\n            # refresh archive with top ensemble fragments occasionally\n            if it % 5 == 0:\n                for seq, sc in ensemble[:min(3, len(ensemble))]:\n                    add_fragments_from(seq, sc)\n            # occasional restructuring to escape local patterns\n            if it % 30 == 0 and len(ensemble) >= 3:\n                restructure(ensemble)\n            # small random restart: add a random candidate occasionally\n            if rand.rand() < 0.05:\n                rseq = random_shuffle()\n                rscore = makespan(rseq)\n                accept_into_ensemble(rseq, rscore, ensemble)\n            # end loop by time or iterations\n        # Post-processing: polish the best candidate\n        best_seq, best_score = ensemble[0]\n        polished_seq, polished_score = polish(best_seq)\n        # If polish improved, accept\n        final_seq = polished_seq if polished_score <= best_score else best_seq\n        # ensure valid permutation\n        if len(final_seq) != n or set(final_seq) != set(range(n)):\n            # fallback to best ensemble unique repair\n            seq = list(range(n))\n            final_seq = seq\n        return final_seq"
        ],
        "function_bodies": [
            "# Your key function here\ndef _apply_2opt_once(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:\n    \"\"\"\n    Vectorized/optimized best-improvement single 2-opt pass.\n    Tries all i, and for each i computes gains for all feasible j in a vectorized manner\n    to avoid Python-level inner loops. Returns improved tour and its length; if none found,\n    returns original tour and its length.\n    \"\"\"\n    n = self.n\n    dmat = self.distance_matrix\n    idx = tour\n    cur_len = float(np.sum(dmat[idx, np.roll(idx, -1)]))  # current tour length\n    best_gain = 0.0\n    best_i = -1\n    best_j = -1\n\n    # Loop over i but vectorize j computations\n    for i in range(0, n - 2):\n        a = int(idx[i])\n        b = int(idx[i + 1])\n        dab = dmat[a, b]\n\n        # compute candidate j range (matches original boundaries)\n        j_start = i + 2\n        j_end = n - (0 if i > 0 else 1)  # exclusive upper bound\n        if j_start >= j_end:\n            continue\n\n        j_candidates = np.arange(j_start, j_end)\n        # vectorized access to c and d_ nodes\n        c_nodes = idx[j_candidates]\n        d_nodes = idx[(j_candidates + 1) % n]\n\n        # vectorized gain computation: new edges - old edges\n        # gain < 0 indicates improvement\n        gains = (dmat[a, c_nodes] + dmat[b, d_nodes]) - (dab + dmat[c_nodes, d_nodes])\n\n        # find minimal gain for this i\n        local_min_idx = np.argmin(gains)\n        local_min_gain = float(gains[local_min_idx])\n        if local_min_gain < best_gain:\n            best_gain = local_min_gain\n            best_i = i\n            best_j = int(j_candidates[local_min_idx])\n\n    # If an improving 2-opt move was found, apply it efficiently and compute new length using gain\n    if best_gain < -1e-12:\n        i = best_i\n        j = best_j\n        # build new tour by reversing the segment (i+1..j)\n        # use slicing which is efficient in numpy\n        new = np.empty_like(idx)\n        if i + 1 <= j:\n            # straightforward case\n            new[:i + 1] = idx[:i + 1]\n            new[i + 1:j + 1] = idx[i + 1:j + 1][::-1]\n            new[j + 1:] = idx[j + 1:]\n        else:\n            # shouldn't happen given j > i in construction, but keep safe fallback\n            new = idx.copy()\n        new_len = cur_len + best_gain\n        return new, float(new_len)\n    else:\n        return tour, float(cur_len)",
            "# Your key function here\ndef _objective(self, sol):\n    \"\"\"\n    Optimized objective function with caching and vectorized distance summation.\n\n    Computes the total travel distance of a flat solution representation `sol`\n    by summing pairwise distances between consecutive nodes using numpy\n    advanced indexing. Results are cached (keyed by the tuple form of `sol`)\n    to avoid repeated work for identical solutions. A simple cache size\n    limiter prevents unbounded growth.\n\n    This replaces the original Python loop which was a hotspot due to very\n    frequent calls across the solver (ensemble evaluation, critic checks,\n    acceptance rules, polishing cascade, etc.).\n    \"\"\"\n    # Initialize cache on first use\n    cache = getattr(self, \"_objective_cache\", None)\n    if cache is None:\n        cache = {}\n        self._objective_cache = cache\n\n    # Use tuple as cache key (works for lists and numpy arrays)\n    if isinstance(sol, tuple):\n        key = sol\n    else:\n        # convert to tuple without modifying original; efficient for short lists\n        key = tuple(sol)\n\n    # Return cached value if present\n    if key in cache:\n        return cache[key]\n\n    # Vectorized sum of distances between consecutive nodes\n    arr = np.asarray(key, dtype=np.int64)\n    if arr.size < 2:\n        total = 0.0\n    else:\n        prevs = arr[:-1]\n        nexts = arr[1:]\n        # advanced indexing on the precomputed distance matrix is done in C,\n        # avoiding Python-level loops and speeding up repeated evaluations.\n        total = float(self.distance_matrix[prevs, nexts].sum())\n\n    # Simple cache size control to avoid unbounded memory use\n    MAX_CACHE = 10000\n    if len(cache) >= MAX_CACHE:\n        # evict the oldest inserted key (dict preserves insertion order in CPython >=3.7)\n        try:\n            oldest = next(iter(cache))\n            cache.pop(oldest, None)\n        except StopIteration:\n            pass\n    cache[key] = total\n    return total",
            "# Your key function here\ndef makespan(seq):\n    \"\"\"\n    Optimized makespan computation for the flow-shop.\n    - Caches a Python list copy of self.processing_times on the instance to avoid\n      repeated numpy-to-python conversion overhead.\n    - Minimizes attribute lookups by binding locals.\n    - Avoids Python max() calls in the inner loop (uses an if for branchless-ish speed).\n    \"\"\"\n    # bind locals for speed\n    self_obj = self\n    m = self_obj.num_machines\n    if m == 0 or self_obj.num_jobs == 0:\n        return 0.0\n\n    # ensure sequence is a sequence of ints (handles numpy arrays)\n    # short-circuit empty seq\n    if not seq:\n        return 0.0\n\n    # cache processing times as list-of-lists on the instance to avoid repeated numpy overhead\n    # validate cache shape; recreate if inconsistent\n    pt_cache = getattr(self_obj, \"_pt_list_cache\", None)\n    if pt_cache is None or len(pt_cache) != self_obj.num_jobs or len(pt_cache[0]) != m:\n        # convert once and store\n        # using .tolist() then ensuring float conversion for numeric stability\n        try:\n            raw = self_obj.processing_times.tolist()\n        except Exception:\n            # fallback if processing_times is already a list\n            raw = list(self_obj.processing_times)\n        pt_cache = [list(map(float, row)) for row in raw]\n        self_obj._pt_list_cache = pt_cache\n\n    pt = pt_cache\n\n    # machine completion times (float)\n    mc = [0.0] * m\n\n    # main loop: for each job, update completion times across machines\n    for job in seq:\n        j = int(job)  # handle numpy ints\n        job_times = pt[j]\n        # first machine: simple accumulation\n        prev = mc[0] + job_times[0]\n        mc[0] = prev\n        # subsequent machines: start = max(mc[i], prev); t = start + job_times[i]\n        for i in range(1, m):\n            mi = mc[i]\n            if mi > prev:\n                start = mi\n            else:\n                start = prev\n            t = start + job_times[i]\n            mc[i] = t\n            prev = t\n\n    return float(mc[-1])"
        ]
    },
'''

import numpy as np
import time
from typing import List, Tuple, Callable, Dict

class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, rng_seed: int = 0):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
            rng_seed: optional random seed for reproducibility.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.n = len(coordinates)
        self.rng = np.random.RandomState(rng_seed)

    # ---------------- Utility functions ----------------
    def tour_length(self, tour: np.ndarray) -> float:
        idx = tour
        d = self.distance_matrix
        return float(np.sum(d[idx, np.roll(idx, -1)]))

    def _apply_2opt_once(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Vectorized best-improvement single 2-opt pass with a memory-aware fallback.

        Tries to find the single best 2-opt move (i, j) that reduces tour length.
        For moderate-sized problems it builds row-wise gains in a vectorized manner.
        For very large n it falls back to a memory-friendly per-i loop that still
        uses NumPy operations for each row.

        Returns (new_tour, new_length). If no improving move exists, returns the
        original tour and its length.
        """
        n = int(self.n)
        dmat = self.distance_matrix
        idx = tour

        # current tour length (ensure float)
        cur_len = float(np.sum(dmat[idx, np.roll(idx, -1)]))

        # nothing to do for tiny tours
        if n < 4:
            return tour, cur_len

        # i ranges 0 .. n-3 (we consider breaking edge (i,i+1) and reconnecting to (j,j+1))
        i_vals = np.arange(0, n - 2, dtype=int)
        m = i_vals.size
        a = idx[i_vals]           # node at position i
        b = idx[i_vals + 1]       # node at position i+1
        dab = dmat[a, b]          # distance for edge (a,b) per i

        # memory threshold (number of matrix cells). Tweak if needed.
        MAX_CELLS = 8_000_000

        # small tolerance for "no improvement" (to account for floating rounding)
        NO_IMPROV_TOL = -1e-12

        if m * n <= MAX_CELLS:
            # Fully vectorized path (efficient for moderate n)
            # Build grids using broadcasting where possible to avoid explicit repeats
            j_range = np.arange(n, dtype=int)
            # i_grid shape (m,1), j_grid shape (1,n) broadcastable to (m,n)
            i_grid = i_vals[:, None]
            j_grid = j_range[None, :]

            # j is valid when j >= i+2 and j < n - (i==0)
            j_max = (n - (i_vals == 0))[:, None]  # shape (m,1)
            valid_mask = (j_grid >= (i_grid + 2)) & (j_grid < j_max)

            # corresponding tour nodes c = tour[j], d = tour[j+1]
            # broadcasting idx over rows
            c_nodes = idx[j_grid]                 # shape (m, n)
            d_nodes = idx[(j_grid + 1) % n]       # shape (m, n)

            # compute distance components (each yields shape (m, n))
            # Use advanced indexing with broadcasting: a[:,None] has shape (m,1)
            d_a_c = dmat[a[:, None], c_nodes]
            d_b_d = dmat[b[:, None], d_nodes]
            d_c_d = dmat[c_nodes, d_nodes]

            # gains = (a->c + b->d) - (a->b + c->d)
            gains = d_a_c + d_b_d - (dab[:, None] + d_c_d)

            # invalidate infeasible positions
            gains[~valid_mask] = np.inf

            # find global best gain
            # use unravel_index for clarity
            flat_min_idx = int(np.argmin(gains))
            best_i_row, best_j_col = np.unravel_index(flat_min_idx, gains.shape)
            min_gain = float(gains[best_i_row, best_j_col])

            # no improving move found
            if min_gain >= NO_IMPROV_TOL:
                return tour, cur_len

            # Map to actual i, j positions in the tour
            best_i = int(i_vals[best_i_row])
            best_j = int(best_j_col)
        else:
            # Memory-conserving row-wise path: iterate over i but compute arrays per-row with NumPy
            best_gain = 0.0
            best_i = -1
            best_j = -1

            # Pre-extract idx for speed
            for ii in range(m):
                i = int(i_vals[ii])
                j_start = i + 2
                j_end = n - (1 if i == 0 else 0)  # exclusive upper bound
                if j_start >= j_end:
                    continue

                js = np.arange(j_start, j_end, dtype=int)
                c = idx[js]
                d = idx[(js + 1) % n]

                # vectorized computation for this row
                da_c = dmat[a[ii], c]
                db_d = dmat[b[ii], d]
                dc_d = dmat[c, d]
                gains_row = da_c + db_d - (dab[ii] + dc_d)

                # find best in this row
                row_min_pos = int(np.argmin(gains_row))
                row_min_gain = float(gains_row[row_min_pos])

                if row_min_gain < best_gain:
                    best_gain = row_min_gain
                    best_i = i  # actual i value (not row index)
                    best_j = int(js[row_min_pos])

            if best_i == -1:
                return tour, cur_len

            min_gain = float(best_gain)

        # Apply 2-opt: best_i is actual tour index i, best_j is actual tour index j
        i = int(best_i)
        j = int(best_j)

        # Defensive check (shouldn't happen because of masks)
        if not (0 <= i < n and 0 <= j < n and (i + 1) <= j):
            return tour, cur_len

        # Construct new tour by reversing segment between i+1 .. j (inclusive)
        new = np.empty_like(idx)
        if i + 1 <= j:
            # copy prefix up to i
            if i + 1 > 0:
                new[: i + 1] = idx[: i + 1]
            # reversed middle segment
            new[i + 1 : j + 1] = idx[i + 1 : j + 1][::-1]
            # suffix after j
            if j + 1 < n:
                new[j + 1 :] = idx[j + 1 :]
        else:
            # defensive fallback
            return tour, cur_len

        new_len = cur_len + min_gain
        return new, float(new_len)

    def two_opt_local_search(self, tour: np.ndarray, max_iters: int = 100) -> Tuple[np.ndarray, float]:
        """
        Apply repeated best 2-opt improvements until no improvement or max_iters reached.
        """
        current = tour.copy()
        cur_len = self.tour_length(current)
        it = 0
        while it < max_iters:
            new, new_len = self._apply_2opt_once(current)
            if new_len + 1e-12 < cur_len:
                current = new
                cur_len = new_len
                it += 1
            else:
                break
        return current, cur_len

    def swap_operator(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Random swap of two nodes followed by 2-opt local quick-improve."""
        n = self.n
        i, j = self.rng.choice(n, size=2, replace=False)
        new = tour.copy()
        new[i], new[j] = new[j], new[i]
        new, new_len = self.two_opt_local_search(new, max_iters=5)
        return new, new_len

    def reinsertion_operator(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Remove a block of size 1-3 and insert at random position (or-opt)."""
        n = self.n
        k = self.rng.randint(1, min(4, n))
        i = self.rng.randint(0, n)
        block = []
        for x in range(k):
            block.append(tour[(i + x) % n])
        block = np.array(block, dtype=int)
        remaining = []
        for t in tour:
            if not np.any(t == block):
                remaining.append(t)
        remaining = np.array(remaining, dtype=int)
        insert_pos = self.rng.randint(0, len(remaining) + 1)
        new = np.concatenate([remaining[:insert_pos], block, remaining[insert_pos:]])
        new_len = self.tour_length(new)
        new, new_len = self.two_opt_local_search(new, max_iters=5)
        return new, new_len

    def double_bridge(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perturbation via double-bridge move to escape local minima."""
        n = self.n
        if n < 8:
            return self.swap_operator(tour)
        # select four cut points
        pts = sorted(self.rng.choice(range(1, n), size=4, replace=False))
        a, b, c, d = pts
        p = tour
        new = np.concatenate([p[:a], p[c:d], p[b:c], p[a:b], p[d:]])
        new_len = self.tour_length(new)
        return new, new_len

    # ---------------- Ensemble and archive initialization ----------------
    def _nearest_neighbor_tour(self, start: int = 0) -> np.ndarray:
        n = self.n
        d = self.distance_matrix
        unvisited = set(range(n))
        curr = start
        tour = [curr]
        unvisited.remove(curr)
        while unvisited:
            # choose nearest unvisited
            rem = np.array(list(unvisited))
            next_idx = rem[np.argmin(d[curr, rem])]
            tour.append(int(next_idx))
            unvisited.remove(int(next_idx))
            curr = int(next_idx)
        return np.array(tour, dtype=int)

    def init_ensemble(self, ensemble_size: int) -> List[Dict]:
        """
        Generate a diverse ensemble of feasible solutions.
        Each ensemble member is a dict: {'tour': np.ndarray, 'cost': float, 'age': int, 'score': float}
        """
        ensemble = []
        n = self.n
        # seeds: nearest neighbor from several starts, greedy heuristics, random permutations
        starts = list(range(min(n, 10)))
        for s in starts:
            tour = self._nearest_neighbor_tour(start=s)
            tour, cost = self.two_opt_local_search(tour, max_iters=20)
            ensemble.append({'tour': tour, 'cost': cost, 'age': 0, 'score': -cost})
        # add random perturbed tours
        while len(ensemble) < ensemble_size:
            tour = np.arange(n)
            self.rng.shuffle(tour)
            tour, cost = self.two_opt_local_search(tour, max_iters=10)
            ensemble.append({'tour': tour, 'cost': cost, 'age': 0, 'score': -cost})
        # sort by cost
        ensemble.sort(key=lambda x: x['cost'])
        return ensemble

    # ---------------- Consensus sketch and hybrid construction ----------------
    def consensus_sketch(self, ensemble: List[Dict]) -> np.ndarray:
        """
        Build an edge frequency matrix (soft template).
        Returns a matrix P of shape (n, n) with values proportional to frequency of directed edges.
        """
        n = self.n
        P = np.zeros((n, n), dtype=float)
        for member in ensemble:
            tour = member['tour']
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                P[a, b] += 1.0
        # normalize
        total = np.sum(P)
        if total > 0:
            P /= total
        # smooth to avoid zeros
        P += 1e-6
        return P

    def build_hybrid_from_sketch(self, P: np.ndarray, archive: List[Tuple[np.ndarray, float]], blend_prob: float = 0.6) -> np.ndarray:
        """
        Construct a hybrid tour guided by consensus P and optionally seeded with fragments from archive.
        """
        n = self.n
        # attempt to place high-quality archived fragments first with some probability
        used = np.zeros(n, dtype=bool)
        tour = []
        if archive and self.rng.rand() < 0.5:
            # pick a fragment proportional to fragment quality
            frags, scores = zip(*archive)
            scores = np.array(scores)
            probs = (np.max(scores) - scores + 1e-6)
            probs /= probs.sum()
            idx = self.rng.choice(len(frags), p=probs)
            frag = frags[idx]
            # insert fragment into tour
            for node in frag:
                if not used[node]:
                    tour.append(node)
                    used[node] = True
        # pick random start among unvisited
        if not tour:
            start = self.rng.randint(0, n)
            tour.append(start)
            used[start] = True
        # greedy build using P and distance heuristics
        while len(tour) < n:
            curr = tour[-1]
            candidates = np.where(~used)[0]
            # score combining P (consensus) and inverse distance
            pvals = P[curr, candidates]
            dvals = self.distance_matrix[curr, candidates]
            invd = 1.0 / (dvals + 1e-6)
            score = pvals * 2.0 + invd  # weight consensus more
            # add a small randomization
            score = score ** (1.0) + 1e-4 * self.rng.rand(len(score))
            # sample proportional to score to maintain diversity
            probs = score / score.sum()
            chosen = self.rng.choice(candidates, p=probs)
            tour.append(int(chosen))
            used[int(chosen)] = True
        return np.array(tour, dtype=int)

    # ---------------- Archive management ----------------
    def update_archive(self, archive: List[Tuple[np.ndarray, float]], fragment: np.ndarray, cost: float, max_size: int = 50):
        """
        Maintain archive of high-quality fragments (subsequences).
        Store fragments as tuples (array, cost).
        """
        # store shorter fragments to encourage recombination
        frag_len = len(fragment)
        if frag_len < 2:
            return
        # keep unique by tuple
        key = tuple(fragment.tolist())
        for f, c in archive:
            if tuple(f.tolist()) == key:
                # update if better
                return
        archive.append((fragment.copy(), cost))
        # trim archive by best (lower cost considered better)
        archive.sort(key=lambda x: x[1])
        if len(archive) > max_size:
            del archive[max_size:]

    # ---------------- Critics committee ----------------
    def get_critics(self) -> List[Callable[[np.ndarray], Tuple[np.ndarray, float]]]:
        """
        Return the set of critic operators.
        """
        return [
            self.two_opt_local_search,
            self.swap_operator,
            self.reinsertion_operator,
            self.double_bridge
        ]

    # ---------------- Main solver ----------------
    def solve(self,
              max_iters: int = None,
              ensemble_size: int = None,
              time_limit: float = None) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP) using a Consensus-driven Critic Ensemble metaheuristic.

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.
        """
        n = self.n
        if n <= 1:
            return np.arange(n, dtype=int)
        # configure parameters heuristically
        if ensemble_size is None:
            ensemble_size = min(max(8, n // 2), 40)
        if max_iters is None:
            max_iters = 200 + 20 * int(np.sqrt(n))
        if time_limit is None:
            time_limit = 30.0  # seconds default cap
        start_time = time.time()

        # initialize ensemble and archive
        ensemble = self.init_ensemble(ensemble_size)
        archive: List[Tuple[np.ndarray, float]] = []
        # populate archive with fragments from initial ensemble
        for member in ensemble:
            tour = member['tour']
            cost = member['cost']
            # extract several random fragments
            for _ in range(3):
                a = self.rng.randint(0, n)
                b = self.rng.randint(0, n)
                if a == b:
                    continue
                if a < b:
                    frag = tour[a:b+1]
                else:
                    frag = np.concatenate([tour[a:], tour[:b+1]])
                self.update_archive(archive, frag, cost)

        critics = self.get_critics()
        critic_weights = np.ones(len(critics), dtype=float)
        critic_success = np.zeros(len(critics), dtype=int)
        # acceptance parameters
        diversity_bias = 0.3  # weight on novelty
        restructure_period = max(10, max_iters // 10)
        iter_count = 0

        # main loop
        while iter_count < max_iters and (time.time() - start_time) < time_limit:
            iter_count += 1
            # age increment
            for member in ensemble:
                member['age'] += 1

            # consensus sketch
            P = self.consensus_sketch(ensemble)

            # create hybrid proposals
            proposals = []
            num_hybrids = min(len(ensemble), max(2, ensemble_size // 2))
            for _ in range(num_hybrids):
                hybrid = self.build_hybrid_from_sketch(P, archive)
                proposals.append(hybrid)

            # for each hybrid, apply a committee of critics (sample critics probabilistically)
            for hybrid in proposals:
                # initial candidate
                base_tour = hybrid
                base_cost = self.tour_length(base_tour)
                candidate_pool = []
                # apply committee: pick a small committee of critics by weight
                probs = critic_weights / critic_weights.sum()
                committee_indices = self.rng.choice(len(critics), size=min(3, len(critics)), replace=False, p=probs)
                for ci in committee_indices:
                    critic = critics[ci]
                    try:
                        new_tour, new_cost = critic(base_tour.copy())
                    except Exception:
                        continue
                    candidate_pool.append((ci, new_tour, new_cost))
                # also consider applying pairwise critic compositions occasionally
                if self.rng.rand() < 0.2 and len(candidate_pool) >= 2:
                    # take best two and combine with small perturbation
                    candidate_pool.sort(key=lambda x: x[2])
                    _, t1, _ = candidate_pool[0]
                    _, t2, _ = candidate_pool[1]
                    # splice t2 fragment into t1
                    a = self.rng.randint(0, n)
                    b = self.rng.randint(0, n)
                    if a != b:
                        if a < b:
                            frag = t2[a:b+1]
                        else:
                            frag = np.concatenate([t2[a:], t2[:b+1]])
                        # insert frag into t1
                        remaining = [x for x in t1 if x not in frag]
                        insert_pos = self.rng.randint(0, len(remaining)+1)
                        new = np.array(remaining[:insert_pos] + list(frag) + remaining[insert_pos:], dtype=int)
                        new_cost = self.tour_length(new)
                        candidate_pool.append((None, new, new_cost))

                # evaluate and accept according to acceptance rule
                # acceptance favoring improvement and novelty
                for ci, cand_tour, cand_cost in candidate_pool:
                    # measure novelty: average edge overlap with ensemble
                    # compute fraction of edges shared with best ensemble member
                    best_member = ensemble[0]['tour']
                    # edge set for best member
                    best_edges = set()
                    for i in range(n):
                        a = best_member[i]
                        b = best_member[(i+1) % n]
                        best_edges.add((a,b))
                    cand_edges = 0
                    for i in range(n):
                        a = cand_tour[i]
                        b = cand_tour[(i+1) % n]
                        if (a,b) in best_edges:
                            cand_edges += 1
                    overlap = cand_edges / n
                    novelty = 1.0 - overlap
                    # compute acceptance score
                    improvement = ensemble[-1]['cost'] - cand_cost  # relative to worst
                    # we define objective score
                    score = 0.7 * (improvement) + diversity_bias * novelty * (ensemble[-1]['cost'] * 0.1)
                    # accept if score positive or probabilistic based on small improvements
                    if score > 1e-9 or (self.rng.rand() < 0.01 and cand_cost < ensemble[0]['cost'] * 1.05):
                        # replace worst or a random high-age member
                        # find candidate to replace: worst or oldest among worse ones
                        worst_idx = max(range(len(ensemble)), key=lambda i: ensemble[i]['cost'])
                        # ensure we do not always replace best
                        if cand_cost < ensemble[worst_idx]['cost'] or self.rng.rand() < 0.5:
                            ensemble[worst_idx] = {'tour': cand_tour.copy(), 'cost': cand_cost, 'age': 0, 'score': -cand_cost}
                            # update archive with fragments from candidate
                            # extract a few fragments
                            for _ in range(3):
                                a = self.rng.randint(0, n)
                                b = self.rng.randint(0, n)
                                if a == b:
                                    continue
                                if a < b:
                                    frag = cand_tour[a:b+1]
                                else:
                                    frag = np.concatenate([cand_tour[a:], cand_tour[:b+1]])
                                self.update_archive(archive, frag, cand_cost)
                            # update critic success
                            if ci is not None:
                                critic_success[ci] += 1
                                critic_weights[ci] = 1.0 + critic_success[ci]
                            # sort ensemble
                            ensemble.sort(key=lambda x: x['cost'])
            # attenuate ineffective critics periodically
            if iter_count % 10 == 0:
                # reduce weight of critics that rarely succeed
                total_success = critic_success.sum() + 1e-6
                for i in range(len(critic_weights)):
                    # proportional to recent success
                    critic_weights[i] = 1.0 + critic_success[i] / (1.0 + total_success)
                # small random jitter
                critic_weights += 1e-3 * self.rng.rand(len(critic_weights))

            # occasional restructuring to escape entrenched patterns
            if iter_count % restructure_period == 0 and iter_count > 0:
                # replace a few worst members by recombinations seeded from top and archive
                num_replace = max(1, ensemble_size // 8)
                for _ in range(num_replace):
                    worst_idx = max(range(len(ensemble)), key=lambda i: ensemble[i]['cost'])
                    # build recombination from best and an archive fragment
                    top = ensemble[0]['tour']
                    if archive and self.rng.rand() < 0.8:
                        frag, _ = archive[self.rng.randint(0, len(archive))]
                        # splice frag into top
                        remaining = [x for x in top if x not in frag]
                        insert_pos = self.rng.randint(0, len(remaining)+1)
                        new = np.array(remaining[:insert_pos] + list(frag) + remaining[insert_pos:], dtype=int)
                    else:
                        # blend top and a random ensemble member
                        partner = ensemble[self.rng.randint(0, len(ensemble))]['tour']
                        cut = self.rng.randint(1, n-1)
                        part = partner[:cut]
                        remaining = [x for x in top if x not in part]
                        new = np.array(list(part) + remaining, dtype=int)
                    new, new_cost = self.two_opt_local_search(new, max_iters=10)
                    ensemble[worst_idx] = {'tour': new, 'cost': new_cost, 'age': 0, 'score': -new_cost}
                ensemble.sort(key=lambda x: x['cost'])

        # Post-processing: concentrated critic cascade on best candidate
        best = ensemble[0]
        best_tour = best['tour']
        best_cost = best['cost']
        # apply multiple intensification passes
        for _ in range(3):
            best_tour, best_cost = self.two_opt_local_search(best_tour, max_iters=50)
            # attempt a few reinsertion and swap improvements
            for _ in range(10):
                new, new_cost = self.reinsertion_operator(best_tour)
                if new_cost + 1e-12 < best_cost:
                    best_tour, best_cost = new, new_cost
            # occasional double-bridge + local search to try escape shallow local minima
            new, new_cost = self.double_bridge(best_tour)
            new, new_cost = self.two_opt_local_search(new, max_iters=50)
            if new_cost + 1e-12 < best_cost:
                best_tour, best_cost = new, new_cost

        # final feasibility pass (identity for TSP) and return
        return best_tour.copy()
            

