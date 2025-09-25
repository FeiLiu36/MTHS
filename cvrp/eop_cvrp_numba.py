import numpy as np
import random
import math
import time
from copy import deepcopy
from numba import njit, types
from numba.typed import List

# ----------------- Numba JIT-compiled helper functions -----------------
# These are defined outside the class as Numba doesn't directly support JITing 
# class methods with `self` in nopython mode.

@njit
def _numba_total_cost(routes: types.List(types.List(types.int64)), distance_matrix: np.ndarray) -> float:
    """Numba-accelerated: Compute total travel cost of a solution."""
    cost = 0.0
    for r in routes:
        if len(r) < 2:
            continue
        for i in range(len(r) - 1):
            cost += distance_matrix[r[i], r[i + 1]]
    return cost

@njit
def _numba_route_demand(route: types.List(types.int64), demands: np.ndarray) -> int:
    """Numba-accelerated: Sum of demands of a route excluding depots (0)."""
    demand = 0
    for n in route:
        if n != 0:
            demand += demands[n]
    return demand

@njit
def _numba_calculate_savings(n: int, distance_matrix: np.ndarray) -> list:
    """Numba-accelerated: Calculate savings for Clarke-Wright heuristic."""
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((s, i, j))
    return savings

@njit
def _numba_kmeans_loop(pts: np.ndarray, centroids: np.ndarray, max_iter: int) -> np.ndarray:
    """Numba-accelerated: Core K-Means clustering loop."""
    m = pts.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(m, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.empty((m, k), dtype=np.float64)
        for i in range(m):
            for j in range(k):
                dists[i, j] = np.linalg.norm(pts[i] - centroids[j])

        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = pts[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean()
            else:
                centroids[j] = pts[np.random.randint(0, m)]
    return labels

@njit
def _numba_two_opt_route(route: types.List(types.int64), distance_matrix: np.ndarray) -> types.List(types.int64):
    """Numba-accelerated: Apply 2-opt intra-route."""
    n = len(route)
    if n <= 4:
        return route
    
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                if delta < -1e-9:
                    # Numba requires careful list construction
                    new_route_list = best[:i]
                    segment_to_reverse = best[i:j + 1]
                    for k in range(len(segment_to_reverse)):
                        new_route_list.append(segment_to_reverse[len(segment_to_reverse) - 1 - k])
                    
                    new_route_list.extend(best[j + 1:])
                    best = new_route_list
                    improved = True
                    break
            if improved:
                break
    return best

@njit
def _numba_find_best_relocate_move(routes: types.List(types.List(types.int64)), demands: np.ndarray, vehicle_capacity: int, distance_matrix: np.ndarray):
    """Numba-accelerated: Find the best single-customer relocate move."""
    best_delta = 0.0
    best_move = (-1, -1, -1, -1)  # (src_idx, pos_in_src, dst_idx, insert_pos)
    
    for i in range(len(routes)):
        src = routes[i]
        if len(src) <= 2: continue
        for pos in range(1, len(src) - 1):
            v = src[pos]
            a = src[pos - 1]
            b = src[pos + 1]
            remove_cost = distance_matrix[a, b] - (distance_matrix[a, v] + distance_matrix[v, b])

            for j in range(len(routes)):
                if i == j: continue
                
                dst = routes[j]
                
                dst_demand = 0
                for node in dst:
                    if node != 0:
                        dst_demand += demands[node]
                
                if dst_demand + demands[v] > vehicle_capacity:
                    continue

                for ins in range(1, len(dst)):
                    p = dst[ins - 1]
                    q = dst[ins]
                    insert_cost = distance_matrix[p, v] + distance_matrix[v, q] - distance_matrix[p, q]
                    delta = remove_cost + insert_cost
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (i, pos, j, ins)
    return best_move

@njit
def _numba_find_best_insertion(v: int, routes: types.List(types.List(types.int64)), demands: np.ndarray, vehicle_capacity: int, distance_matrix: np.ndarray):
    """Numba-accelerated: Find the best greedy insertion for a single customer."""
    best_increase = np.inf
    best_r_idx = -1
    best_pos = -1

    for idx in range(len(routes)):
        r = routes[idx]
        
        cap = 0
        for node in r:
            if node != 0:
                cap += demands[node]
        
        if cap + demands[v] > vehicle_capacity:
            continue
        
        for pos in range(1, len(r)):
            a = r[pos - 1]
            b = r[pos]
            inc = distance_matrix[a, v] + distance_matrix[v, b] - distance_matrix[a, b]
            if inc < best_increase:
                best_increase = inc
                best_r_idx = idx
                best_pos = pos
                
    return best_increase, best_r_idx, best_pos

@njit
def _numba_nearest_neighbor(n: int, demands: np.ndarray, vehicle_capacity: int, distance_matrix: np.ndarray):
    """Numba-accelerated: Nearest neighbor heuristic."""
    unvisited_mask = np.ones(n, dtype=np.bool_)
    unvisited_mask[0] = False
    unvisited_count = n - 1
    
    routes = List()
    
    while unvisited_count > 0:
        cur = 0
        # Numba requires explicit list creation
        route = List()
        route.append(0)
        load = 0
        
        while True:
            best_v = -1
            best_dist = np.inf
            
            for v_idx in range(1, n):
                if unvisited_mask[v_idx]:
                    if load + demands[v_idx] <= vehicle_capacity:
                        d = distance_matrix[cur, v_idx]
                        if d < best_dist:
                            best_dist = d
                            best_v = v_idx
                            
            if best_v == -1:
                break
                
            route.append(best_v)
            load += demands[best_v]
            unvisited_mask[best_v] = False
            unvisited_count -= 1
            cur = best_v
            
        route.append(0)
        routes.append(route)
        
    return routes


class CVRPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: list, vehicle_capacity: int):
        """
        Initialize the CVRP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each node, including the depot.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between nodes.
            demands: List of integers representing the demand of each node (first node is typically the depot with zero demand).
            vehicle_capacity: Integer representing the maximum capacity of each vehicle.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = np.array(demands) # Numba works best with NumPy arrays
        self.vehicle_capacity = vehicle_capacity
        self.n = len(coordinates)
        # random seed for reproducibility-ish
        random.seed(42)
        np.random.seed(42)

    # ----------------- Utility functions -----------------
    def total_cost(self, routes):
        """Compute total travel cost of a solution given as list of routes (each route = list of nodes including depots at ends)."""
        typed_routes = List()
        for r in routes:
            typed_routes.append(List(r))
        return _numba_total_cost(typed_routes, self.distance_matrix)

    def route_demand(self, route):
        """Sum of demands of a route excluding depots (0)."""
        return _numba_route_demand(List(route), self.demands)

    def flatten_solution(self, routes):
        """Flatten list of routes into required output format [0,...,0]."""
        flat = []
        for r in routes:
            if len(r) == 0:
                continue
            if r[0] != 0:
                r = [0] + r
            if r[-1] != 0:
                r = r + [0]
            flat.extend(r)
        if len(flat) == 0 or flat[0] != 0:
            flat = [0] + flat
        if flat[-1] != 0:
            flat.append(0)
        return flat

    def parse_flat(self, flat):
        """Convert flat list to list of routes."""
        routes = []
        cur = []
        for v in flat:
            if v == 0:
                if cur:
                    routes.append([0] + cur + [0])
                    cur = []
                else:
                    # consecutive depots or leading depot -> skip
                    continue
            else:
                cur.append(v)
        if cur:
            routes.append([0] + cur + [0])
        return routes

    def ensure_feasible_routes(self, routes):
        """Ensure all customers visited once and capacity respected. Repair if necessary."""
        n = self.n
        visited = set()
        for r in routes:
            for v in r:
                if v != 0:
                    visited.add(v)
        missing = [i for i in range(1, n) if i not in visited]
        # remove duplicates if any
        counts = {}
        for r in routes:
            to_remove = []
            for v in r:
                if v == 0:
                    continue
                counts[v] = counts.get(v, 0) + 1
                if counts[v] > 1:
                    to_remove.append(v)
            for v in to_remove:
                r.remove(v)
        # insert missing greedily
        if missing:
            typed_routes = List()
            for r_py in routes:
                typed_routes.append(List(r_py))

            for v in missing:
                _, best_route_idx, best_pos = _numba_find_best_insertion(v, typed_routes, self.demands, self.vehicle_capacity, self.distance_matrix)
                
                if best_route_idx != -1:
                    routes[best_route_idx].insert(best_pos, v)
                    typed_routes[best_route_idx].insert(best_pos, v)
                else:
                    new_route = [0, v, 0]
                    routes.append(new_route)
                    typed_routes.append(List(new_route))

        # split any overloaded routes
        new_routes = []
        for r in routes:
            cur = [0]
            cur_load = 0
            for v in r[1:-1]:
                if cur_load + self.demands[v] > self.vehicle_capacity:
                    cur.append(0)
                    if len(cur) > 2:
                        new_routes.append(cur)
                    cur = [0]
                    cur_load = 0
                cur.append(v)
                cur_load += self.demands[v]
            cur.append(0)
            if len(cur) > 2:
                new_routes.append(cur)
        return new_routes

    # ----------------- Initialization heuristics -----------------
    def nearest_neighbor_solution(self):
        numba_routes = _numba_nearest_neighbor(self.n, self.demands, self.vehicle_capacity, self.distance_matrix)
        routes = [list(r) for r in numba_routes]
        return routes

    def sweep_solution(self):
        # sort customers by polar angle around depot and pack into routes
        depot = self.coordinates[0]
        angles = []
        for i in range(1, self.n):
            dx, dy = self.coordinates[i] - depot
            angle = math.atan2(dy, dx)
            angles.append((angle, i))
        angles.sort()
        routes = []
        cur = [0]
        load = 0
        for _, v in angles:
            if load + self.demands[v] > self.vehicle_capacity:
                cur.append(0)
                routes.append(cur)
                cur = [0]
                load = 0
            cur.append(v)
            load += self.demands[v]
        if len(cur) > 1:
            cur.append(0)
            routes.append(cur)
        return routes

    def clarke_wright(self):
        # Basic Clarke-Wright savings algorithm
        n = self.n
        savings = _numba_calculate_savings(n, self.distance_matrix)
        savings.sort(key=lambda x: x[0], reverse=True)
        # initially each customer in own route
        routes = {i: [0, i, 0] for i in range(1, n)}
        route_of = {i: i for i in range(1, n)}
        loads = {i: self.demands[i] for i in range(1, n)}
        for s, i, j in savings:
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            # check if i at end of its route and j at start (or vice versa)
            if route_i[-2] == i and route_j[1] == j:
                if loads[ri] + loads[rj] <= self.vehicle_capacity:
                    # merge
                    new_route = route_i[:-1] + route_j[1:]
                else:
                    continue
            elif route_j[-2] == j and route_i[1] == i:
                if loads[ri] + loads[rj] <= self.vehicle_capacity:
                    new_route = route_j[:-1] + route_i[1:]
                else:
                    continue
            else:
                continue
            # assign new id
            new_id = ri
            routes[new_id] = new_route
            loads[new_id] = loads[ri] + loads[rj]
            # update mapping
            for v in new_route:
                if v != 0:
                    route_of[v] = new_id
            # remove old
            if rj in routes and rj != new_id:
                del routes[rj]
                del loads[rj]
        return list(routes.values())

    # ----------------- Clustering / hierarchical decomposition -----------------
    def kmeans_clusters(self, k, max_iter=50):
        # simple kmeans on coordinates excluding depot
        pts = self.coordinates[1:]
        m = pts.shape[0]
        if k <= 0:
            return {0: list(range(1, self.n))}
        k = min(k, m)
        # initialize centroids randomly
        idx = np.random.choice(m, k, replace=False)
        centroids = pts[idx].copy()
        labels = _numba_kmeans_loop(pts, centroids, max_iter)
        clusters = {}
        for i, lab in enumerate(labels, start=1):
            clusters.setdefault(lab, []).append(i)
        return clusters

    def build_hierarchy(self):
        n = self.n
        scales = []
        # coarse
        coarse_k = max(2, int(math.sqrt(max(2, n - 1))))
        mid_k = max(2, int(max(2, (n - 1) / 5)))
        fine_k = max(2, int((n - 1) // 3))
        ks = sorted(list({coarse_k, mid_k, fine_k}))
        hierarchy = []
        for k in ks:
            clusters = self.kmeans_clusters(k)
            # convert clusters values to list of customer ids
            groups = [members for members in clusters.values() if len(members) > 0]
            hierarchy.append(groups)
        return hierarchy  # list of clusterings at different scales

    # ----------------- Local search operators -----------------
    def two_opt_route(self, route):
        # route includes depots at ends [0, ... ,0]. Apply 2-opt intra-route.
        numba_route = List(route)
        optimized_route = _numba_two_opt_route(numba_route, self.distance_matrix)
        return list(optimized_route)

    def intra_route_opt(self, routes, time_budget=0.0):
        # apply 2-opt to each route
        new_routes = []
        for r in routes:
            nr = self.two_opt_route(r)
            new_routes.append(nr)
        return new_routes

    def relocate_between_routes(self, routes, max_moves=50):
        # single-customer relocate to improve cost
        moves = 0
        improved = True
        routes = deepcopy(routes)
        while improved and moves < max_moves:
            improved = False
            moves += 1
            
            typed_routes = List()
            for r_py in routes:
                typed_routes.append(List(r_py))
            
            best_move = _numba_find_best_relocate_move(typed_routes, self.demands, self.vehicle_capacity, self.distance_matrix)

            if best_move[0] != -1:
                i, pos, j, ins = best_move
                v = routes[i].pop(pos)
                
                if len(routes[i]) <= 2:
                    routes.pop(i)
                    if i < j:
                        j -= 1

                routes[j].insert(ins, v)
                improved = True
        return routes

    # ----------------- Patch-and-merge (ruin-and-recreate & recombination) -----------------
    def ruin_and_recreate(self, base_routes, customers_subset, ruin_fraction=0.5):
        # Remove a subset of customers from base_routes (prefer customers in customers_subset)
        routes = deepcopy(base_routes)
        all_customers = [v for r in routes for v in r if v != 0]
        target_customers = [v for v in all_customers if v in customers_subset]
        k = max(1, int(len(target_customers) * ruin_fraction))
        if k < 1:
            return routes
        remove_set = set(random.sample(target_customers, k))
        # also sometimes remove a few outside customers for diversification
        extra = max(0, int(k * 0.1))
        others = [v for v in all_customers if v not in remove_set and v not in customers_subset]
        if others and extra > 0:
            take = min(extra, len(others))
            remove_set.update(random.sample(others, take))
        # remove them
        for r in routes:
            r[:] = [v for v in r if v not in remove_set]
            if len(r) == 0:
                r.extend([0, 0])
        # repair: greedy cheapest insertion for removed customers
        removed = list(remove_set)
        random.shuffle(removed)

        typed_routes = List()
        for r_py in routes:
            typed_routes.append(List(r_py))

        for v in removed:
            _, best_r_idx, best_pos = _numba_find_best_insertion(v, typed_routes, self.demands, self.vehicle_capacity, self.distance_matrix)
            
            if best_r_idx != -1:
                routes[best_r_idx].insert(best_pos, v)
                typed_routes[best_r_idx].insert(best_pos, v)
            else:
                new_route = [0, v, 0]
                routes.append(new_route)
                typed_routes.append(List(new_route))

        # cleanup empty dummy zeros
        new_routes = []
        for r in routes:
            if len(r) <= 2:
                continue
            if r[0] != 0: r = [0] + r
            if r[-1] != 0: r = r + [0]
            new_routes.append(r)
        return new_routes

    def recombine_with_partner(self, base_routes, partner_routes, customers_subset):
        # for customers in customers_subset, prefer partner's routing arrangement
        routes = deepcopy(base_routes)
        # collect partner routes that cover customers_subset heavily
        partner_selected = []
        for r in partner_routes:
            inter = [v for v in r if v != 0 and v in customers_subset]
            if inter:
                partner_selected.append((len(inter), r))
        partner_selected.sort(reverse=True)
        # remove customers in partner_selected from base and then insert partner's entire subroutes where feasible
        used = set()
        for _, pr in partner_selected:
            sub = [v for v in pr if v != 0]
            if sum(self.demands[v] for v in sub) > self.vehicle_capacity:
                continue
            for r in routes:
                r[:] = [v for v in r if v not in sub]
            routes.append([0] + sub + [0])
            used.update(sub)
        # insert remaining customers left (not used) using cheapest insertion
        remaining_set = set(v for r in routes for v in r if v != 0)
        all_customers = set(range(1, self.n))
        missing = list(all_customers - remaining_set)
        # repair missing
        if missing:
            typed_routes = List()
            for r_py in routes:
                typed_routes.append(List(r_py))

            for v in missing:
                _, best_r_idx, best_pos = _numba_find_best_insertion(v, typed_routes, self.demands, self.vehicle_capacity, self.distance_matrix)
                if best_r_idx != -1:
                    routes[best_r_idx].insert(best_pos, v)
                    typed_routes[best_r_idx].insert(best_pos, v)
                else:
                    new_route = [0, v, 0]
                    routes.append(new_route)
                    typed_routes.append(List(new_route))
        # cleanup
        new_routes = []
        for r in routes:
            r = [v for v in r if v != 0 or v == 0]
            if len(r) <= 2: continue
            if r[0] != 0: r = [0] + r
            if r[-1] != 0: r = r + [0]
            new_routes.append(r)
        return new_routes

    # ----------------- Main solve function -----------------
    def solve(self) -> list:
        n = self.n
        start_time = time.time()
        # parameters
        ensemble_size = min(10, max(4, n // 5))
        elite_size = min(5, ensemble_size)
        max_iters = max(80, 20 + n * 2)  # adaptive iterations
        agents = min(6, max(2, ensemble_size // 2))
        # initialize hierarchy
        hierarchy = self.build_hierarchy()
        # initialize ensemble
        ensemble = []
        # seed with multiple heuristics
        try:
            s1 = self.nearest_neighbor_solution()
            ensemble.append(s1)
        except Exception:
            pass
        try:
            s2 = self.clarke_wright()
            ensemble.append(s2)
        except Exception:
            pass
        try:
            s3 = self.sweep_solution()
            ensemble.append(s3)
        except Exception:
            pass
        # fill rest with random perturbations of nearest neighbor
        while len(ensemble) < ensemble_size:
            base = deepcopy(random.choice(ensemble))
            # random ruin and recreate small
            customers = [i for i in range(1, n)]
            subset = random.sample(customers, max(1, len(customers) // 6))
            sol = self.ruin_and_recreate(base, subset, ruin_fraction=0.6)
            ensemble.append(sol)
        # compute qualities
        def sol_cost(sol):
            return self.total_cost(sol)
        ensemble = [self.ensure_feasible_routes(e) for e in ensemble]
        ensemble_costs = [sol_cost(e) for e in ensemble]
        # elite memory
        elites = sorted(zip(ensemble_costs, ensemble), key=lambda x: x[0])[:elite_size]
        elite_solutions = [deepcopy(x[1]) for x in elites]
        elite_costs = [x[0] for x in elites]
        # operator statistics
        op_success = {"ruin_recreate": 1, "recombine": 1, "relocate": 1, "two_opt": 1}
        op_attempts = {"ruin_recreate": 1, "recombine": 1, "relocate": 1, "two_opt": 1}
        ruin_fraction = 0.4

        best_solution = deepcopy(elite_solutions[0])
        best_cost = elite_costs[0]

        # main loop
        for it in range(max_iters):
            # early termination if time too long
            if time.time() - start_time > 15 + n * 0.02:
                break
            # assign agents to cluster scales and regions
            for a in range(agents):
                # select clustering scale probabilistically (favor mid scales)
                scale_idx = random.randrange(len(hierarchy))
                clusters = hierarchy[scale_idx]
                # choose a region cluster
                cluster = random.choice(clusters)
                customers_subset = set(cluster)
                # select base solution by quality/diversity: weighted by inverse cost rank
                costs = [self.total_cost(s) for s in ensemble]
                ranks = np.argsort(costs)
                probs = np.zeros(len(ensemble))
                for idx, rnk in enumerate(ranks):
                    probs[rnk] = 1.0 / (1 + idx)
                probs = probs / probs.sum()
                base_idx = np.random.choice(len(ensemble), p=probs)
                base = deepcopy(ensemble[base_idx])
                # select operator based on success history
                weights = np.array([op_success[k] / op_attempts[k] for k in op_success.keys()]) + 1e-6
                keys = list(op_success.keys())
                weights = weights / weights.sum()
                chosen_op = np.random.choice(keys, p=weights)
                candidate = None
                pre_cost = self.total_cost(base)
                # apply operator
                if chosen_op == "ruin_recreate":
                    op_attempts["ruin_recreate"] += 1
                    candidate = self.ruin_and_recreate(base, customers_subset, ruin_fraction=ruin_fraction)
                elif chosen_op == "recombine":
                    op_attempts["recombine"] += 1
                    partner = deepcopy(random.choice(ensemble))
                    candidate = self.recombine_with_partner(base, partner, customers_subset)
                else:
                    # default to local improvements
                    op_attempts[chosen_op] += 1
                    candidate = deepcopy(base)
                # local refinements: two-stage evaluator
                # fast filtering: apply quick local moves
                candidate = self.relocate_between_routes(candidate, max_moves=30)
                candidate = self.intra_route_opt(candidate)
                # high-fidelity selective validation: apply heavier 2-opt if promising
                cand_cost = self.total_cost(candidate)
                # acceptance: if candidate improves over worst in ensemble or is elite-worthy
                worst_idx = int(np.argmax([self.total_cost(s) for s in ensemble]))
                worst_cost = self.total_cost(ensemble[worst_idx])
                accepted = False
                if cand_cost + 1e-9 < worst_cost:
                    # replace worst
                    ensemble[worst_idx] = deepcopy(candidate)
                    accepted = True
                else:
                    # sometimes accept if close and increases diversity
                    best_idx = int(np.argmin([self.total_cost(s) for s in ensemble]))
                    bestc = self.total_cost(ensemble[best_idx])
                    if cand_cost < bestc * 1.02:
                        # try to inject if diverse
                        # simple diversity: measure overlap with existing solutions
                        cand_set = set(v for r in candidate for v in r if v != 0)
                        avg_ov = 0.0
                        for s in ensemble:
                            s_set = set(v for r in s for v in r if v != 0)
                            avg_ov += len(cand_set & s_set) / max(1, len(cand_set | s_set))
                        avg_ov /= len(ensemble)
                        if avg_ov < 0.95 or random.random() < 0.05:
                            # replace a random worse solution
                            idxs = [i for i in range(len(ensemble)) if self.total_cost(ensemble[i]) >= cand_cost]
                            if idxs:
                                rep = random.choice(idxs)
                                ensemble[rep] = deepcopy(candidate)
                                accepted = True
                # update operator stats
                if accepted and cand_cost + 1e-9 < pre_cost:
                    op_success[chosen_op] = op_success.get(chosen_op, 1) + 1
                # update elites
                all_costs = [self.total_cost(s) for s in ensemble]
                combined = list(zip(all_costs, ensemble))
                combined.sort(key=lambda x: x[0])
                elites_new = [deepcopy(x[1]) for x in combined[:elite_size]]
                elites_costs = [x[0] for x in combined[:elite_size]]
                elite_solutions = elites_new
                elite_costs = elites_costs
                if elite_costs[0] < best_cost - 1e-9:
                    best_cost = elite_costs[0]
                    best_solution = deepcopy(elite_solutions[0])
                # adapt ruin_fraction based on success rate
                succ_rate = sum(op_success[k] for k in op_success) / sum(op_attempts[k] for k in op_attempts)
                # nudge parameters
                if succ_rate < 0.2:
                    ruin_fraction = min(0.9, ruin_fraction + 0.02)
                elif succ_rate > 0.6:
                    ruin_fraction = max(0.1, ruin_fraction - 0.02)
            # end agents loop

        # post-processing: enforce feasibility and final smoothing
        final = self.ensure_feasible_routes(best_solution)
        final = self.intra_route_opt(final)
        final = self.relocate_between_routes(final, max_moves=200)
        final = self.ensure_feasible_routes(final)
        flat = self.flatten_solution(final)
        # final sanity: ensure every customer once
        visited = [v for v in flat if v != 0]
        missing = [i for i in range(1, n) if i not in visited]
        if missing:
            # add missing each as its own short route
            for v in missing:
                flat.extend([0, v, 0])
        # compress repeated zeros
        out = []
        prev = None
        for v in flat:
            if v == 0 and prev == 0:
                continue
            out.append(v)
            prev = v
        if out[0] != 0:
            out = [0] + out
        if out[-1] != 0:
            out.append(0)
        return out