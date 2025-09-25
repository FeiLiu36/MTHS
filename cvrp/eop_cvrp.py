# our updated program here
import numpy as np
import random
import math
from copy import deepcopy
from numba import njit

@njit
def two_opt_route(route, distance_matrix):
    # apply 2-opt on interior nodes (route includes depot endpoints)
    if len(route) <= 4:
        return route  # nothing to improve
    improved = True
    dm = distance_matrix
    best_route = route
    while improved:
        improved = False
        n = len(best_route)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # reverse segment i..j
                a = best_route[i - 1]
                b = best_route[i]
                c = best_route[j]
                d = best_route[j + 1]
                delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if delta < -1e-9:
                    best_route[i:j + 1] = best_route[j:i - 1:-1]
                    #new_route = best_route[:i] + list(reversed(best_route[i:j + 1])) + best_route[j + 1:]
                    #best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route

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
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.n = len(coordinates)

    # Helper functions
    def route_cost(self, route):
        # route is list of node indices including depot at start and end
        if len(route) < 2:
            return 0.0
        d = 0.0
        dm = self.distance_matrix
        for i in range(len(route)-1):
            d += dm[route[i], route[i+1]]
        return d

    def route_load(self, route):
        # sum demands excluding depot
        return sum(self.demands[i] for i in route if i != 0)

    def best_insertion_cost(self, cust, route):
        # returns (best_cost_increase, best_pos)
        # route includes depot at ends
        best = float('inf')
        best_pos = None
        dm = self.distance_matrix
        if len(route) == 2:
            # only depot -> depot, inserting between 0 and 0
            inc = dm[route[0], cust] + dm[cust, route[-1]] - dm[route[0], route[-1]]
            return inc, 1
        for pos in range(1, len(route)):
            a = route[pos-1]
            b = route[pos]
            inc = dm[a, cust] + dm[cust, b] - dm[a, b]
            if inc < best:
                best = inc
                best_pos = pos
        return best, best_pos

    def removal_cost(self, cust, route):
        # cost change when removing cust from route (positive means cost decreases)
        dm = self.distance_matrix
        # find positions of cust in route (should be one)
        try:
            idx = route.index(cust)
        except ValueError:
            return None
        a = route[idx-1]
        b = route[idx+1]
        removed = dm[a, cust] + dm[cust, b] - dm[a, b]
        return removed  # positive removed means cost saved
    #
    # def two_opt_route(self, route):
    #     # apply 2-opt on interior nodes (route includes depot endpoints)
    #     if len(route) <= 4:
    #         return route  # nothing to improve
    #     improved = True
    #     dm = self.distance_matrix
    #     best_route = route
    #     while improved:
    #         improved = False
    #         n = len(best_route)
    #         for i in range(1, n-2):
    #             for j in range(i+1, n-1):
    #                 # reverse segment i..j
    #                 a = best_route[i-1]
    #                 b = best_route[i]
    #                 c = best_route[j]
    #                 d = best_route[j+1]
    #                 delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
    #                 if delta < -1e-9:
    #                     new_route = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
    #                     best_route = new_route
    #                     improved = True
    #                     break
    #             if improved:
    #                 break
    #     return best_route

    def flatten_routes(self, routes):
        flat = []
        for r in routes:
            if len(r) == 0:
                continue
            if flat and flat[-1] == 0 and r[0] == 0:
                # avoid duplicate depot
                flat.extend(r[1:])
            else:
                flat.extend(r)
        # ensure starts and ends with 0
        if not flat or flat[0] != 0:
            flat = [0] + flat
        if flat[-1] != 0:
            flat.append(0)
        return flat

    def initial_clarke_wright(self):
        # Basic Clarke-Wright savings to generate initial feasible routes
        n = self.n
        dm = self.distance_matrix
        demands = self.demands
        capacity = self.vehicle_capacity

        # start with one route per customer: [0, i, 0]
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])

        loads = [demands[i] for i in range(1, n)]
        # savings list
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = dm[0, i] + dm[0, j] - dm[i, j]
                savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])

        # helper maps: which route index contains customer
        cust_route = {i: idx for idx, i in enumerate(range(1, n))}
        for s, i, j in savings:
            if i not in cust_route or j not in cust_route:
                continue
            ri = cust_route[i]
            rj = cust_route[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            load_i = self.route_load(route_i)
            load_j = self.route_load(route_j)
            if load_i + load_j > capacity:
                continue
            # Only merge if i is at an end of its route and j at an end (classic CW)
            if route_i[1] == i and route_j[-2] == j:
                # j ... 0   0 ... i  -> merge route_j + route_i (without duplicate depots)
                new_route = route_j[:-1] + route_i[1:]
            elif route_i[-2] == i and route_j[1] == j:
                # i ... 0   0 ... j -> merge route_i + route_j
                new_route = route_i[:-1] + route_j[1:]
            elif route_i[1] == i and route_j[1] == j:
                # reverse route_j and merge
                rev = [0] + list(reversed(route_j[1:-1])) + [0]
                new_route = rev[:-1] + route_i[1:]
            elif route_i[-2] == i and route_j[-2] == j:
                rev = [0] + list(reversed(route_i[1:-1])) + [0]
                new_route = route_j[:-1] + rev[1:]
            else:
                continue
            # perform merge
            # mark removed routes
            new_idx = min(ri, rj)
            keep_idx = max(ri, rj)
            routes[new_idx] = new_route
            routes.pop(keep_idx)
            # update cust_route
            cust_route.clear()
            for idx, route in enumerate(routes):
                for c in route:
                    if c != 0:
                        cust_route[c] = idx

        # final cleaning: ensure capacity constraint satisfied (split if needed)
        final_routes = []
        for route in routes:
            load = self.route_load(route)
            if load <= capacity:
                final_routes.append(route)
            else:
                # split greedily: fill until capacity then start new route, preserving order except depot duplicates
                cur = [0]
                cur_load = 0
                for node in route[1:-1]:
                    d = self.demands[node]
                    if cur_load + d > capacity:
                        cur.append(0)
                        final_routes.append(cur)
                        cur = [0]
                        cur_load = 0
                    cur.append(node)
                    cur_load += d
                cur.append(0)
                final_routes.append(cur)
        return final_routes

    def total_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    def solve(self) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP).

        Returns:
            A one-dimensional list of integers representing the sequence of nodes visited by all vehicles.
        """
        import random
        random.seed(0)

        n = self.n
        capacity = self.vehicle_capacity

        # helper to ensure route format [0, ..., 0]
        def normalize_route(r):
            if not r:
                return [0, 0]
            if r[0] != 0:
                r = [0] + r
            if r[-1] != 0:
                r = r + [0]
            return r

        # Initialization Phase
        routes = self.initial_clarke_wright() or []
        # Normalize and ensure proper depot endpoints
        routes = [normalize_route(r) for r in routes]

        # ensure all customers present: add missing as single-customer routes
        assigned = set()
        for r in routes:
            for c in r:
                if c != 0:
                    assigned.add(c)
        for i in range(1, n):
            if i not in assigned:
                routes.append([0, i, 0])

        # initialize route signals (market penalties)
        route_signals = [1.0 for _ in routes]

        best_routes = deepcopy(routes)
        best_cost = self.total_cost(routes)
        no_improve = 0
        max_iters = max(20000, 20 * n)
        stagnation_limit = 5000

        for it in range(max_iters):
            improved = False

            # Build maps and caches
            cust_to_route = {}
            for idx, r in enumerate(routes):
                for c in r:
                    if c != 0:
                        cust_to_route[c] = idx
            route_loads = [self.route_load(r) for r in routes]

            # Precompute removal costs for customers (if possible)
            removal_cache = {}
            for cust in range(1, n):
                if cust in cust_to_route:
                    frm_idx = cust_to_route[cust]
                    removal_cache[cust] = self.removal_cost(cust, routes[frm_idx])

            # Customers advertise marginal relocation gains (offers)
            offers = []  # (net_gain, cust, from_idx, to_idx)
            for cust, rem_cost in removal_cache.items():
                if rem_cost is None:
                    continue
                from_idx = cust_to_route[cust]
                # try inserting into other routes
                for to_idx, to_route in enumerate(routes):
                    if to_idx == from_idx:
                        continue
                    if route_loads[to_idx] + self.demands[cust] > capacity:
                        continue
                    ins_cost, pos = self.best_insertion_cost(cust, to_route)
                    if ins_cost is None:
                        continue
                    signal = route_signals[to_idx] if to_idx < len(route_signals) else 1.0
                    net_gain = rem_cost - ins_cost * signal
                    if net_gain > 1e-9:
                        offers.append((net_gain, cust, from_idx, to_idx, pos, rem_cost, ins_cost))

            # Sort offers by net_gain descending and greedily apply non-conflicting moves
            offers.sort(reverse=True, key=lambda x: x[0])
            moved_customers = set()
            applied_moves = []
            for gain, cust, from_idx, to_idx, pos, rem_cost, ins_cost in offers:
                # validate indices and that cust not moved already
                if cust in moved_customers:
                    continue
                if from_idx < 0 or to_idx < 0 or from_idx >= len(routes) or to_idx >= len(routes):
                    continue
                # check still in from route
                if cust not in routes[from_idx]:
                    continue
                # capacity check with current loads
                if route_loads[to_idx] + self.demands[cust] > capacity:
                    continue
                # recompute best insertion position and removal cost in current routes (they may have changed)
                ins_cost_cur, pos_cur = self.best_insertion_cost(cust, routes[to_idx])
                rem_cost_cur = self.removal_cost(cust, routes[from_idx])
                if ins_cost_cur is None or rem_cost_cur is None:
                    continue
                signal = route_signals[to_idx] if to_idx < len(route_signals) else 1.0
                net_gain_cur = rem_cost_cur - ins_cost_cur * signal
                if net_gain_cur <= 1e-9:
                    continue
                # perform move: insert then remove
                routes[to_idx] = routes[to_idx][:pos_cur] + [cust] + routes[to_idx][pos_cur:]
                # remove first occurrence from from route
                try:
                    routes[from_idx].remove(cust)
                except ValueError:
                    # unlikely, but skip if remove fails
                    continue
                # normalize affected routes
                routes[to_idx] = normalize_route(routes[to_idx])
                routes[from_idx] = normalize_route(routes[from_idx])
                # update loads and mappings
                route_loads[to_idx] += self.demands[cust]
                route_loads[from_idx] -= self.demands[cust]
                moved_customers.add(cust)
                cust_to_route[cust] = to_idx
                applied_moves.append((cust, from_idx, to_idx))
            if applied_moves:
                improved = True

            # Local route improvements: apply 2-opt on each route to reduce route cost
            for idx in range(len(routes)):
                r = routes[idx]
                if len(r) <= 3:
                    continue
                new_r = two_opt_route(r,self.distance_matrix)
                # compare costs once to avoid repeated computations
                if self.route_cost(new_r) + 1e-9 < self.route_cost(r):
                    routes[idx] = normalize_route(new_r)
                    improved = True

            # Pairwise negotiations: limited swaps between routes
            R = len(routes)
            pair_tries = 0
            max_pair_tries = 5 * max(1, R)
            for a in range(R):
                for b in range(a + 1, R):
                    if pair_tries >= max_pair_tries:
                        break
                    pair_tries += 1
                    ra = routes[a]
                    rb = routes[b]
                    la = self.route_load(ra)
                    lb = self.route_load(rb)
                    best_delta = 1e-9
                    best_pair = None
                    # try swapping single customers
                    for ca in ra:
                        if ca == 0:
                            continue
                        for cb in rb:
                            if cb == 0:
                                continue
                            new_la = la - self.demands[ca] + self.demands[cb]
                            new_lb = lb - self.demands[cb] + self.demands[ca]
                            if new_la > capacity or new_lb > capacity:
                                continue
                            # build temporary routes without the swapped customers
                            ra_tmp = [x for x in ra if x != ca]
                            rb_tmp = [x for x in rb if x != cb]
                            ra_tmp = normalize_route(ra_tmp)
                            rb_tmp = normalize_route(rb_tmp)
                            ins_cb_cost, pos_cb = self.best_insertion_cost(cb, ra_tmp)
                            ins_ca_cost, pos_ca = self.best_insertion_cost(ca, rb_tmp)
                            rem_ca = self.removal_cost(ca, ra)
                            rem_cb = self.removal_cost(cb, rb)
                            if None in (ins_cb_cost, ins_ca_cost, rem_ca, rem_cb):
                                continue
                            # delta: (inserts) - (removals saved). Negative delta => improvement
                            delta = (ins_cb_cost - rem_ca) + (ins_ca_cost - rem_cb)
                            if delta < best_delta:
                                best_delta = delta
                                best_pair = (ca, cb, pos_cb, pos_ca)
                    if best_pair:
                        ca, cb, pos_cb, pos_ca = best_pair
                        # apply swap (remove and insert)
                        routes[a] = [x for x in routes[a] if x != ca]
                        routes[b] = [x for x in routes[b] if x != cb]
                        routes[a] = routes[a][:pos_cb] + [cb] + routes[a][pos_cb:]
                        routes[b] = routes[b][:pos_ca] + [ca] + routes[b][pos_ca:]
                        routes[a] = normalize_route(routes[a])
                        routes[b] = normalize_route(routes[b])
                        improved = True
                if pair_tries >= max_pair_tries:
                    break

            # Cleanup: remove trivial routes (only depot or empty)
            cleaned_routes = []
            for r in routes:
                r = normalize_route(r)
                # skip routes that visit no customer
                if len([x for x in r if x != 0]) == 0:
                    continue
                if len(r) <= 2:
                    continue
                cleaned_routes.append(r)
            routes = cleaned_routes

            # Update route signals: encourage balanced loads, clamp to [0.5, 2.0]
            beta = 1.0
            route_signals = []
            for r in routes:
                lr = self.route_load(r)
                signal = 1.0 + beta * ((lr / capacity) - 0.5)
                signal = max(0.5, min(2.0, signal))
                route_signals.append(signal)

            # Evaluate cost and track best
            current_cost = self.total_cost(routes)
            if current_cost + 1e-9 < best_cost:
                best_cost = current_cost
                best_routes = deepcopy(routes)
                no_improve = 0
            else:
                no_improve += 1

            # Diversification if stagnating
            if no_improve >= stagnation_limit:
                num_perturb = max(1, int(0.05 * (n - 1)))
                all_customers = list(range(1, n))
                random.shuffle(all_customers)
                pert = all_customers[:num_perturb]
                for cust in pert:
                    # remove cust from its route if present
                    for r in routes:
                        if cust in r:
                            r.remove(cust)
                            break
                    # try to reinsert into best feasible route
                    best_ins = None
                    best_cost_inc = float('inf')
                    for idx, r in enumerate(routes):
                        if self.route_load(r) + self.demands[cust] > capacity:
                            continue
                        inc, pos = self.best_insertion_cost(cust, r)
                        if inc is None:
                            continue
                        if inc < best_cost_inc:
                            best_cost_inc = inc
                            best_ins = (idx, pos)
                    if best_ins is None:
                        routes.append([0, cust, 0])
                    else:
                        idx, pos = best_ins
                        routes[idx] = routes[idx][:pos] + [cust] + routes[idx][pos:]
                no_improve = 0
                # continue to next iteration to allow local improvements
                continue

            # If nothing improved this iteration, attempt small fixes: insert any orphan customers
            if not improved:
                moved = False
                # rebuild cust_to_route to catch any changes
                cust_to_route = {}
                for idx, r in enumerate(routes):
                    for c in r:
                        if c != 0:
                            cust_to_route[c] = idx
                for cust in range(1, n):
                    if cust not in cust_to_route:
                        # find best feasible insertion
                        best_ins = None
                        best_inc = float('inf')
                        for idx, r in enumerate(routes):
                            if self.route_load(r) + self.demands[cust] > capacity:
                                continue
                            inc, pos = self.best_insertion_cost(cust, r)
                            if inc is None:
                                continue
                            if inc < best_inc:
                                best_inc = inc
                                best_ins = (idx, pos)
                        if best_ins:
                            idx, pos = best_ins
                            routes[idx] = routes[idx][:pos] + [cust] + routes[idx][pos:]
                            moved = True
                if not moved:
                    no_improve += 1
                    if no_improve >= stagnation_limit:
                        break

        # Post-processing: use best found routes
        final_routes = deepcopy(best_routes)
        # Final local improvements
        for idx in range(len(final_routes)):
            final_routes[idx] = normalize_route(two_opt_route(final_routes[idx],self.distance_matrix))

        # Ensure all customers covered exactly once: insert missing
        covered = set()
        for r in final_routes:
            for c in r:
                if c != 0:
                    covered.add(c)
        missing = [i for i in range(1, n) if i not in covered]
        for m in missing:
            best_inc = float('inf')
            best_idx = None
            best_pos = None
            for idx, r in enumerate(final_routes):
                if self.route_load(r) + self.demands[m] > capacity:
                    continue
                inc, pos = self.best_insertion_cost(m, r)
                if inc is None:
                    continue
                if inc < best_inc:
                    best_inc = inc
                    best_idx = idx
                    best_pos = pos
            if best_idx is None:
                final_routes.append([0, m, 0])
            else:
                final_routes[best_idx] = final_routes[best_idx][:best_pos] + [m] + final_routes[best_idx][best_pos:]

        # Final split on over-capacity routes (safe fallback)
        cleaned = []
        for r in final_routes:
            if self.route_load(r) <= capacity:
                cleaned.append(normalize_route(r))
            else:
                cur = [0]
                cur_load = 0
                for node in r[1:-1]:
                    d = self.demands[node]
                    if cur_load + d > capacity:
                        cur.append(0)
                        cleaned.append(normalize_route(cur))
                        cur = [0]
                        cur_load = 0
                    cur.append(node)
                    cur_load += d
                cur.append(0)
                cleaned.append(normalize_route(cur))

        # Flatten to required format
        solution = self.flatten_routes(cleaned)

        # Final verification: ensure each customer visited exactly once, otherwise fallback
        visited = [0] * n
        for v in solution:
            if 0 <= v < n and v != 0:
                visited[v] += 1
        invalid = any(visited[i] != 1 for i in range(1, n))
        if invalid:
            # fallback: simple sequential packing by index ensuring feasibility
            solution = [0]
            cur_cap = 0
            for i in range(1, n):
                if cur_cap + self.demands[i] > capacity:
                    solution.append(0)
                    cur_cap = 0
                solution.append(i)
                cur_cap += self.demands[i]
            if solution[-1] != 0:
                solution.append(0)

        return solution

