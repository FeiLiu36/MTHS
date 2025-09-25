
import numpy as np
import time
import random
from copy import deepcopy
from numba import njit

@njit
def _two_opt_route(distance_matrix, route):
    """
    Best-improvement 2-opt for a single route.

    Scans all possible 2-opt exchanges in a full pass, selects the best
    (maximum negative) delta, applies it, and repeats until no improving
    move exists. Micro-optimizations:
      - Uses local variables and row caching for faster index access.
      - Handles routes that optionally include depot (node 0) at ends.
      - Preserves tuple input by returning a tuple; otherwise returns a list.
      - Adds a large iteration safeguard to avoid pathological infinite loops.

    Assumes self.distance_matrix supports indexing like D[i][j] or D[i, j].
    """
    # Preserve input type (common case is list; if tuple, return tuple)
    return_tuple = isinstance(route, tuple)
    r = list(route)  # work on a mutable local copy
    n = len(r)

    # If route includes depot at first and/or last position (0), strip them
    # so logic below can treat depot as implicit node 0. We only strip if
    # depot appears at the ends (common representations).
    strip_front = n > 0 and r[0] == 0
    strip_back = n > 1 and r[-1] == 0
    if strip_front:
        r = r[1:]
    if strip_back:
        r = r[:-1]
    n = len(r)

    # Trivial cases
    # if n <= 1:
    #     return tuple(r) if return_tuple else r if not strip_front and not strip_back else \
    #            ([0] + r + ([0] if strip_back else [])) if not return_tuple else \
    #            tuple(([0] + r + ([0] if strip_back else [])))
    # Trivial cases
    if n <= 1:
        # Start with the base list 'r'
        result = r

        # Apply stripping logic first to modify the list
        if strip_front:
            result = [0] + result
        if strip_back:
            result = result + [0]

        # Finally, convert to a tuple if required
        if return_tuple:
            return tuple(result)
        else:
            return result

    D = distance_matrix  # local alias for speed
    eps = 1e-12

    # Safeguard to prevent pathological infinite loops (very generous limit)
    max_iterations = max(1000, n * n)
    iteration = 0

    # Main best-improvement 2-opt loop
    while True:
        iteration += 1
        if iteration > max_iterations:
            # fallback: stop further improvement attempts
            break

        best_delta = -eps
        best_i = -1
        best_j = -1

        # scan all i < j pairs
        # removing edges (a-b) and (c-d), adding (a-c) and (b-d)
        for i in range(n - 1):
            a = 0 if i == 0 else r[i - 1]
            b = r[i]
            D_a = D[a]  # row for a (works for numpy array or nested list)
            D_b = D[b]  # row for b
            D_ab = D_a[b]

            # inner loop j from i+1 .. n-1
            # when j == n-1, d is depot (0)
            for j in range(i + 1, n):
                c = r[j]
                d = 0 if j == n - 1 else r[j + 1]
                # compute delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]
                # use cached rows where useful
                D_c = D[c]
                delta = D_a[c] + D_b[d] - D_ab - D_c[d]
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        # if we found an improving move apply the best one, otherwise stop
        if best_delta < -eps:
            i = best_i
            j = best_j
            # reverse segment [i..j] in-place
            # Python slice reversal is efficient in C
            r[i:j + 1] = r[i:j + 1][::-1]
            # continue searching for further improvements
            continue
        else:
            break

    # If depot was stripped, reattach it at front/back
    if strip_front:
        r.insert(0, 0)
    if strip_back:
        r.append(0)

    return tuple(r) if return_tuple else r

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
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.n = len(coordinates)

        # internal algorithm components
        self.random = random.Random(0)

    # ---------- Helper utilities ----------
    def _cost_of_routes(self, routes_flat):
        # routes_flat is flat list with 0 separators
        cost = 0.0
        for i in range(len(routes_flat) - 1):
            a = routes_flat[i]
            b = routes_flat[i + 1]
            cost += self.distance_matrix[a, b]
        return cost

    def _flat_to_routes(self, flat):
        routes = []
        current = []
        for v in flat:
            if v == 0:
                if current:
                    routes.append(current[:])
                    current = []
            else:
                current.append(v)
        # If last route not closed add if exists
        return routes

    def _routes_to_flat(self, routes):
        flat = []
        for r in routes:
            flat.append(0)
            flat.extend(r)
            flat.append(0)
        # remove duplicate 0 at end if any
        if len(flat) > 1 and flat[-1] == 0:
            # keep final depot; required by format
            pass
        return flat

    def _route_demand(self, route):
        return sum(self.demands[idx] for idx in route)

    def _is_feasible(self, flat):
        # check coverage and capacity
        visited = set()
        for v in flat:
            if v != 0:
                visited.add(v)
        all_customers = set(range(1, self.n))
        if visited != all_customers:
            return False
        for r in self._flat_to_routes(flat):
            if self._route_demand(r) > self.vehicle_capacity + 1e-9:
                return False
        return True

    def _initial_greedy_seed(self, shuffle=False):
        # simple capacity-first sequential fill or randomized order
        customers = list(range(1, self.n))
        if shuffle:
            self.random.shuffle(customers)
        routes = []
        cur = []
        cur_cap = 0
        for c in customers:
            d = self.demands[c]
            if cur_cap + d > self.vehicle_capacity:
                routes.append(cur)
                cur = [c]
                cur_cap = d
            else:
                cur.append(c)
                cur_cap += d
        if cur:
            routes.append(cur)
        return self._routes_to_flat(routes)

    def _clarke_wright(self):
        # Clarke-Wright savings heuristic
        n = self.n
        depot = 0
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = self.distance_matrix[depot, i] + self.distance_matrix[depot, j] - self.distance_matrix[i, j]
                savings.append((s, i, j))
        savings.sort(reverse=True)
        routes = {i: [i] for i in range(1, n)}
        load = {i: self.demands[i] for i in range(1, n)}
        for s, i, j in savings:
            if i not in routes or j not in routes:
                continue
            ri = routes[i]
            rj = routes[j]
            # merge only if i is at end of ri and j at start of rj or vice versa
            if ri is rj:
                continue
            if load[ri[0]] is None:
                pass
            # check endpoints
            if (ri[-1] == i and rj[0] == j) or (ri[0] == i and rj[-1] == j):
                total_load = sum(self.demands[x] for x in ri) + sum(self.demands[x] for x in rj)
                if total_load <= self.vehicle_capacity:
                    # merge ri and rj properly
                    if ri[-1] == i and rj[0] == j:
                        newr = ri + rj
                    elif ri[0] == i and rj[-1] == j:
                        newr = rj + ri
                    else:
                        continue
                    for x in newr:
                        routes[x] = newr
        # build unique routes
        seen = set()
        final = []
        for i in range(1, n):
            r = routes[i]
            rid = tuple(r)
            if rid not in seen:
                seen.add(rid)
                final.append(list(r))
        return self._routes_to_flat(final)
    #
    # def _two_opt_route(self, route):
    #     """
    #     Best-improvement 2-opt for a single route.
    #
    #     Scans all possible 2-opt exchanges in a full pass, selects the best
    #     (maximum negative) delta, applies it, and repeats until no improving
    #     move exists. Micro-optimizations:
    #       - Uses local variables and row caching for faster index access.
    #       - Handles routes that optionally include depot (node 0) at ends.
    #       - Preserves tuple input by returning a tuple; otherwise returns a list.
    #       - Adds a large iteration safeguard to avoid pathological infinite loops.
    #
    #     Assumes self.distance_matrix supports indexing like D[i][j] or D[i, j].
    #     """
    #     # Preserve input type (common case is list; if tuple, return tuple)
    #     return_tuple = isinstance(route, tuple)
    #     r = list(route)  # work on a mutable local copy
    #     n = len(r)
    #
    #     # If route includes depot at first and/or last position (0), strip them
    #     # so logic below can treat depot as implicit node 0. We only strip if
    #     # depot appears at the ends (common representations).
    #     strip_front = n > 0 and r[0] == 0
    #     strip_back = n > 1 and r[-1] == 0
    #     if strip_front:
    #         r = r[1:]
    #     if strip_back:
    #         r = r[:-1]
    #     n = len(r)
    #
    #     # Trivial cases
    #     if n <= 1:
    #         return tuple(r) if return_tuple else r if not strip_front and not strip_back else \\
    #                ([0] + r + ([0] if strip_back else [])) if not return_tuple else \\
    #                tuple(([0] + r + ([0] if strip_back else [])))
    #
    #     D = self.distance_matrix  # local alias for speed
    #     eps = 1e-12
    #
    #     # Safeguard to prevent pathological infinite loops (very generous limit)
    #     max_iterations = max(1000, n * n)
    #     iteration = 0
    #
    #     # Main best-improvement 2-opt loop
    #     while True:
    #         iteration += 1
    #         if iteration > max_iterations:
    #             # fallback: stop further improvement attempts
    #             break
    #
    #         best_delta = -eps
    #         best_i = -1
    #         best_j = -1
    #
    #         # scan all i < j pairs
    #         # removing edges (a-b) and (c-d), adding (a-c) and (b-d)
    #         for i in range(n - 1):
    #             a = 0 if i == 0 else r[i - 1]
    #             b = r[i]
    #             D_a = D[a]  # row for a (works for numpy array or nested list)
    #             D_b = D[b]  # row for b
    #             D_ab = D_a[b]
    #
    #             # inner loop j from i+1 .. n-1
    #             # when j == n-1, d is depot (0)
    #             for j in range(i + 1, n):
    #                 c = r[j]
    #                 d = 0 if j == n - 1 else r[j + 1]
    #                 # compute delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]
    #                 # use cached rows where useful
    #                 D_c = D[c]
    #                 delta = D_a[c] + D_b[d] - D_ab - D_c[d]
    #                 if delta < best_delta:
    #                     best_delta = delta
    #                     best_i = i
    #                     best_j = j
    #
    #         # if we found an improving move apply the best one, otherwise stop
    #         if best_delta < -eps:
    #             i = best_i
    #             j = best_j
    #             # reverse segment [i..j] in-place
    #             # Python slice reversal is efficient in C
    #             r[i:j + 1] = r[i:j + 1][::-1]
    #             # continue searching for further improvements
    #             continue
    #         else:
    #             break
    #
    #     # If depot was stripped, reattach it at front/back
    #     if strip_front:
    #         r.insert(0, 0)
    #     if strip_back:
    #         r.append(0)
    #
    #     return tuple(r) if return_tuple else r

    # ---------- Fractional representation and fragments ----------
    def _make_fractional_flows(self, temps=(0.5, 1.0, 2.0)):
        flows = []
        n = self.n
        D = self.distance_matrix
        for t in temps:
            # use softmax over negative distances scaled by t
            mat = np.exp(-D / (np.maximum(t, 1e-6)))
            # zero diagonal and normalize rows to sum to 1
            np.fill_diagonal(mat, 0.0)
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat = mat / row_sums
            flows.append(mat)
        return flows

    def _extract_fragments_from_flow(self, flow, threshold=0.15, max_frag_len=6):
        n = self.n
        used = set()
        fragments = []
        # build edges sorted by weight
        edges = []
        for i in range(1, n):  # only consider customer outgoing
            for j in range(1, n):
                if i != j:
                    edges.append((flow[i, j], i, j))
        edges.sort(reverse=True)
        for w, i, j in edges:
            if w < threshold:
                break
            if i in used or j in used:
                continue
            # attempt to grow chain
            chain = [i, j]
            used.update(chain)
            # grow forward greedily
            while len(chain) < max_frag_len:
                tail = chain[-1]
                # choose next with highest flow from tail
                cand = np.argmax(flow[tail])
                if cand == 0 or cand in chain or flow[tail, cand] < threshold:
                    break
                chain.append(int(cand))
                if cand in used:
                    break
                used.add(cand)
            fragments.append(chain)
        # also add singletons (high confidence nodes that weren't used)
        if not fragments:
            # fallback: add best singletons
            top_nodes = np.argsort(-flow.sum(axis=1))[1: min(n, 1+5)]
            for v in top_nodes:
                if v != 0:
                    fragments.append([int(v)])
        return fragments

    # ---------- Multi-scale relinking and combination ----------
    def _combine_parents_and_fragments(self, parent_flat, other_flat, fragments, operator):
        # parent_flat, other_flat are solutions; fragments is list of chains
        # operator chooses combination strategy
        p_routes = self._flat_to_routes(parent_flat)
        o_routes = self._flat_to_routes(other_flat)
        customers = set(range(1, self.n))
        in_solution = set()
        new_routes = deepcopy(p_routes)

        if operator == 'replace_route':
            # replace one route in p with a route from other if feasible insert
            if len(o_routes) > 0:
                r = random.choice(o_routes)
                # attempt to insert r by splitting/merging if capacity allows
                # find route(s) to remove customers that conflict
                removed = []
                for c in r:
                    for rr in new_routes:
                        if c in rr:
                            removed.append((rr, c))
                for rr, c in removed:
                    if c in rr:
                        rr.remove(c)
                # remove empty
                new_routes = [rr for rr in new_routes if rr]
                # insert r as new route, but if it exceeds capacity then split
                if self._route_demand(r) <= self.vehicle_capacity:
                    new_routes.append(r[:])
                else:
                    # split greedily
                    tmp = []
                    cur = []
                    curcap = 0
                    for c in r:
                        if curcap + self.demands[c] > self.vehicle_capacity:
                            tmp.append(cur)
                            cur = [c]; curcap = self.demands[c]
                        else:
                            cur.append(c); curcap += self.demands[c]
                    if cur: tmp.append(cur)
                    new_routes.extend(tmp)
        elif operator == 'insert_fragments':
            # Insert fragments into best positions greedily
            # start with routes as p_routes
            for frag in fragments:
                frag_d = sum(self.demands[x] for x in frag)
                # try to insert into existing route position minimizing incremental cost
                best_cost_inc = None
                best_route_idx = None
                best_pos = None
                for idx, rr in enumerate(new_routes):
                    if sum(self.demands[x] for x in rr) + frag_d > self.vehicle_capacity:
                        continue
                    # try to insert fragment as contiguous block at any position
                    for pos in range(len(rr)+1):
                        # compute incremental cost
                        left = 0 if pos==0 else self.distance_matrix[rr[pos-1], frag[0]]
                        right = 0 if pos==len(rr) else self.distance_matrix[frag[-1], rr[pos]]
                        replaced = 0
                        # cost change: remove link from left->right and add left->frag[0], frag[-1]->right
                        base = 0
                        if pos==0:
                            base += self.distance_matrix[0, rr[0]]
                            inc = self.distance_matrix[0, frag[0]] + self.distance_matrix[frag[-1], rr[0]] - base
                        elif pos==len(rr):
                            base += self.distance_matrix[rr[-1], 0]
                            inc = self.distance_matrix[rr[-1], frag[0]] + self.distance_matrix[frag[-1], 0] - base
                        else:
                            base += self.distance_matrix[rr[pos-1], rr[pos]]
                            inc = self.distance_matrix[rr[pos-1], frag[0]] + self.distance_matrix[frag[-1], rr[pos]] - base
                        if best_cost_inc is None or inc < best_cost_inc:
                            best_cost_inc = inc
                            best_route_idx = idx
                            best_pos = pos
                if best_route_idx is not None:
                    rr = new_routes[best_route_idx]
                    rr[best_pos:best_pos] = frag[:]
                else:
                    # open new route(s)
                    if frag_d <= self.vehicle_capacity:
                        new_routes.append(frag[:])
                    else:
                        tmp = []
                        cur = []
                        curcap = 0
                        for c in frag:
                            if curcap + self.demands[c] > self.vehicle_capacity:
                                tmp.append(cur)
                                cur = [c]; curcap = self.demands[c]
                            else:
                                cur.append(c); curcap += self.demands[c]
                        if cur: tmp.append(cur)
                        new_routes.extend(tmp)
        elif operator == 'merge_small':
            # merge small routes via best insertion
            smalls = [r for r in new_routes if sum(self.demands[x] for x in r) < self.vehicle_capacity * 0.5]
            others = [r for r in new_routes if r not in smalls]
            for s in smalls:
                placed = False
                for rr in others:
                    if sum(self.demands[x] for x in rr) + sum(self.demands[x] for x in s) <= self.vehicle_capacity:
                        rr.extend(s)
                        placed = True
                        break
                if not placed:
                    others.append(s)
            new_routes = others
        else:
            # default: use parent with some routes replaced by fragments
            if fragments:
                chosen = random.choice(fragments)
                # remove those nodes from new_routes then append chosen
                for c in chosen:
                    for rr in new_routes:
                        if c in rr:
                            rr.remove(c)
                new_routes = [rr for rr in new_routes if rr]
                # insert chosen greedily
                if sum(self.demands[x] for x in chosen) <= self.vehicle_capacity:
                    new_routes.append(chosen[:])
                else:
                    tmp = []
                    cur = []; curcap = 0
                    for c in chosen:
                        if curcap + self.demands[c] > self.vehicle_capacity:
                            tmp.append(cur)
                            cur = [c]; curcap = self.demands[c]
                        else:
                            cur.append(c); curcap += self.demands[c]
                    if cur: tmp.append(cur)
                    new_routes.extend(tmp)

        # after combination, fill missing customers from other_flat or parent_flat
        assigned = set()
        for rr in new_routes:
            for c in rr:
                assigned.add(c)
        all_customers = set(range(1, self.n))
        missing = list(all_customers - assigned)
        # append missing using greedy best insertion
        for c in missing:
            best_inc = None
            best_route_idx = None
            best_pos = None
            for idx, rr in enumerate(new_routes):
                if sum(self.demands[x] for x in rr) + self.demands[c] > self.vehicle_capacity:
                    continue
                for pos in range(len(rr)+1):
                    if pos==0:
                        base = self.distance_matrix[0, rr[0]] if rr else self.distance_matrix[0,0]
                        inc = self.distance_matrix[0, c] + (self.distance_matrix[c, rr[0]] if rr else self.distance_matrix[c, 0]) - base
                    elif pos==len(rr):
                        base = self.distance_matrix[rr[-1], 0]
                        inc = self.distance_matrix[rr[-1], c] + self.distance_matrix[c, 0] - base
                    else:
                        base = self.distance_matrix[rr[pos-1], rr[pos]]
                        inc = self.distance_matrix[rr[pos-1], c] + self.distance_matrix[c, rr[pos]] - base
                    if best_inc is None or inc < best_inc:
                        best_inc = inc; best_route_idx = idx; best_pos = pos
            if best_route_idx is not None:
                new_routes[best_route_idx][best_pos:best_pos] = [c]
            else:
                new_routes.append([c])
        # final repair to ensure capacity
        final_routes = []
        for rr in new_routes:
            if sum(self.demands[x] for x in rr) <= self.vehicle_capacity:
                final_routes.append(rr)
            else:
                # split greedily
                cur = []; curcap = 0
                for c in rr:
                    if curcap + self.demands[c] > self.vehicle_capacity:
                        final_routes.append(cur)
                        cur = [c]; curcap = self.demands[c]
                    else:
                        cur.append(c); curcap += self.demands[c]
                if cur:
                    final_routes.append(cur)
        # ensure every customer appears exactly once
        seen = set()
        cleaned = []
        for rr in final_routes:
            rr2 = [x for x in rr if x not in seen]
            seen.update(rr2)
            if rr2:
                cleaned.append(rr2)
        missing2 = set(range(1, self.n)) - seen
        for c in missing2:
            # create new route for missing
            cleaned.append([c])
        # apply intra-route 2-opt smoothing (light)
        cleaned2 = [_two_opt_route(self.distance_matrix,r) for r in cleaned]
        return self._routes_to_flat(cleaned2)

    # ---------- Hierarchical rounding and layered repair ----------
    def _hierarchical_round_and_repair(self, fractional_fragments, base_flat):
        # fractional_fragments: list of chains (lists)
        # base_flat: starting solution
        # build new candidate by inserting fragments with varied aggressiveness
        operators = ['insert_fragments', 'replace_route', 'merge_small', 'default']
        op = self.random.choice(operators)
        other_flat = base_flat  # use base as secondary
        cand = self._combine_parents_and_fragments(base_flat, other_flat, fractional_fragments, op)
        # layered repair: relocation moves to fix capacity overflows
        improved = True
        iter_caps = 0
        while not self._is_feasible(cand) and iter_caps < 20:
            routes = self._flat_to_routes(cand)
            # find overloaded routes
            overloaded = [r for r in routes if self._route_demand(r) > self.vehicle_capacity]
            if not overloaded:
                break
            for r in overloaded:
                # try to move largest-demand customers out
                demands_sorted = sorted(r, key=lambda x: self.demands[x], reverse=True)
                for c in demands_sorted:
                    moved = False
                    # try to insert into existing routes
                    for rr in routes:
                        if rr is r:
                            continue
                        if sum(self.demands[x] for x in rr) + self.demands[c] <= self.vehicle_capacity:
                            # insert at best pos
                            best_pos = 0; best_inc = None
                            for pos in range(len(rr)+1):
                                if pos==0:
                                    base = self.distance_matrix[0, rr[0]] if rr else self.distance_matrix[0,0]
                                    inc = self.distance_matrix[0,c] + (self.distance_matrix[c, rr[0]] if rr else self.distance_matrix[c, 0]) - base
                                elif pos==len(rr):
                                    base = self.distance_matrix[rr[-1], 0]
                                    inc = self.distance_matrix[rr[-1], c] + self.distance_matrix[c, 0] - base
                                else:
                                    base = self.distance_matrix[rr[pos-1], rr[pos]]
                                    inc = self.distance_matrix[rr[pos-1], c] + self.distance_matrix[c, rr[pos]] - base
                                if best_inc is None or inc < best_inc:
                                    best_inc = inc; best_pos = pos
                            rr[best_pos:best_pos] = [c]
                            r.remove(c)
                            moved = True
                            break
                    if moved:
                        break
                # if still overloaded, split route
                if sum(self.demands[x] for x in r) > self.vehicle_capacity:
                    # split preserving order
                    tmp = []
                    cur = []; curcap = 0
                    for c in r:
                        if curcap + self.demands[c] > self.vehicle_capacity:
                            tmp.append(cur); cur = [c]; curcap = self.demands[c]
                        else:
                            cur.append(c); curcap += self.demands[c]
                    if cur: tmp.append(cur)
                    # replace this route with tmp
                    routes.remove(r)
                    routes.extend(tmp)
                    break
            cand = self._routes_to_flat(routes)
            iter_caps += 1
        # final ensure coverage
        allcust = set(range(1, self.n))
        assigned = set(v for v in cand if v != 0)
        missing = list(allcust - assigned)
        for c in missing:
            # insert missing as separate route or greedy
            # try to append to smallest route that can hold it
            routes = self._flat_to_routes(cand)
            placed = False
            sizes = [(sum(self.demands[x] for x in rr), idx) for idx, rr in enumerate(routes)]
            sizes.sort()
            for _, idx in sizes:
                if sum(self.demands[x] for x in routes[idx]) + self.demands[c] <= self.vehicle_capacity:
                    routes[idx].append(c)
                    placed = True
                    break
            if not placed:
                routes.append([c])
            cand = self._routes_to_flat(routes)
        # smoothing
        routes = self._flat_to_routes(cand)
        routes = [_two_opt_route(self.distance_matrix,r) for r in routes]
        cand = self._routes_to_flat(routes)
        return cand

    # ---------- Surrogate estimator ----------
    class _Surrogate:
        def __init__(self, dim=4):
            self.dim = dim
            self.X = []
            self.y = []
            self.w = None
        def featurize(self, flat, distance_matrix, demands, vehicle_capacity):
            routes = []
            current = []
            for v in flat:
                if v == 0:
                    if current:
                        routes.append(current[:])
                        current = []
                else:
                    current.append(v)
            num_routes = len(routes)
            total_distance = 0.0
            for r in routes:
                if r:
                    total_distance += distance_matrix[0, r[0]]
                    for i in range(len(r)-1):
                        total_distance += distance_matrix[r[i], r[i+1]]
                    total_distance += distance_matrix[r[-1], 0]
            avg_load = 0.0
            if num_routes > 0:
                loads = [sum(demands[x] for x in r) for r in routes]
                avg_load = float(np.mean(loads))
            distinct_pairs = 0
            for r in routes:
                distinct_pairs += len(r) - 1 if len(r) >= 2 else 0
            return np.array([num_routes, total_distance, avg_load, distinct_pairs], dtype=float)

        def predict(self, x):
            if self.w is None:
                # fallback: simple linear proxy: use total_distance feature if present
                if len(x.shape) == 1:
                    return x[1]
                else:
                    return x[:,1]
            return x.dot(self.w)

        def update(self, X, y):
            # X: list of feature vectors, y: list of costs
            if len(X) < 2:
                return
            A = np.array(X)
            b = np.array(y)
            # ridge regularization
            lambda_reg = 1e-3
            try:
                self.w = np.linalg.solve(A.T.dot(A) + lambda_reg * np.eye(A.shape[1]), A.T.dot(b))
            except np.linalg.LinAlgError:
                self.w = np.linalg.lstsq(A, b, rcond=None)[0]

    # ---------- Main solve ----------
    def solve(self, max_time=60.0, max_iter=10000) -> list:
        n = self.n
        start_time = time.time()

        # Prepare initial mixed pool
        pool = []
        # several greedy seeds
        pool.append(self._initial_greedy_seed(shuffle=False))
        pool.append(self._initial_greedy_seed(shuffle=True))
        pool.append(self._clarke_wright())
        # random greedy seeds
        for _ in range(3):
            pool.append(self._initial_greedy_seed(shuffle=True))
        # fractional representations
        temps = [0.5, 1.0, 2.0]
        flows = self._make_fractional_flows(temps=temps)
        # create fractional "soft solutions" represented by top edges -> convert to routings via fragments
        for f in flows:
            frags = self._extract_fragments_from_flow(f, threshold=0.12 + self.random.random()*0.05)
            cand = self._hierarchical_round_and_repair(frags, pool[0])
            pool.append(cand)

        # elite archive: keep top K
        def score(sol):
            return self._cost_of_routes(sol)
        pool_scores = [(s, score(s)) for s in pool]
        pool_scores.sort(key=lambda x: x[1])
        pool = [s for s,_ in pool_scores[:10]]

        elite_archive = [pool[0]]
        elite_scores = [score(pool[0])]
        pattern_memory = {}  # pair -> count
        surrogate = CVRPSolver._Surrogate(dim=4)
        surrogate.update([surrogate.featurize(elite_archive[0], self.distance_matrix, self.demands, self.vehicle_capacity)], [elite_scores[0]])
        operator_weights = {'insert_fragments':1.0, 'replace_route':1.0, 'merge_small':1.0, 'default':1.0}
        no_improve = 0
        best_sol = elite_archive[0]
        best_cost = elite_scores[0]

        iteration = 0
        while iteration < max_iter and (time.time() - start_time) < max_time:
            iteration += 1
            # Improve continuous representations: slightly adjust temperatures and regenerate flows
            temp_scale = 0.8 + 0.4 * (np.tanh((iteration / max_iter) * 3.0))
            temps = [0.5*temp_scale, 1.0*temp_scale, 2.0*temp_scale]
            flows = self._make_fractional_flows(temps=temps)

            # extract fragments across flows
            all_frags = []
            for f in flows:
                th = 0.12 + self.random.random()*0.08
                fr = self._extract_fragments_from_flow(f, threshold=th)
                all_frags.extend(fr)

            # select parents guided by quality and diversity
            # compute scores for pool
            pool = list(set(tuple(p) for p in pool))  # unique
            pool = [list(p) for p in pool]
            pool_scores = [(p, score(p)) for p in pool]
            pool_scores.sort(key=lambda x: x[1])
            pool = [p for p,_ in pool_scores]
            parent_a = pool[0]
            parent_b = pool[1] if len(pool) > 1 else pool[0]

            # build ensemble: sometimes mix in elite
            parents = [parent_a, parent_b]
            if self.random.random() < 0.3 and elite_archive:
                parents.append(elite_archive[0])

            # choose operator adaptively
            ops = list(operator_weights.keys())
            w = np.array([operator_weights[k] for k in ops], dtype=float)
            probs = w / w.sum()
            op = self.random.choices(ops, weights=probs, k=1)[0]

            # multi-scale relinking: combine fragments and parents
            chosen_frags = []
            # pick some high-confidence fragments probabilistically
            if all_frags:
                kfrag = max(1, min(5, int(1 + (self.random.random()*4))))
                chosen_frags = self.random.sample(all_frags, min(kfrag, len(all_frags)))
            base_parent = parents[0]
            other_parent = parents[1] if len(parents)>1 else parents[0]
            candidate = self._combine_parents_and_fragments(base_parent, other_parent, chosen_frags, op)

            # hierarchical rounding and layered repair applied (again)
            candidate = self._hierarchical_round_and_repair(chosen_frags, candidate)

            # evaluate true cost
            cand_cost = score(candidate)

            # update pattern memory: count frequent consecutive pairs
            routes = self._flat_to_routes(candidate)
            for r in routes:
                for i in range(len(r)-1):
                    pair = (r[i], r[i+1])
                    pattern_memory[pair] = pattern_memory.get(pair, 0) + 1

            # surrogate feedback - update dataset
            feat = surrogate.featurize(candidate, self.distance_matrix, self.demands, self.vehicle_capacity)
            surrogate.X.append(feat)
            surrogate.y.append(cand_cost)
            if len(surrogate.X) % 10 == 0:
                surrogate.update(surrogate.X, surrogate.y)

            # update elite archive
            improved_now = False
            if cand_cost + 1e-9 < best_cost:
                best_cost = cand_cost
                best_sol = candidate
                improved_now = True
                no_improve = 0
            else:
                no_improve += 1
            # maintain elite set of up to 5
            elite_archive.append(candidate)
            elite_archive = sorted(elite_archive, key=lambda s: score(s))[:5]
            elite_scores = [score(s) for s in elite_archive]

            # update pool: add candidate and keep diversity
            pool.append(candidate)
            # prune pool keeping best and some randoms
            pool_scores = [(p, score(p)) for p in pool]
            pool_scores.sort(key=lambda x: x[1])
            pool = [p for p,_ in pool_scores[:12]]

            # update operator preferences via simple reinforcement (reward if candidate improved)
            reward = 1.0 if improved_now else 0.1
            operator_weights[op] = operator_weights.get(op, 1.0) * (1.0 - 0.1) + reward * 0.1

            # adapt rounding aggressiveness (implicitly via thresholds variation in extraction)
            if no_improve > 15:
                # perturb: destroy and repair some routes
                routes = self._flat_to_routes(best_sol)
                if routes:
                    # randomly remove customers from some routes and reinsert
                    remove_set = set()
                    num_remove = max(1, int(0.05 * (self.n-1)))
                    all_customers = list(range(1, self.n))
                    self.random.shuffle(all_customers)
                    remove_set = set(all_customers[:num_remove])
                    remaining_routes = []
                    for r in routes:
                        rr = [c for c in r if c not in remove_set]
                        if rr:
                            remaining_routes.append(rr)
                    # reinsert removed greedily
                    for c in remove_set:
                        inserted = False
                        best_inc = None; best_idx = None; best_pos = None
                        for idx, rr in enumerate(remaining_routes):
                            if sum(self.demands[x] for x in rr) + self.demands[c] > self.vehicle_capacity:
                                continue
                            for pos in range(len(rr)+1):
                                if pos==0:
                                    base = self.distance_matrix[0, rr[0]] if rr else self.distance_matrix[0,0]
                                    inc = self.distance_matrix[0, c] + (self.distance_matrix[c, rr[0]] if rr else self.distance_matrix[c, 0]) - base
                                elif pos==len(rr):
                                    base = self.distance_matrix[rr[-1], 0]
                                    inc = self.distance_matrix[rr[-1], c] + self.distance_matrix[c, 0] - base
                                else:
                                    base = self.distance_matrix[rr[pos-1], rr[pos]]
                                    inc = self.distance_matrix[rr[pos-1], c] + self.distance_matrix[c, rr[pos]] - base
                                if best_inc is None or inc < best_inc:
                                    best_inc = inc; best_idx = idx; best_pos = pos
                        if best_idx is not None:
                            remaining_routes[best_idx][best_pos:best_pos] = [c]
                            inserted = True
                        if not inserted:
                            remaining_routes.append([c])
                    # smooth and adopt if better
                    remaining_routes = [_two_opt_route(self.distance_matrix,r) for r in remaining_routes]
                    cand2 = self._routes_to_flat(remaining_routes)
                    c2_cost = score(cand2)
                    if c2_cost < best_cost:
                        best_cost = c2_cost
                        best_sol = cand2
                        elite_archive.append(cand2)
                        elite_archive = sorted(elite_archive, key=lambda s: score(s))[:5]
                        no_improve = 0
                else:
                    no_improve = 0

            # small local search: relocate and swap among random pair of routes
            if self.random.random() < 0.4:
                routes = self._flat_to_routes(candidate)
                if len(routes) >= 2:
                    i, j = self.random.sample(range(len(routes)), 2)
                    ri = routes[i]
                    rj = routes[j]
                    if ri and rj:
                        # try relocating best-benefit single customer from ri to rj
                        best_delta = 0
                        best_move = None
                        for idx_c, c in enumerate(ri):
                            if sum(self.demands[x] for x in rj) + self.demands[c] <= self.vehicle_capacity:
                                # compute delta cost
                                # remove c from ri
                                ri_cost_before = self._route_cost_estimate(ri)
                                rj_cost_before = self._route_cost_estimate(rj)
                                ri2 = ri[:idx_c] + ri[idx_c+1:]
                                bestpos = 0; best_inc = None
                                for pos in range(len(rj)+1):
                                    if pos==0:
                                        base = self.distance_matrix[0, rj[0]] if rj else self.distance_matrix[0,0]
                                        inc = self.distance_matrix[0,c] + (self.distance_matrix[c, rj[0]] if rj else self.distance_matrix[c, 0]) - base
                                    elif pos==len(rj):
                                        base = self.distance_matrix[rj[-1], 0]
                                        inc = self.distance_matrix[rj[-1], c] + self.distance_matrix[c, 0] - base
                                    else:
                                        base = self.distance_matrix[rj[pos-1], rj[pos]]
                                        inc = self.distance_matrix[rj[pos-1], c] + self.distance_matrix[c, rj[pos]] - base
                                    if best_inc is None or inc < best_inc:
                                        best_inc = inc; bestpos = pos
                                ri_cost_after = self._route_cost_estimate(ri2)
                                rj_cost_after = rj_cost_before + (best_inc if best_inc is not None else 0)
                                delta = (ri_cost_after + rj_cost_after) - (ri_cost_before + rj_cost_before)
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (i, j, idx_c, bestpos)
                        if best_move:
                            i, j, idx_c, pos = best_move
                            c = routes[i][idx_c]
                            routes[i].pop(idx_c)
                            routes[j][pos:pos] = [c]
                            routes = [r for r in routes if r]
                            candidate = self._routes_to_flat(routes)
                            cand_cost = score(candidate)
                            if cand_cost < best_cost:
                                best_cost = cand_cost
                                best_sol = candidate
            # progress time check
            if (time.time() - start_time) > max_time:
                break

        # post-processing: select best archived routing, final smoothing and consolidation
        best = best_sol
        # ensure feasibility: final layered repair
        frags = []
        best = self._hierarchical_round_and_repair(frags, best)
        # final local smoothing: 2-opt on each route, and merge small routes where possible
        routes = self._flat_to_routes(best)
        routes = [_two_opt_route(self.distance_matrix,r) for r in routes]
        # try merge small routes
        merged = True
        while merged:
            merged = False
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    if sum(self.demands[x] for x in routes[i]) + sum(self.demands[x] for x in routes[j]) <= self.vehicle_capacity:
                        # try to merge in best order (i+j or j+i)
                        r1 = routes[i] + routes[j]
                        r2 = routes[j] + routes[i]
                        if self._route_cost_estimate(r1) < self._route_cost_estimate(routes[i]) + self._route_cost_estimate(routes[j]):
                            routes[i] = _two_opt_route(self.distance_matrix,r1)
                            routes.pop(j)
                            merged = True
                            break
                        elif self._route_cost_estimate(r2) < self._route_cost_estimate(routes[i]) + self._route_cost_estimate(routes[j]):
                            routes[i] = _two_opt_route(self.distance_matrix,r2)
                            routes.pop(j)
                            merged = True
                            break
                if merged:
                    break
        final_flat = self._routes_to_flat(routes)
        # final quick repair and ensure coverage
        if not self._is_feasible(final_flat):
            final_flat = self._hierarchical_round_and_repair([], final_flat)
        # ensure flat starts with 0
        if len(final_flat) == 0 or final_flat[0] != 0:
            final_flat = [0] + final_flat
        # ensure ends with depot
        if final_flat[-1] != 0:
            final_flat = final_flat + [0]
        # final sanity: ensure each customer once
        assigned = [v for v in final_flat if v != 0]
        counts = {}
        for c in assigned:
            counts[c] = counts.get(c, 0) + 1
        missing = [c for c in range(1, self.n) if c not in counts]
        # remove duplicates by keeping first occurrence
        seen = set()
        cleaned = []
        for v in final_flat:
            if v == 0:
                cleaned.append(0)
            else:
                if v not in seen:
                    cleaned.append(v)
                    seen.add(v)
        for c in missing:
            cleaned.insert(-1, c)
        # ensure depot separators correct: avoid consecutive zeros
        flat2 = []
        prev0 = False
        for v in cleaned:
            if v == 0 and prev0:
                continue
            flat2.append(v)
            prev0 = (v == 0)
        if flat2[-1] != 0:
            flat2.append(0)
        return flat2

    def _route_cost_estimate(self, route):
        if not route:
            return 0.0
        cost = self.distance_matrix[0, route[0]]
        for i in range(len(route)-1):
            cost += self.distance_matrix[route[i], route[i+1]]
        cost += self.distance_matrix[route[-1], 0]
        return cost
