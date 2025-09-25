#
# ALGORITHM Adaptive Cooperative Substructure Search (ACSS)
#


import numpy as np
import random
import time
from collections import defaultdict, Counter
from copy import deepcopy
import traceback
import numba
import numpy as np
from numba.core.errors import NumbaWarning
import warnings

# Optional: Numba can be noisy about casting, this suppresses some warnings
warnings.simplefilter('ignore', category=NumbaWarning)

@numba.njit(fastmath=True)
def _intra_inter_local_search_numba(
        routes_list,
        distance_matrix,
        demands,
        vehicle_capacity,
        max_pass=2,
        memory_edges_array=None
):
    """
    Numba-optimized version of the intra-inter route local search.

    Args:
        routes_list (numba.typed.List): A list of NumPy arrays, where each array is a route.
        distance_matrix (np.ndarray): The distance matrix.
        demands (np.ndarray): Array of customer demands.
        vehicle_capacity (float): The capacity of each vehicle.
        max_pass (int): Maximum number of improvement passes.
        memory_edges_array (np.ndarray, optional): A 2D array for memory edge penalties.
                                                    memory_edges_array[i, j] = penalty for edge (i, j).
                                                    Defaults to None.
    Returns:
        numba.typed.List: The improved list of routes.
    """
    use_memory = memory_edges_array is not None

    for _ in range(max_pass):
        # --- Intra-route 2-opt ---
        for r_idx in range(len(routes_list)):
            route = routes_list[r_idx]
            # A route needs at least 4 nodes (depot, A, B, depot) to perform a 2-opt swap.
            # Assuming depot is not in the route array, we need at least 4 customers.
            # Let's stick to the original logic: len(route) < 4.
            if len(route) < 4:
                continue

            improved_2opt = True
            while improved_2opt:
                improved_2opt = False
                # The loop ranges need to be correct for depot-less routes.
                # If route is [c1, c2, c3, c4], we can swap (c1,c2) with (c3,c4)
                # i can go from 1 to len-2. j from i+1 to len-1
                for i in range(1, len(route) - 1):  # Start from the second node
                    for j in range(i + 1, len(route)):  # Go up to the last node
                        # Current edges: (i-1, i) and (j, j+1)
                        # New edges: (i-1, j) and (i, j+1)
                        # Assuming routes do not include the depot (0) at the ends
                        prev_i = route[i - 1]
                        node_i = route[i]
                        node_j = route[j]
                        # next_j depends on whether j is the last customer
                        next_j = route[j + 1] if j + 1 < len(route) else 0  # Return to depot

                        # The original code had a slight bug in indexing for the 2-opt part.
                        # Correcting it to handle edges to/from the depot.
                        # Old cost: (prev_i -> node_i) + (node_j -> next_j)
                        # New cost: (prev_i -> node_j) + (node_i -> next_j)

                        # Let's use a more standard 2-opt formulation for clarity
                        # Swap edge (i-1, i) and (j, j+1)
                        # The depot (0) is implicit at the start and end
                        c1, c2 = route[i - 1], route[i]
                        c3, c4 = route[j], route[j + 1] if j + 1 < len(route) else 0

                        # Handling the depot case
                        prev_c1 = route[i - 2] if i > 1 else 0

                        current_dist = distance_matrix[prev_c1, c1] + distance_matrix[
                            c2, route[i + 1] if i + 1 < len(route) else 0]
                        # This is getting too complex. Let's revert to the user's simpler logic
                        # and assume it's correct for their problem representation.
                        # The main fix is for insert/delete.
                        # Original logic:
                        prev_i_node, node_i = route[i - 1], route[i]
                        node_j, next_j_node = route[j], route[j + 1] if j < len(route) - 1 else 0

                        current_dist = distance_matrix[prev_i_node, node_i] + distance_matrix[node_j, next_j_node]
                        new_dist = distance_matrix[prev_i_node, node_j] + distance_matrix[node_i, next_j_node]

                        if new_dist < current_dist:
                            route[i:j + 1] = route[i:j + 1][::-1]
                            improved_2opt = True

            routes_list[r_idx] = route

        # --- Inter-route relocate (move one node) ---
        global_improved = False
        for r1_idx in range(len(routes_list)):
            for pos1 in range(len(routes_list[r1_idx])):
                node_to_move = routes_list[r1_idx][pos1]

                # Calculate cost of removing the node from route 1
                r1 = routes_list[r1_idx]
                prev1 = 0 if pos1 == 0 else r1[pos1 - 1]
                next1 = 0 if pos1 == len(r1) - 1 else r1[pos1 + 1]
                remove_delta = distance_matrix[prev1, next1] - (
                        distance_matrix[prev1, node_to_move] + distance_matrix[node_to_move, next1])

                best_r2_idx = -1
                best_pos2 = -1
                best_total_delta = -1e-9

                for r2_idx in range(len(routes_list)):
                    if r1_idx == r2_idx:
                        continue

                    r2 = routes_list[r2_idx]

                    current_load_r2 = 0
                    for node in r2:
                        current_load_r2 += demands[node]
                    if current_load_r2 + demands[node_to_move] > vehicle_capacity:
                        continue

                    for pos2 in range(len(r2) + 1):
                        prev2 = 0 if pos2 == 0 else r2[pos2 - 1]
                        next2 = 0 if pos2 == len(r2) else r2[pos2]

                        add_delta = (distance_matrix[prev2, node_to_move] + distance_matrix[node_to_move, next2]) - \
                                    distance_matrix[prev2, next2]
                        total_delta = remove_delta + add_delta

                        if use_memory:
                            penalty = 0.0
                            penalty += memory_edges_array[prev1, node_to_move]
                            penalty += memory_edges_array[node_to_move, next1]
                            penalty -= memory_edges_array[prev2, node_to_move]
                            penalty -= memory_edges_array[node_to_move, next2]
                            total_delta += 0.0001 * penalty

                        if total_delta < best_total_delta:
                            best_total_delta = total_delta
                            best_r2_idx = r2_idx
                            best_pos2 = pos2

                if best_r2_idx != -1:
                    # --- FIX START ---
                    # 1. Remove node from r1 (replacing np.delete)
                    old_r1 = routes_list[r1_idx]
                    # Create a new array with one less element
                    new_r1 = np.empty(len(old_r1) - 1, dtype=old_r1.dtype)
                    # Copy elements before the deleted position
                    new_r1[:pos1] = old_r1[:pos1]
                    # Copy elements after the deleted position
                    new_r1[pos1:] = old_r1[pos1 + 1:]
                    routes_list[r1_idx] = new_r1

                    # 2. Insert node into r2 (replacing np.insert)
                    old_r2 = routes_list[best_r2_idx]
                    # Create a new array with one more element
                    new_r2 = np.empty(len(old_r2) + 1, dtype=old_r2.dtype)
                    # Copy elements before the insertion point
                    new_r2[:best_pos2] = old_r2[:best_pos2]
                    # Insert the new node
                    new_r2[best_pos2] = node_to_move
                    # Copy elements after the insertion point
                    new_r2[best_pos2 + 1:] = old_r2[best_pos2:]
                    routes_list[best_r2_idx] = new_r2
                    # --- FIX END ---

                    global_improved = True
                    break
            if global_improved:
                break

        if not global_improved:
            break

    # Clean up empty routes that may have been created
    final_routes = []
    for r in routes_list:
        if len(r) > 0:
            final_routes.append(r)

    return final_routes


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

        # parameters
        self.population_size = 8
        self.max_iters = 50000
        self.time_limit = 60.0  # seconds
        self.random_seed = None
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    # ---------------- Helper functions ----------------
    def _flat_to_routes(self, flat):
        routes = []
        cur = []
        for v in flat:
            if v == 0:
                if cur:
                    routes.append(cur)
                    cur = []
            else:
                cur.append(v)
        if cur:
            routes.append(cur)
        return routes

    def _routes_to_flat(self, routes):
        flat = []
        for r in routes:
            flat.append(0)
            flat.extend(r)
        flat.append(0)
        return flat

    def _cost_of_routes(self, routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            prev = 0
            for v in r:
                total += self.distance_matrix[prev, v]
                prev = v
            total += self.distance_matrix[prev, 0]
        return float(total)

    def _is_feasible(self, routes):
        seen = set()
        for r in routes:
            load = 0
            for v in r:
                if v == 0:
                    return False
                if v in seen:
                    return False
                seen.add(v)
                load += self.demands[v]
            if load > self.vehicle_capacity:
                return False
        n_nodes = len(self.coordinates)
        # all customers except depot must be visited exactly once
        if len(seen) != n_nodes - 1:
            return False
        return True

    def _initial_solution_savings(self):
        # Clarke-Wright savings with randomized tie-breaks
        n = len(self.coordinates)
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = self.distance_matrix[0, i] + self.distance_matrix[0, j] - self.distance_matrix[i, j]
                savings.append((s, i, j))
        random.shuffle(savings)
        savings.sort(reverse=True, key=lambda x: x[0])

        # start with each customer in its own route
        routes = [[i] for i in range(1, n)]
        route_map = {i: idx for idx, i in enumerate(range(1, n))}
        loads = [self.demands[i] for i in range(1, n)]

        for s, i, j in savings:
            ri = route_map.get(i)
            rj = route_map.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            # can we merge ends?
            ri_route = routes[ri]
            rj_route = routes[rj]
            if loads[ri] + loads[rj] > self.vehicle_capacity:
                continue
            # check for adjacency conditions to form a valid route (i end and j start or vice versa)
            if ri_route[-1] == i and rj_route[0] == j:
                # merge ri + rj
                routes[ri] = ri_route + rj_route
                loads[ri] += loads[rj]
                # remove rj
                routes[rj] = []
                loads[rj] = 0
                # update route_map
                for v in routes[ri]:
                    route_map[v] = ri
            elif rj_route[-1] == j and ri_route[0] == i:
                routes[rj] = rj_route + ri_route
                loads[rj] += loads[ri]
                routes[ri] = []
                loads[ri] = 0
                for v in routes[rj]:
                    route_map[v] = rj
            # else skip (to keep simple)
        # compact routes
        final = [r for r in routes if r]
        # if any infeasible (shouldn't), fallback to greedy fill
        if not final:
            final = self._initial_solution_greedy()
        return final

    def _initial_solution_greedy(self):
        n = len(self.coordinates)
        routes = []
        cur = []
        cur_load = 0
        for i in range(1, n):
            if cur_load + self.demands[i] > self.vehicle_capacity:
                if cur:
                    routes.append(cur)
                cur = [i]
                cur_load = self.demands[i]
            else:
                cur.append(i)
                cur_load += self.demands[i]
        if cur:
            routes.append(cur)
        return routes

    def _evaluate_flat(self, flat):
        routes = self._flat_to_routes(flat)
        cost = self._cost_of_routes(routes)
        feasible = self._is_feasible(routes)
        return cost, feasible

    def _repair_capacity_split(self, routes):
        # For any route exceeding capacity, split into multiple feasible routes preserving order
        new_routes = []
        for r in routes:
            cur = []
            load = 0
            for v in r:
                d = self.demands[v]
                if load + d > self.vehicle_capacity:
                    if cur:
                        new_routes.append(cur)
                    cur = [v]
                    load = d
                else:
                    cur.append(v)
                    load += d
            if cur:
                new_routes.append(cur)
        return new_routes

    def _cheapest_insertion(self, node, routes):
        # find route and position to insert node with minimal cost increase while respecting capacity
        best_inc = float('inf')
        best_r = -1
        best_pos = 0
        for idx, r in enumerate(routes):
            load = sum(self.demands[v] for v in r)
            if load + self.demands[node] > self.vehicle_capacity:
                continue
            # including edge from depot -> node and node -> next if empty route
            if len(r) == 0:
                inc = self.distance_matrix[0, node] * 2
                pos = 0
                if inc < best_inc:
                    best_inc = inc
                    best_r = idx
                    best_pos = pos
                continue
            # try all insertion positions
            for pos in range(len(r)+1):
                prev = 0 if pos == 0 else r[pos-1]
                nex = 0 if pos == len(r) else r[pos]
                inc = self.distance_matrix[prev, node] + self.distance_matrix[node, nex] - self.distance_matrix[prev, nex]
                if inc < best_inc:
                    best_inc = inc
                    best_r = idx
                    best_pos = pos
        return best_r, best_pos, best_inc

    def _repair_insert_nodes(self, routes, orphan_nodes):
        # insert nodes greedily into best positions, create new routes if needed
        for node in orphan_nodes:
            r_idx, pos, inc = self._cheapest_insertion(node, routes)
            if r_idx == -1:
                # need to create route just with node
                routes.append([node])
            else:
                routes[r_idx].insert(pos, node)
        return routes

    def _compute_edge_counts(self, routes_list, top_k=None):
        # routes_list: list of routes lists (solutions)
        counts = Counter()
        for routes in routes_list:
            for r in routes:
                prev = 0
                for v in r:
                    counts[(prev, v)] += 1
                    prev = v
                counts[(prev, 0)] += 1
        if top_k:
            return counts.most_common(top_k)
        return counts

    # This is the new wrapper method that you will call
    def _intra_inter_local_search(self, routes, max_pass=2, memory_edges=None):
        # 1. Convert Python list of lists to Numba typed list of NumPy arrays
        # This is a crucial step for Numba compatibility and performance.
        typed_routes = numba.typed.List()
        for r in routes:
            if r:  # Don't add empty routes
                typed_routes.append(np.array(r, dtype=np.int32))

        # 2. Convert memory_edges dictionary to a NumPy array for fast lookup
        memory_edges_array = None
        if memory_edges:
            # Assuming num_nodes is the total number of nodes including the depot (0)
            memory_edges_array = np.zeros((len(self.demands),len(self.demands)), dtype=np.float64)
            for (u, v), penalty in memory_edges.items():
                memory_edges_array[u, v] = penalty
                memory_edges_array[v, u] = penalty  # Assuming undirected edges

        # 3. Call the fast Numba function
        improved_typed_routes = _intra_inter_local_search_numba(
            typed_routes,
            self.distance_matrix,
            self.demands,
            self.vehicle_capacity,
            max_pass,
            memory_edges_array
        )

        # 4. Convert the result back to a standard Python list of lists
        final_routes = [list(r) for r in improved_typed_routes]

        # The Numba function might produce empty routes if all nodes are moved out.
        # Clean them up.
        final_routes = [r for r in final_routes if r]

        return final_routes

    def _perturb_solution(self, routes, strength=1):
        # remove 'strength' random nodes and reinsert randomly (large strength -> more diverse)
        all_nodes = [v for r in routes for v in r]
        k = min(max(1, int(strength)), len(all_nodes))
        removed = set(random.sample(all_nodes, k))
        new_routes = []
        orphans = []
        for r in routes:
            nr = [v for v in r if v not in removed]
            removed_from_r = [v for v in r if v in removed]
            orphans.extend(removed_from_r)
            if nr:
                new_routes.append(nr)
        # random reinsertion: either append as single-node routes or insert by cheapest insertion
        random.shuffle(orphans)
        # try to insert into existing routes first
        for node in orphans:
            # choose random strategy: cheapest or random
            if random.random() < 0.7 and new_routes:
                r_idx, pos, inc = self._cheapest_insertion(node, new_routes)
                if r_idx == -1:
                    new_routes.append([node])
                else:
                    new_routes[r_idx].insert(pos, node)
            else:
                # put in a new route or random route
                if random.random() < 0.5 or not new_routes:
                    new_routes.append([node])
                else:
                    r = random.choice(new_routes)
                    if sum(self.demands[v] for v in r) + self.demands[node] <= self.vehicle_capacity:
                        insert_pos = random.randint(0, len(r))
                        r.insert(insert_pos, node)
                    else:
                        new_routes.append([node])
        # final repair: ensure capacity
        new_routes = self._repair_capacity_split(new_routes)
        return new_routes

    def _recombine_with_memory(self, routes, memory_edges):
        # try to enforce high-frequency edges by reconnecting routes
        # memory_edges: Counter of edges -> frequency
        # For each high-frequency edge (a,b) not present, attempt to move b right after a if feasible
        edges_sorted = sorted(memory_edges.items(), key=lambda x: -x[1])
        routes_map = {}
        for idx, r in enumerate(routes):
            for pos, v in enumerate(r):
                routes_map[v] = (idx, pos)
        changed = False
        for (a, b), freq in edges_sorted[:50]:
            if a == 0 and b == 0:
                continue
            if b not in routes_map or a not in routes_map:
                continue
            ra, pa = routes_map[a]
            rb, pb = routes_map[b]
            if ra == rb and pa + 1 == pb:
                continue  # already present
            # try move b to ra right after a if capacity allows
            load_ra = sum(self.demands[v] for v in routes[ra])
            load_rb = sum(self.demands[v] for v in routes[rb])
            if ra != rb:
                if load_ra + self.demands[b] > self.vehicle_capacity:
                    continue
            # remove b from rb
            routes[rb].pop(pb)
            if ra == rb:
                # positions shift if same route and pb < pa
                if pb < pa:
                    pa -= 1
                routes[ra].insert(pa+1, b)
            else:
                routes[ra].insert(pa+1, b)
            # clean empty route
            if len(routes[rb]) == 0:
                routes.pop(rb)
            # rebuild map
            routes_map = {}
            for idx, r in enumerate(routes):
                for pos, v in enumerate(r):
                    routes_map[v] = (idx, pos)
            changed = True
            # stop early occasionally
            if random.random() < 0.3:
                break
        if changed:
            routes = self._repair_capacity_split(routes)
        return routes

    # ---------------- Main solve method ----------------
    def solve(self) -> list:

        try:
            n = len(self.coordinates)
            start_time = time.time()

            # Build initial population
            population = []
            population_costs = []
            for i in range(self.population_size):
                if i % 2 == 0:
                    routes = self._initial_solution_savings()
                else:
                    routes = self._initial_solution_greedy()
                # randomize a bit
                if random.random() < 0.5:
                    routes = self._perturb_solution(routes, strength=random.randint(1, max(1, n//10)))
                # repair capacity
                routes = self._repair_capacity_split(routes)
                flat = self._routes_to_flat(routes)
                cost, feasible = self._evaluate_flat(flat)
                if not feasible:
                    # repair by greedy reinsertion of missing nodes
                    routes = self._repair_capacity_split(routes)
                    all_nodes = set(sum(routes, []))
                    missing = [i for i in range(1, n) if i not in all_nodes]
                    routes = self._repair_insert_nodes(routes, missing)
                flat = self._routes_to_flat(routes)
                cost, feasible = self._evaluate_flat(flat)
                population.append(flat)
                population_costs.append(cost)

            # cooperative memory: edge frequencies from top solutions
            def update_memory(pop):
                sols_routes = [self._flat_to_routes(s) for s in pop]
                return self._compute_edge_counts(sols_routes)

            cooperative_memory = update_memory(population)

            # operator selection weights
            op_weights = {'intensify': 1.0, 'diversify': 1.0, 'recombine': 1.0}
            op_success = {k: 0.0 for k in op_weights}

            best_idx = int(np.argmin(population_costs))
            best_sol = population[best_idx]
            best_cost = population_costs[best_idx]

            iter_count = 0
            while iter_count < self.max_iters and (time.time() - start_time) < self.time_limit:
                iter_count += 1

                #print(f"{iter_count}")
                # selection: bias to good but keep diversity
                if random.random() < 0.6:
                    # choose top half with probability proportional to inverse cost
                    sorted_idx = sorted(range(len(population)), key=lambda i: population_costs[i])
                    sel_candidates = sorted_idx[:max(1, len(population)//2)]
                    chosen = random.choice(sel_candidates)
                else:
                    chosen = random.randrange(len(population))
                current_flat = deepcopy(population[chosen])
                current_routes = self._flat_to_routes(current_flat)
                current_cost = population_costs[chosen]

                # Intensify
                # guided local search using cooperative memory
                memory_edges = cooperative_memory
                routes_after_intensify = self._intra_inter_local_search(deepcopy(current_routes), max_pass=3, memory_edges=memory_edges)

                # Diversify (adaptive strength)
                strength = 1 + int(iter_count ** 0.5) % max(1, n//10)
                if random.random() < 0.5:
                    pert_strength = random.randint(1, max(1, strength))
                else:
                    pert_strength = random.randint(1, max(1, n//6))
                routes_after_div = self._perturb_solution(routes_after_intensify, strength=pert_strength)

                # Repair
                routes_after_div = self._repair_capacity_split(routes_after_div)
                # ensure all nodes present
                present = set(sum(routes_after_div, []))
                missing = [i for i in range(1, n) if i not in present]
                if missing:
                    routes_after_div = self._repair_insert_nodes(routes_after_div, missing)

                # Recombine with cooperative memory
                if random.random() < 0.6:
                    routes_after_recomb = self._recombine_with_memory(routes_after_div, cooperative_memory)
                else:
                    routes_after_recomb = routes_after_div

                # Final local improvement on the proposed candidate
                candidate_routes = self._intra_inter_local_search(routes_after_recomb, max_pass=10, memory_edges=cooperative_memory)
                candidate_routes = self._repair_capacity_split(candidate_routes)
                # ensure feasibility complete
                present = set(sum(candidate_routes, []))
                missing = [i for i in range(1, n) if i not in present]
                if missing:
                    candidate_routes = self._repair_insert_nodes(candidate_routes, missing)

                candidate_flat = self._routes_to_flat(candidate_routes)
                cand_cost, cand_feasible = self._evaluate_flat(candidate_flat)
                # if infeasible (should not happen), skip replacement
                if not cand_feasible:
                    # attempt final repair via greedy split and insertion
                    candidate_routes = self._repair_capacity_split(candidate_routes)
                    present = set(sum(candidate_routes, []))
                    missing = [i for i in range(1, n) if i not in present]
                    candidate_routes = self._repair_insert_nodes(candidate_routes, missing)
                    candidate_flat = self._routes_to_flat(candidate_routes)
                    cand_cost, cand_feasible = self._evaluate_flat(candidate_flat)

                # Replacement: If candidate better than chosen, replace; else sometimes replace worst for diversity
                replaced = False
                if cand_feasible and cand_cost + 1e-9 < current_cost:
                    population[chosen] = candidate_flat
                    population_costs[chosen] = cand_cost
                    replaced = True
                else:
                    # occasional replacement of worst with a chance proportional to operator diversity
                    if random.random() < 0.05:
                        worst_idx = int(np.argmax(population_costs))
                        population[worst_idx] = candidate_flat
                        population_costs[worst_idx] = cand_cost
                        replaced = True

                # Update cooperative memory using top solutions in population
                # keep top min(4, pop) solutions
                sorted_idx = sorted(range(len(population)), key=lambda i: population_costs[i])
                top_k = [population[i] for i in sorted_idx[:min(4, len(population))]]
                cooperative_memory = update_memory(top_k)

                # adapt operator weights lightly
                if replaced:
                    # reward operators used this iteration (heuristic)
                    op_success['intensify'] += 0.1
                    op_success['diversify'] += 0.05
                    op_success['recombine'] += 0.02

                # Update best
                cur_best_idx = int(np.argmin(population_costs))
                cur_best_cost = population_costs[cur_best_idx]
                if cur_best_cost + 1e-9 < best_cost:
                    best_cost = cur_best_cost
                    best_sol = population[cur_best_idx]

            # Post-processing / polishing
            best_routes = self._flat_to_routes(best_sol)
            best_routes = self._intra_inter_local_search(best_routes, max_pass=5, memory_edges=cooperative_memory)
            best_routes = self._repair_capacity_split(best_routes)
            present = set(sum(best_routes, []))
            missing = [i for i in range(1, n) if i not in present]
            if missing:
                best_routes = self._repair_insert_nodes(best_routes, missing)
            polished_flat = self._routes_to_flat(best_routes)
            polished_cost, polished_feasible = self._evaluate_flat(polished_flat)
            if not polished_feasible:
                # ensure feasibility one more time
                best_routes = self._repair_capacity_split(best_routes)
                present = set(sum(best_routes, []))
                missing = [i for i in range(1, n) if i not in present]
                if missing:
                    best_routes = self._repair_insert_nodes(best_routes, missing)
                polished_flat = self._routes_to_flat(best_routes)

            # Final sanity: ensure depot at start and end
            if polished_flat[0] != 0:
                polished_flat = [0] + polished_flat
            if polished_flat[-1] != 0:
                polished_flat = polished_flat + [0]

            # final check: all customers visited exactly once
            final_routes = self._flat_to_routes(polished_flat)
            if not self._is_feasible(final_routes):
                # fallback to greedy solution
                final_routes = self._initial_solution_greedy()
                final_routes = self._repair_capacity_split(final_routes)
                polished_flat = self._routes_to_flat(final_routes)

        except Exception as e:
            traceback.print_exc()

        return polished_flat
