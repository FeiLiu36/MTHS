import numpy as np
import numpy.typing as npt
import concurrent.futures
from numba import njit
import traceback
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


@njit(cache=True)
def _calculate_route_cost(distmat, route):
    """计算单条路径的成本"""
    if len(route) <= 1:
        return 0.0
    cost = distmat[0, route[0]]  # 从depot到第一个客户
    for i in range(len(route) - 1):
        cost += distmat[route[i], route[i + 1]]
    cost += distmat[route[-1], 0]  # 从最后一个客户回到depot
    return cost


def _calculate_total_cost(distmat, routes):
    """计算所有路径的总成本"""
    total_cost = 0.0
    # Numba kann nicht über eine Liste von Listen iterieren, daher bleibt diese Funktion in reinem Python.
    # Sie ruft jedoch die kompilierte _calculate_route_cost-Funktion auf.
    for route in routes:
        if len(route) > 0:
            # Konvertieren der Liste in ein Numpy-Array für die Numba-Funktion
            total_cost += _calculate_route_cost(distmat, np.array(route, dtype=np.int_))
    return total_cost


def _check_capacity_constraint(demands, routes, vehicle_capacity):
    """检查容量约束是否满足"""
    for route in routes:
        route_demand = sum(demands[customer] for customer in route)
        if route_demand > vehicle_capacity:
            return False
    return True


@njit(cache=True)
def _two_opt_route(distmat: FloatArray, route: IntArray) -> tuple[float, IntArray]:
    n = len(route)
    if n < 2:
        return 0.0, route

    best_delta = 0.0
    best_i, best_j = -1, -1

    # Create a temporary route including the depot at both ends for easier calculations
    full_route = np.empty(n + 2, dtype=np.int_)
    full_route[0] = 0
    full_route[1:-1] = route
    full_route[-1] = 0

    # Iterate i from 0 to n. i=0 represents the depot->customer edge.
    for i in range(n):
        # Iterate j from i+1 to n.
        # This considers swapping edge (i, i+1) with (j, j+1)
        for j in range(i + 2, n + 1):
            node_A = full_route[i]
            node_B = full_route[i + 1]
            node_C = full_route[j]
            node_D = full_route[j + 1]

            cost_before = distmat[node_A, node_B] + distmat[node_C, node_D]
            cost_after = distmat[node_A, node_C] + distmat[node_B, node_D]
            delta = cost_after - cost_before

            if delta < best_delta:
                best_delta = delta
                # Store indices relative to the original route array
                best_i = i
                best_j = j - 1  # j in full_route corresponds to j-1 in original route

    if best_delta < -1e-9:
        # Reverse the segment from best_i to best_j in the original route
        # The segment to reverse is route[best_i : best_j + 1]
        sub_route = route[best_i: best_j + 1]
        route[best_i: best_j + 1] = sub_route[::-1]
        return best_delta, route

    return 0.0, route


@njit(cache=True)
def _relocate_customer_numba(distmat, demands, routes_array, route_lengths, vehicle_capacity):
    """
    Numba-optimized version to find the best customer relocation move.
    This function operates on NumPy arrays for performance.
    """
    best_delta = 0.0
    # Using a tuple of -1s to indicate no move found, as None is not efficient in Numba
    best_move = (-1, -1, -1, -1)
    num_routes = len(routes_array)

    # Pre-calculate initial route costs and demands to avoid recalculation
    initial_route_costs = np.zeros(num_routes, dtype=np.float64)
    initial_route_demands = np.zeros(num_routes, dtype=np.float64)
    for i in range(num_routes):
        # Create a view of the current route without padding
        current_route = routes_array[i, :route_lengths[i]]
        initial_route_costs[i] = _calculate_route_cost(distmat, current_route)
        for customer_node in current_route:
            initial_route_demands[i] += demands[customer_node]

    # Iterate over all customers in all routes
    for i in range(num_routes):
        route_i = routes_array[i, :route_lengths[i]]
        if route_lengths[i] == 0:
            continue

        for j in range(route_lengths[i]):
            customer = route_i[j]
            customer_demand = demands[customer]

            # Calculate cost change from removing the customer from route_i
            # This is much faster than recalculating the whole route cost
            prev_node_i = 0 if j == 0 else route_i[j - 1]
            next_node_i = 0 if j == route_lengths[i] - 1 else route_i[j + 1]
            cost_change_i = (distmat[prev_node_i, next_node_i] -
                             (distmat[prev_node_i, customer] + distmat[customer, next_node_i]))

            # Iterate over all possible destination routes
            for k in range(num_routes):
                if i == k:
                    continue

                # Check capacity constraint before proceeding
                if initial_route_demands[k] + customer_demand > vehicle_capacity:
                    continue

                route_k = routes_array[k, :route_lengths[k]]

                # Try inserting the customer at every possible position in route_k
                for pos in range(route_lengths[k] + 1):
                    # Calculate cost change from inserting the customer into route_k
                    prev_node_k = 0 if pos == 0 else route_k[pos - 1]
                    next_node_k = 0 if pos == route_lengths[k] else route_k[pos]
                    cost_change_k = ((distmat[prev_node_k, customer] + distmat[customer, next_node_k]) -
                                     distmat[prev_node_k, next_node_k])

                    delta = cost_change_i + cost_change_k

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j, k, pos)

    return best_delta, best_move


def _relocate_customer(distmat, demands, routes, vehicle_capacity):
    """
    Finds the best relocation move and applies it to the routes list.
    This function wraps the Numba-compiled core for easy integration.

    Returns the change in cost (delta) or 0.0 if no improvement was found.
    """
    # Convert Python list of lists to a NumPy array for Numba
    # Using a placeholder (-1) for empty spots in the array
    if not routes:
        return 0.0
    max_len = max(len(r) for r in routes) if routes else 0
    if max_len == 0:  # Handle case where all routes are empty
        return 0.0

    routes_array = np.full((len(routes), max_len), -1, dtype=np.int32)
    route_lengths = np.array([len(r) for r in routes], dtype=np.int32)

    for i, route in enumerate(routes):
        if route:  # only if route is not empty
            routes_array[i, :len(route)] = route

    # Ensure other inputs are NumPy arrays, which is good practice for Numba
    distmat_np = np.array(distmat, dtype=np.float64)
    demands_np = np.array(demands, dtype=np.float64)

    # Call the fast Numba function
    best_delta, best_move = _relocate_customer_numba(
        distmat_np, demands_np, routes_array, route_lengths, vehicle_capacity
    )

    # If an improving move was found, apply it to the original `routes` list
    if best_move[0] != -1:
        i, j, k, pos = best_move
        # The move is applied to the original mutable list of lists
        customer = routes[i].pop(j)
        routes[k].insert(pos, customer)
        return best_delta

    return 0.0


def _local_search_cvrp(distmat, demands, routes, vehicle_capacity, max_iter=1000):
    """CVRP局部搜索"""
    # Using a list of operators allows for easy extension and randomization
    #operators = [_two_opt_route, _relocate_customer]

    iteration = 0
    while iteration < max_iter:
        iteration += 1

        # --- Apply Relocate Operator ---
        # This is generally more powerful, so it's good to try it first.
        delta_relocate = _relocate_customer(distmat, demands, routes, vehicle_capacity)
        if delta_relocate < -1e-9:
            # If we found a good move, restart the search process
            # to capitalize on the new solution structure.
            iteration = 0
            continue

        # --- Apply 2-Opt Operator on each route ---
        # We only proceed to 2-opt if relocate found no improvement.
        improved_by_2opt = False
        for i, route in enumerate(routes):
            if len(route) > 2:
                route_arr = np.array(route, dtype=np.int_)
                delta_2opt, new_route_arr = _two_opt_route(distmat, route_arr)
                if delta_2opt < -1e-9:
                    routes[i] = list(new_route_arr)
                    improved_by_2opt = True

        if improved_by_2opt:
            # If 2-opt made a change, restart to try relocate again.
            iteration = 0
            continue

        # If neither operator found an improvement, break the loop.
        break

        # 尝试重新分配客户
        #delta = _relocate_customer(distmat, demands, routes, vehicle_capacity)
        # if delta < -1e-6:
        #     improved = True


def _nearest_neighbor_cvrp(distmat, demands, vehicle_capacity, start_depot=0):
    """最近邻启发式构造CVRP初始解"""
    n = distmat.shape[0]
    unvisited = set(range(1, n))  # 排除depot
    routes = []

    while unvisited:
        route = []
        current_capacity = 0
        current_node = start_depot

        while unvisited:
            # 找到最近的可行客户
            best_customer = -1
            best_distance = float('inf')

            # Diese Schleife könnte mit Numba beschleunigt werden, aber die Verwaltung des 'unvisited'-Sets
            # wäre komplex. Für den Moment belassen wir es bei Python.
            for customer in unvisited:
                if current_capacity + demands[customer] <= vehicle_capacity:
                    distance = distmat[current_node, customer]
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer

            if best_customer == -1:
                break  # 没有可行的客户，开始新路径

            route.append(best_customer)
            current_capacity += demands[best_customer]
            current_node = best_customer
            unvisited.remove(best_customer)

        if route:
            routes.append(route)

    return routes


@njit(cache=True)
def _perturbation_cvrp_numba_part(guide, penalty, route):
    """Numba-Teil der Perturbationsfunktion zur Ermittlung der besten Kante"""
    max_util = -1.0
    max_edge_u, max_edge_v = -1, -1

    if len(route) == 0:
        return max_util, max_edge_u, max_edge_v

    # depot到第一个客户
    u, v = 0, route[0]
    util = guide[u, v] / (1.0 + penalty[u, v])
    if util > max_util:
        max_util = util
        max_edge_u, max_edge_v = u, v

    # 路径内的边
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        util = guide[u, v] / (1.0 + penalty[u, v])
        if util > max_util:
            max_util = util
            max_edge_u, max_edge_v = u, v

    # 最后一个客户到depot
    u, v = route[-1], 0
    util = guide[u, v] / (1.0 + penalty[u, v])
    if util > max_util:
        max_util = util
        max_edge_u, max_edge_v = u, v

    return max_util, max_edge_u, max_edge_v


def _perturbation_cvrp(distmat, guide, penalty, demands, routes, vehicle_capacity, k, perturbation_moves=30):
    """CVRP扰动操作"""
    moves = 0

    while moves < perturbation_moves:
        # 找到具有最大utility的边进行惩罚
        overall_max_util = -1.0
        best_edge_overall = None

        for route in routes:
            if not route:
                continue
            route_arr = np.array(route, dtype=np.int_)
            util, u, v = _perturbation_cvrp_numba_part(guide, penalty, route_arr)
            if util > overall_max_util:
                overall_max_util = util
                best_edge_overall = (u, v)

        if best_edge_overall:
            penalty[best_edge_overall[0], best_edge_overall[1]] += 1.0
            penalty[best_edge_overall[1], best_edge_overall[0]] += 1.0  # Für symmetrische Matrizen
            moves += 1
        else:
            break

        # 使用惩罚后的距离矩阵进行局部搜索
        edge_weight_guided = distmat + k * penalty
        _local_search_cvrp(edge_weight_guided, demands, routes, vehicle_capacity, max_iter=10)


def _guided_local_search_cvrp(distmat, guide, demands, vehicle_capacity, perturbation_moves=30, iter_limit=1000):
    """CVRP引导局部搜索"""
    penalty = np.zeros_like(distmat)

    # 构造初始解
    best_routes = _nearest_neighbor_cvrp(distmat, demands, vehicle_capacity)
    _local_search_cvrp(distmat, demands, best_routes, vehicle_capacity)
    best_cost = _calculate_total_cost(distmat, best_routes)

    k = 0.1 * best_cost / distmat.shape[0] if distmat.shape[0] > 0 else 0.1
    current_routes = [route.copy() for route in best_routes]

    for iteration in range(iter_limit):
        #print(iteration) # Auskommentiert für saubere Ausgabe
        _perturbation_cvrp(distmat, guide, penalty, demands, current_routes, vehicle_capacity, k, perturbation_moves)
        _local_search_cvrp(distmat, demands, current_routes, vehicle_capacity)

        current_cost = _calculate_total_cost(distmat, current_routes)
        if current_cost < best_cost:
            best_routes = [route.copy() for route in current_routes]
            best_cost = current_cost
        #current_routes = best_routes

    return best_routes


def update_edge_distance_cvrp(coordinates: np.ndarray, edge_distance: np.ndarray, demands: np.ndarray,
                              vehicle_capacity: int) -> np.ndarray: 


    """
    Designs a novel heuristic to update the distance matrix.

    Args:
    coordinates: Coordinate matrix with depot at first position.
    edge_distance: Euclidean distance matrix of each node pairs.
    demands: Node demand vector. 
    vehicle_capacity: Maximum vehicle capacity
    
    Return:
    updated_edge_distance: A matrix of the updated distances.
    """
    import numpy as np

    rng = np.random.default_rng()
    eps = 1e-12

    D = np.array(edge_distance, dtype=float)
    coords = np.array(coordinates, dtype=float)
    demands = np.array(demands, dtype=float).flatten()
    n = D.shape[0]
    if n == 0:
        return D.copy()
    if demands.shape[0] != n:
        raise ValueError("demands length must match number of nodes in edge_distance")
    demands = demands.copy()
    demands[0] = 0.0  # depot demand zero

    capacity = float(max(1.0, vehicle_capacity))
    if n <= 1:
        avg_dist = 1.0
    else:
        triu = D[np.triu_indices(n, k=1)]
        avg_dist = float(max(np.mean(triu), eps))

    # Pairwise demand sums and ratios
    sum_dem = demands[:, None] + demands[None, :]
    cap_ratio = sum_dem / (capacity + eps)

    # 1) Base affinity: geometry softened by capacity logistic (blend of both)
    tau = 1.32
    A_base = np.exp(-tau * (D / (avg_dist + eps)))

    s_cap = 11.0 / (capacity + eps)
    cap_center = 0.82
    cap_compat = 1.0 / (1.0 + np.exp(s_cap * (cap_ratio - cap_center)))  # near 1 if comfortably below center
    # mix a small floor so very low affinities don't vanish
    A = A_base * (0.14 + 0.86 * cap_compat)
    np.fill_diagonal(A, A.diagonal() + 1e-9)

    # Row-normalize and symmetrize to form diffusion matrix
    row_sums = A.sum(axis=1, keepdims=True)
    P = A / (row_sums + eps)
    M = 0.5 * (P + P.T)

    # 2) Spectral / diffusion embedding (robust scaling)
    k_eig = min(6, max(1, n - 1))
    try:
        eigvals_all, eigvecs_all = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        eigvals_all, eigvecs_all = np.linalg.eigh(M + 1e-12 * np.eye(n))
    idx = np.argsort(eigvals_all)[::-1][:k_eig]
    eigvals = eigvals_all[idx]
    eigvecs = eigvecs_all[:, idx]
    eigvals_pos = np.clip(eigvals, 0.0, None)
    if eigvals_pos.size == 0:
        eigvals_pos = np.ones(k_eig)
    max_ev = eigvals_pos.max() if eigvals_pos.size > 0 else 1.0
    lam_norm = eigvals_pos / (max_ev + eps)

    # emphasize mid-high modes but keep some diffusion time
    t_diff = 2.5
    lam_t = np.power(lam_norm, t_diff)
    s_heat = 1.9
    lam_scale = (1.0 / (1.0 + np.exp(-s_heat * (lam_t - 0.30)))) * np.sqrt(lam_t + eps)
    lam_scale = np.clip(lam_scale, 0.0, 1.0)
    embedding = eigvecs * lam_scale[None, :]

    # fused embedding distances: cos, euclid, l1 (weighted robust fusion)
    emb_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    emb_norm_safe = np.maximum(emb_norm, eps)
    cos_sim = (embedding @ embedding.T) / (emb_norm_safe @ emb_norm_safe.T + eps)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    cos_dist = 1.0 - cos_sim

    sq_norms = np.sum(embedding * embedding, axis=1, keepdims=True)
    cross = embedding @ embedding.T
    diff_sq = np.maximum(sq_norms + sq_norms.T - 2.0 * cross, 0.0)
    diff_euc = np.sqrt(diff_sq + 1e-16)

    emb_abs = np.abs(embedding[:, None, :] - embedding[None, :, :])
    diff_l1 = np.sum(emb_abs, axis=2)

    def safe_norm(X):
        mx = X.max()
        return X / (mx + eps) if mx > eps else X.copy()

    cos_n = safe_norm(cos_dist)
    euc_n = safe_norm(diff_euc)
    l1_n = safe_norm(diff_l1)
    # slightly favor euclidean but keep angular influence
    diff_norm = 0.25 * cos_n + 0.55 * euc_n + 0.20 * l1_n
    diff_norm = np.clip(diff_norm, 0.0, 1.0)
    spectral_boost = 1.0 + float(np.clip(eigvals_pos.max(), 0.0, 1.0)) * 0.36

    # 3) Pair similarity: fill closeness + complementarity
    target_fill = 0.62 * capacity
    fill_sigma = 0.32 * capacity
    fill_closeness = np.exp(-0.5 * ((sum_dem - target_fill) / (fill_sigma + eps)) ** 2)
    abs_diff = np.abs(demands[:, None] - demands[None, :])
    complement = 1.0 - np.clip(abs_diff / (capacity + eps), 0.0, 1.0)
    # weight toward complement when sum reasonable, else fill_closeness
    pair_sim = 0.55 * fill_closeness + 0.45 * complement
    pair_sim = np.clip(pair_sim, 0.0, 1.0)
    # logistic stretch for discrimination
    sim_s = 8.5
    sim_raw = 1.0 / (1.0 + np.exp(-sim_s * (pair_sim - 0.48)))
    # map to helpful multiplier domain (we will use it to shrink multiplicatively)
    sim_shrink = 0.42 + sim_raw * (1.42 - 0.42)  # in [0.42,1.42], higher -> more favorable
    sim_shrink = np.clip(sim_shrink, 0.42, 1.42)

    # 4) Depot proximity & heaviness
    depot = coords[0]
    rel = coords - depot
    radii = np.hypot(rel[:, 0], rel[:, 1])
    avg_rad = max(np.mean(radii), eps)
    depot_prox = np.exp(- (radii / (1.02 * avg_rad + eps)) ** 1.12)
    heaviness = np.clip(demands / (capacity + eps), 0.0, 1.0) ** 1.06
    depot_score_vec = depot_prox * heaviness
    depot_pair = np.maximum(depot_score_vec[:, None], depot_score_vec[None, :])
    # allow up to 68% reduction for strong depot-touch heavy/prox pairs but clipped conservatively
    depot_scale = np.ones((n, n), dtype=float)
    depot_scale[0, :] = 1.0 - 0.68 * depot_score_vec
    depot_scale[:, 0] = 1.0 - 0.68 * depot_score_vec
    depot_scale = np.clip(depot_scale, 0.30, 1.08)
    depot_scale[0, 0] = 1.0

    # 5) Capacity penalty / slack bonus (blend smooth poly + exponential tail)
    over_ratio = np.maximum(0.0, cap_ratio - 1.0)
    # mild polynomial growth immediately past capacity
    poly_part = 1.0 + 4.8 * (over_ratio ** 2.1)
    # exponential tail for larger breaches with threshold softening
    cap_kappa = 7.2
    cap_tail = np.exp(cap_kappa * np.maximum(0.0, cap_ratio - 1.06))
    sig_part = cap_tail
    cap_penalty = poly_part * sig_part
    # slack bonus for comfortably below capacity
    slack_ratio = np.clip((1.0 - cap_ratio) / 1.0, 0.0, 1.0)
    slack_bonus = 1.0 - 0.32 * (slack_ratio ** 0.78)
    cap_factor = np.where(cap_ratio <= 1.0, slack_bonus, cap_penalty)
    cap_factor = np.clip(cap_factor, 0.08, 1e9)

    # 6) Shrink multiplier: combine spectral and sim_shrink (favor spectrally-close & compatible loads)
    diff_strength = 0.94 * spectral_boost
    shrink_base = 0.14 + 1.48 * ((1.0 - diff_norm) ** 1.45)
    shrink_multiplier = shrink_base * (0.50 + 0.95 * (sim_shrink / 1.42)) * (1.0 + 0.12 * diff_strength)
    shrink_multiplier = np.clip(shrink_multiplier, 0.10, 2.05)

    # 7) Combine multiplicatively and apply depot & capacity
    # we interpret sim_shrink >1 as favorable; invert mapping to produce multiplicative shrink <1 for good pairs:
    # convert sim_shrink so higher values reduce distance: sim_factor = 1.0 / sim_shrink
    sim_factor = 1.0 / (sim_shrink + eps)
    updated = D * (shrink_multiplier * sim_factor * cap_factor * depot_scale)

    # 8) Diffusion-informed additive penalty & baseline to discourage risky shortcuts
    add_scale = avg_dist * 0.095
    add_penalty = add_scale * (diff_norm ** 1.08) * (1.0 + 3.8 * (over_ratio ** 1.16))
    baseline = avg_dist * 0.0055
    updated = updated + add_penalty + baseline

    # 9) Moderated hard inflations for infeasibility (graded)
    mid_mask = (sum_dem > 1.05 * capacity)
    if np.any(mid_mask):
        updated[mid_mask] += avg_dist * 68.0
    extreme_mask = (sum_dem > 1.28 * capacity)
    if np.any(extreme_mask):
        updated[extreme_mask] += avg_dist * 940.0

    # 10) Small symmetric multiplicative exploration noise (log-normal style)
    noise_sigma = 0.048
    noise = np.exp(rng.normal(loc=0.0, scale=noise_sigma, size=(n, n)))
    low_clip = np.exp(-0.30)
    high_clip = np.exp(0.30)
    noise = np.clip(noise, low_clip, high_clip)
    noise = 0.5 * (noise + noise.T)
    updated = updated * noise

    # Finalize: symmetrize, non-negative, zero diagonal
    updated = 0.5 * (updated + updated.T)
    np.fill_diagonal(updated, 0.0)
    updated = np.maximum(updated, 0.0)

    return updated



class CVRPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: list, vehicle_capacity):
        """
        Initialize the CVRP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each node, including the depot.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between nodes.
            demands: List of integers representing the demand of each node (first node is typically the depot with zero demand).
            vehicle_capacity: Integer representing the maximum capacity of each vehicle.
        """
        self.coordinates = np.array(coordinates)
        self.distance_matrix = np.array(distance_matrix)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        print(vehicle_capacity)

    def solve(self):

        try:

            guidance_matrix = update_edge_distance_cvrp(self.coordinates, self.distance_matrix.copy(), self.demands.copy(), self.vehicle_capacity)

            distmat_f64 = self.distance_matrix.astype(np.float64)
            guide_f64 = guidance_matrix.astype(np.float64)
            demands_i64 = self.demands
            routes = _guided_local_search_cvrp(
                distmat=distmat_f64,
                guide=guide_f64,
                demands=demands_i64,
                vehicle_capacity=self.vehicle_capacity ,
                perturbation_moves=30,
                iter_limit=1000
            )
            #print(routes)
            #print(self.demands)
            #print(self.vehicle_capacity)
            return routes
        except Exception as e:
            traceback.print_exc()
