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
    if not (isinstance(coordinates, np.ndarray) and isinstance(edge_distance, np.ndarray) and isinstance(demands, np.ndarray)):
        raise TypeError("coordinates, edge_distance, demands must be numpy arrays")
    if edge_distance.ndim != 2 or edge_distance.shape[0] != edge_distance.shape[1]:
        raise ValueError("edge_distance must be a square matrix")
    n = int(edge_distance.shape[0])
    if coordinates.shape[0] != n:
        raise ValueError("coordinates must have same number of rows as edge_distance")
    if demands.shape[0] != n:
        raise ValueError("demands length must match number of nodes")
    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity must be positive")

    # Trivial cases
    if n == 0:
        return edge_distance.astype(float).copy()
    # Work with float symmetric base matrix
    base = edge_distance.astype(float).copy()
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)
    if n == 1:
        return base

    eps = 1e-12
    cap = float(vehicle_capacity)
    demands = demands.astype(float).copy()

    # Global demand metrics
    total_demand = float(np.sum(demands))
    est_routes = max(1.0, float(np.ceil(total_demand / (cap + eps))))
    total_cap_est = est_routes * cap
    slack_ratio = np.clip((total_cap_est - total_demand) / (total_cap_est + eps), 0.0, 1.0)  # 0 tight, 1 loose
    utilization = np.clip(total_demand / (total_cap_est + eps), 0.0, 2.0)

    # Combined demands and normalized variants
    combined = demands[:, None] + demands[None, :]
    norm_combined_cap = combined / (cap + eps)  # <=1 => pair fits on one vehicle

    # Demand distribution characteristics
    if n > 1:
        non_depot_demands = demands[1:]
        med_d = float(np.median(np.abs(non_depot_demands)) if non_depot_demands.size > 0 else (np.median(np.abs(demands)) + eps))
    else:
        med_d = float(np.median(np.abs(demands)) + eps)
    med_d = max(med_d, eps)
    demand_cv = float(np.std(demands) / (np.mean(np.abs(demands)) + eps))

    # Representative distance for scale normalization
    offdiag = base[~np.eye(n, dtype=bool)]
    positive_off = offdiag[offdiag > 0]
    median_pos = float(np.median(positive_off) if positive_off.size > 0 else 1.0)
    median_pos = max(median_pos, 1e-9)

    # Global shaping factors (increases shaping when tight or variable)
    tightness = 1.0 - slack_ratio  # 0 loose, 1 tight
    global_shape = 1.0 + 2.6 * tightness + 0.6 * np.tanh(demand_cv)

    # Capacity-aware multiplier (smoothly blends inside-capacity mild encouragement and overflow penalty)
    steep = np.clip(4.8 * global_shape, 2.4, 40.0)
    gate = 1.0 / (1.0 + np.exp(-steep * (norm_combined_cap - 1.0)))  # ~0 if fits, ~1 if overflow
    inside_pow = 1.25
    inside_amp = 0.42 * global_shape
    inside_term = inside_amp * (np.clip(norm_combined_cap, 0.0, 1.0) ** inside_pow)

    overflow = np.clip(norm_combined_cap - 1.0, 0.0, None)
    overflow_sat = overflow / (1.0 + 0.55 * overflow + eps)
    overflow_pow = 2.4
    overflow_amp = 3.0 * global_shape
    overflow_term = overflow_amp * (overflow_sat ** overflow_pow)

    capacity_multiplier = 1.0 + (1.0 - gate) * inside_term + gate * (inside_term + overflow_term)
    capacity_multiplier = np.maximum(capacity_multiplier, 1.0)

    # Heaviness penalty: discourage pairing two large customers (relative to median and capacity)
    heaviness_strength = 0.55 * global_shape
    heaviness_base = (demands / (med_d + eps))[:, None] + (demands / (med_d + eps))[None, :]
    heaviness_penalty = 1.0 + heaviness_strength * (np.clip(heaviness_base, 0.0, None) ** 1.1)

    # Light-pair attraction: Gaussian on combined demand normalized by median to chain tiny customers
    norm_combined_med = combined / (2.0 * med_d + eps)
    light_cut = 0.22
    light_sigma = light_cut * (1.0 + 0.9 * tightness)
    light_gauss = np.exp(-0.5 * (norm_combined_med / (light_sigma + eps)) ** 2)
    min_light_mult = 0.58
    light_bonus = 1.0 - (1.0 - min_light_mult) * light_gauss
    light_bonus = np.clip(light_bonus, min_light_mult, 1.0)

    # Angle-based chaining with respect to depot
    depot = coordinates[0].astype(float)
    vecs = coordinates.astype(float) - depot
    if vecs.shape[1] >= 2:
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    else:
        x = vecs[:, 0] if vecs.shape[1] >= 1 else np.zeros(n)
        angles = np.sign(x) * (np.pi / 2.0) * (np.abs(x) > 1e-9)
    angdiff = np.abs(angles[:, None] - angles[None, :])
    angdiff = np.minimum(angdiff, 2.0 * np.pi - angdiff)
    ang_sigma = np.pi / 5.0
    ang_sim = np.exp(-0.5 * (angdiff / (ang_sigma + eps)) ** 2)
    ang_light_scale = 0.5 + 0.9 * (1.0 - np.clip(norm_combined_cap, 0.0, 1.0))
    angle_reward = 1.0 - 0.34 * (ang_sim ** 1.15) * ang_light_scale
    angle_reward = np.clip(angle_reward, 0.52, 1.03)

    # Proximity kernel: favor short edges (normalized by median_pos), sharper when tight
    base_norm = base / (median_pos + eps)
    prox_sharp = 1.0 + 1.6 * tightness
    proximity_kernel = np.exp(-prox_sharp * base_norm)
    proximity_boost = 1.0 + 1.95 * (proximity_kernel * (1.0 - np.clip(norm_combined_cap, 0.0, 1.0)))

    # Mutual k-NN reward: prefer mutual nearest neighbors, stronger for light pairs
    k = max(1, int(round(np.sqrt(max(n, 4)) * 0.9)))
    dist_for_knn = base.copy()
    np.fill_diagonal(dist_for_knn, np.inf)
    nn_idx = np.argsort(dist_for_knn, axis=1)[:, :k]
    neighbor_map = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    neighbor_map[rows, nn_idx.ravel()] = True
    mutual_map = neighbor_map & neighbor_map.T
    mutual_mult = np.ones((n, n), dtype=float)
    if mutual_map.any():
        light_scale = np.clip(1.0 - 0.76 * np.clip(norm_combined_cap, 0.0, 1.0), 0.28, 1.0)
        ang_align = 0.5 + 0.5 * ang_sim
        mutual_reduction = 1.0 - 0.30 * (light_scale * ang_align)
        mutual_mult[mutual_map] = np.minimum(mutual_mult[mutual_map], mutual_reduction[mutual_map])
    mutual_mult = np.clip(mutual_mult, 0.44, 1.0)

    # Depot nudges: prefer depot links for heavy nodes, avoid trivial depot hops for many ultrasmall nodes
    depot_mult = np.ones((n, n), dtype=float)
    heavy_thresh = 0.44 * cap
    heavy_nodes = np.where(demands >= heavy_thresh)[0]
    for i in heavy_nodes:
        if i == 0:
            continue
        reduction = 0.70 + 0.14 * (1.0 - slack_ratio)
        depot_mult[0, i] = depot_mult[i, 0] = np.clip(reduction, 0.52, 0.95)
    ultra_light_frac = float((demands[1:] <= 0.05 * cap).sum()) / max(1, n - 1)
    if ultra_light_frac > 0.46:
        for i in range(1, n):
            if demands[i] <= 0.05 * cap:
                depot_mult[0, i] = depot_mult[i, 0] = max(depot_mult[0, i], 1.03)

    # Deterministic structured jitter (stable across runs but depends on instance)
    # Build a small instance-hash from coordinates and demands for reproducibility
    coords_flat = np.asarray(coordinates, dtype=float).ravel()
    seed_hash = int((np.sum(coords_flat) * 1009 + np.sum(demands) * 97 + n * 13) % (2**32))
    idx_i = np.arange(n)[:, None].astype(float)
    idx_j = np.arange(n)[None, :].astype(float)
    phase = np.sin(angles[:, None] * 1.91 + 0.09 * idx_i + (seed_hash % 11) * 0.007) + \
            np.cos(angles[None, :] * 1.31 - 0.06 * idx_j + (seed_hash % 17) * 0.005)
    structured_jitter = 1.0 + 0.0065 * (phase / (np.max(np.abs(phase)) + eps))
    structured_jitter = 0.5 * (structured_jitter + structured_jitter.T)

    # Controlled symmetric stochastic noise seeded deterministically
    rng = np.random.default_rng(seed_hash)
    temp_min, temp_max = 0.0025, 0.085
    # More exploration when moderately tight and when demand distribution variable
    temp = temp_min + (temp_max - temp_min) * (0.7 * tightness + 0.3 * np.clip(demand_cv, 0.0, 2.0))
    # Noise amplitude depends on distance (more for medium/distant pairs), but lower for tiny pairs
    pair_amp = 0.35 + 1.5 * (1.0 - proximity_kernel)
    # reduce noise for pairs that obviously overflow (we don't want to randomize too much there)
    pair_amp *= (1.0 - 0.5 * np.clip(norm_combined_cap - 1.0, 0.0, 1.0))
    pair_std = np.clip(temp * pair_amp, 1e-8, 0.28)
    Z = rng.standard_normal(size=(n, n))
    Z = 0.5 * (Z + Z.T)
    noise_mult = 1.0 + pair_std * Z
    # clamp to avoid negative multipliers or absurd changes
    low_bound = np.maximum(0.24, 1.0 - 6.0 * pair_std)
    high_bound = 1.0 + 6.0 * pair_std
    noise_mult = np.clip(noise_mult, low_bound, high_bound)
    noise_mult = 0.5 * (noise_mult + noise_mult.T)

    # Compose multiplicative matrix
    mult = np.ones((n, n), dtype=float)
    mult *= capacity_multiplier
    mult *= heaviness_penalty
    mult *= light_bonus
    mult *= angle_reward
    mult *= mutual_mult
    mult *= proximity_boost
    mult *= depot_mult
    mult *= structured_jitter
    mult *= noise_mult

    # Ensure symmetry and diagonal neutrality
    mult = 0.5 * (mult + mult.T)
    np.fill_diagonal(mult, 1.0)

    # Clamp multipliers to safe bounds to avoid degeneracy
    MIN_MULT, MAX_MULT = 0.28, 8.5
    mult = np.clip(mult, MIN_MULT, MAX_MULT)
    np.fill_diagonal(mult, 1.0)
    mult = 0.5 * (mult + mult.T)

    # Apply multiplier to base distances
    updated = base * mult
    updated = 0.5 * (updated + updated.T)
    np.fill_diagonal(updated, 0.0)

    # Pivot-based limited triangle relaxation to propagate shortcuts (symmetrized)
    # Pivot selection favors demand and closeness; pivot count adapts with n and tightness
    closeness_score = np.sum(np.exp(-base / (median_pos + eps)), axis=1)
    demand_score = demands / (demands.max() + eps)
    pivot_score = 0.6 * demand_score + 0.4 * (closeness_score / (closeness_score.max() + eps))
    pivot_count = int(np.clip(np.ceil(np.log1p(n) * (2.8 if n < 300 else 1.9) * (1.0 + 0.3 * tightness)), 1, n))
    pivots = np.argsort(pivot_score)[-pivot_count:]

    # Number of passes depends on size and tightness
    if n <= 80:
        passes = 3
    elif n <= 250:
        passes = 2
    else:
        passes = 1
    if tightness > 0.82:
        passes += 1

    for _ in range(passes):
        for k in pivots:
            via_k = updated[:, k][:, None] + updated[k, :][None, :]
            # accept shorter transitive distances and maintain symmetry
            updated = np.minimum(updated, via_k)
        updated = 0.5 * (updated + updated.T)

    # Final numeric safeguards
    np.fill_diagonal(updated, 0.0)
    off_mask = ~np.eye(n, dtype=bool)
    tiny_eps = 1e-12
    updated[off_mask] = np.maximum(updated[off_mask], tiny_eps)
    updated = 0.5 * (updated + updated.T)

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
