import numpy as np
import numpy.typing as npt
import concurrent.futures
import traceback
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

def _calculate_route_cost(distmat, route):
    """计算单条路径的成本"""
    if len(route) <= 1:
        return 0.0
    cost = distmat[0, route[0]]  # 从depot到第一个客户
    for i in range(len(route) - 1):
        cost += distmat[route[i], route[i+1]]
    cost += distmat[route[-1], 0]  # 从最后一个客户回到depot
    return cost

def _calculate_total_cost(distmat, routes):
    """计算所有路径的总成本"""
    total_cost = 0.0
    for route in routes:
        if len(route) > 0:
            total_cost += _calculate_route_cost(distmat, route)
    return total_cost

def _check_capacity_constraint(demands, routes, vehicle_capacity):
    """检查容量约束是否满足"""
    for route in routes:
        route_demand = sum(demands[customer] for customer in route)
        if route_demand > vehicle_capacity:
            return False
    return True

def _two_opt_route(distmat, route):
    """对单条路径进行2-opt优化"""
    if len(route) < 3:
        return 0.0

    best_delta = 0.0
    best_i = best_j = -1

    for i in range(len(route) - 1):
        for j in range(i + 2, len(route)):
            # 计算2-opt交换的成本变化
            if i == 0:
                cost_before = distmat[0, route[0]] + distmat[route[i], route[i+1]] + distmat[route[j], route[(j+1) % len(route)] if j+1 < len(route) else 0]
                cost_after = distmat[0, route[j]] + distmat[route[j], route[i]] + distmat[route[i+1], route[(j+1) % len(route)] if j+1 < len(route) else 0]
            else:
                cost_before = distmat[route[i-1], route[i]] + distmat[route[i], route[i+1]] + distmat[route[j], route[(j+1) % len(route)] if j+1 < len(route) else 0]
                cost_after = distmat[route[i-1], route[j]] + distmat[route[j], route[i]] + distmat[route[i+1], route[(j+1) % len(route)] if j+1 < len(route) else 0]

            delta = cost_after - cost_before
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j

    if best_delta < -1e-6:
        # 执行2-opt交换
        route[best_i:best_j+1] = route[best_i:best_j+1][::-1]
        return best_delta
    return 0.0

def _relocate_customer(distmat, demands, routes, vehicle_capacity):
    """重新分配客户到不同路径"""
    best_delta = 0.0
    best_move = None

    for i, route_i in enumerate(routes):
        for j, customer in enumerate(route_i):
            for k, route_k in enumerate(routes):
                if i == k:
                    continue

                # 检查容量约束
                route_k_demand = sum(demands[c] for c in route_k) + demands[customer]
                if route_k_demand > vehicle_capacity:
                    continue

                # 计算移除客户的成本变化
                old_cost_i = _calculate_route_cost(distmat, route_i)
                new_route_i = route_i[:j] + route_i[j+1:]
                new_cost_i = _calculate_route_cost(distmat, new_route_i)

                # 尝试在route_k的每个位置插入客户
                for pos in range(len(route_k) + 1):
                    old_cost_k = _calculate_route_cost(distmat, route_k)
                    new_route_k = route_k[:pos] + [customer] + route_k[pos:]
                    new_cost_k = _calculate_route_cost(distmat, new_route_k)

                    delta = (new_cost_i + new_cost_k) - (old_cost_i + old_cost_k)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j, k, pos)

    if best_move is not None:
        i, j, k, pos = best_move
        customer = routes[i].pop(j)
        routes[k].insert(pos, customer)
        return best_delta
    return 0.0

def _local_search_cvrp(distmat, demands, routes, vehicle_capacity, max_iter=100):
    """CVRP局部搜索"""
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        # 对每条路径进行2-opt优化
        for route in routes:
            if len(route) > 2:
                delta = _two_opt_route(distmat, route)
                if delta < -1e-6:
                    improved = True

        # 尝试重新分配客户
        delta = _relocate_customer(distmat, demands, routes, vehicle_capacity)
        if delta < -1e-6:
            improved = True

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
            best_customer = None
            best_distance = float('inf')

            for customer in unvisited:
                if current_capacity + demands[customer] <= vehicle_capacity:
                    distance = distmat[current_node, customer]
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer

            if best_customer is None:
                break  # 没有可行的客户，开始新路径

            route.append(best_customer)
            current_capacity += demands[best_customer]
            current_node = best_customer
            unvisited.remove(best_customer)

        if route:
            routes.append(route)

    return routes

def _perturbation_cvrp(distmat, guide, penalty, demands, routes, vehicle_capacity, k, perturbation_moves=30):
    """CVRP扰动操作"""
    moves = 0

    while moves < perturbation_moves:
        # 找到具有最大utility的边进行惩罚
        max_util = 0
        max_edge = None

        for route in routes:
            if len(route) == 0:
                continue
            # depot到第一个客户
            u, v = 0, route[0]
            util = guide[u, v] / (1.0 + penalty[u, v])
            if util > max_util:
                max_util = util
                max_edge = (u, v)

            # 路径内的边
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                util = guide[u, v] / (1.0 + penalty[u, v])
                if util > max_util:
                    max_util = util
                    max_edge = (u, v)

            # 最后一个客户到depot
            if len(route) > 0:
                u, v = route[-1], 0
                util = guide[u, v] / (1.0 + penalty[u, v])
                if util > max_util:
                    max_util = util
                    max_edge = (u, v)

        if max_edge:
            penalty[max_edge[0], max_edge[1]] += 1.0
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

    k = 0.1 * best_cost / distmat.shape[0]
    current_routes = [route.copy() for route in best_routes]

    for iteration in range(iter_limit):
        print(iteration)
        _perturbation_cvrp(distmat, guide, penalty, demands, current_routes, vehicle_capacity, k, perturbation_moves)
        _local_search_cvrp(distmat, demands, current_routes, vehicle_capacity)

        current_cost = _calculate_total_cost(distmat, current_routes)
        if current_cost < best_cost:
            best_routes = [route.copy() for route in current_routes]
            best_cost = current_cost

    return best_routes


def update_edge_distance_cvrp(coordinates: np.ndarray, edge_distance: np.ndarray, demands: np.ndarray,
                              vehicle_capacity: int) -> np.ndarray:
    """
    Designs a novel algorithm to update the distance matrix for a CVRP solution.
    When the search is trapped in a local optimum, this function penalizes the
    edges that form the current solution to guide the search to new regions.
    Args:
    coordinates: Coordinate matrix with depot at first position.
    edge_distance: Euclidean distance matrix of each node pairs.
    demands: Node demand vector.
    vehicle_capacity: Maximum vehicle capacity

    Return:
    updated_edge_distance: A matrix of the updated distances.
    """
    num_nodes = len(coordinates)
    updated_edge_distance = edge_distance.copy()

    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:  # Skip same node
                # Penalize edge based on demand and vehicle capacity
                penalty = demands[i] + demands[j] - vehicle_capacity
                updated_edge_distance[i, j] += penalty

    return updated_edge_distance

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
            demands_i64 = self.demands.astype(np.int_)
            routes = _guided_local_search_cvrp(
                distmat=distmat_f64,
                guide=guide_f64,
                demands=demands_i64,
                vehicle_capacity=self.vehicle_capacity ,
                perturbation_moves=30,
                iter_limit=20
            )
            print(routes)
            print(self.demands)
            print(self.vehicle_capacity)
            return routes
        except Exception as e:
            traceback.print_exc()
