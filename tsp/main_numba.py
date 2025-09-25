import multiprocessing
import os

# Original imports are preserved, with additions for Numba and NumPy
# We will define the manhattan functions ourselves to jit-compile them,
# so they are removed from the entity import list.
from data_process import position_transform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from fbr import f
import copy
import math
import time

# +++ START OF NUMBA ADDITIONS +++
import numpy as np
import numba
from numba import njit
# +++ END OF NUMBA ADDITIONS +++

from blocks_generator import TIMEOUT, FULL_SUPPORTED, BlockJSONHandler, generate_general_blocks
# Assuming entity.py contained Space, BinType, etc., and also manhattan_distance/vector
# We now specify class imports and define the jitted functions separately for clarity and optimization.
from entity import Space, BinType, Box, Action, State
from readData import read_container_loading_file, read_pt_data
# This import is used in the original fiter_legal_blocks function
from collections import Counter


# +++ START OF NUMBA-OPTIMIZED FUNCTIONS +++

@njit(cache=True)
def manhattan_distance(p1, p2):
    """
    Calculates the Manhattan distance between two 3D points.
    JIT-compiled by Numba for high performance.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])


@njit(cache=True)
def manhattan_vector(p1, p2):
    """
    Calculates the component-wise absolute difference between two 3D points.
    JIT-compiled by Numba for high performance.
    """
    return (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]))


@njit(cache=True)
def _calculate_space_metrics_numba(spaces_corners, bin_corners, sf):
    """
    A Numba-jitted helper function to perform the intensive calculations for ranking_spaces.
    It takes NumPy arrays as input and returns a list of tuples with sorting keys and indices.
    """
    num_spaces = spaces_corners.shape[0]
    # Numba works well with lists of simple types like tuples.
    space_metrics_tuples = []

    for i in range(num_spaces):
        space_i_corners = spaces_corners[i]

        # Calculate distances for the current space's 8 corners
        distances = np.empty(8, dtype=np.float64)
        for j in range(8):
            distances[j] = manhattan_distance(space_i_corners[j], bin_corners[j])

        # Find the minimum distance and its corresponding anchor corner index
        if sf:  # Only consider the bottom 4 corners
            min_dist_val = np.min(distances[:4])
            anchor_idx = np.argmin(distances[:4])
        else:  # Consider all 8 corners
            min_dist_val = np.min(distances)
            anchor_idx = np.argmin(distances)

        # Calculate the vector from the anchor corner and sort it to create a canonical key
        vector = manhattan_vector(space_i_corners[anchor_idx], bin_corners[anchor_idx])

        # Numba can't sort lists/tuples with a key, so we sort the 3-element tuple manually
        a, b, c = vector
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        sorted_vector_tuple = (a, b, c)

        # Append a tuple containing the sorting key, the original index, and the anchor index
        space_metrics_tuples.append((sorted_vector_tuple, anchor_idx, i))

    return space_metrics_tuples

# +++ END OF NUMBA-OPTIMIZED FUNCTIONS +++


def beam_search(bin: BinType, general_blocks: list[Block], input_box_data: list[Box], available_boxes: dict, w: int,
                case_number: int, start) -> float:
    # initial state
    bin_type = copy.deepcopy(bin)
    box_data = copy.deepcopy(input_box_data)
    remaining_box_map = copy.deepcopy(available_boxes)
    initial_space = Space(0, 0, 0, bin_type.length, bin_type.width, bin_type.height)
    initial_state = State(space_list=[initial_space], bin_type=bin_type, ava_boxes=remaining_box_map)
    # state store
    current_states = [initial_state]
    all_states = []
    best_volume = 0

    iter = 1
    while current_states and time.time() - start < TIMEOUT:
        print(f'----------case:{case_number}-cur_iter={iter}---------')

        next_states = []
        for state in current_states:
            if state == initial_state:
                successors = expand_state(state, general_blocks, box_data, w ** 2, iter, case_number)
            else:
                successors = expand_state(state, general_blocks, box_data, w, iter, case_number)
            next_states.extend(successors)
            iter += 1
        if not next_states:
            return best_volume

        next_states.sort(key=lambda x: x.est_volume, reverse=True)
        unique_nodes = []
        prev_vol = 0
        prev_est_vol = 0
        for node in next_states:
            if node.volume == prev_vol and node.est_volume == prev_est_vol:
                continue
            unique_nodes.append(node)
            prev_vol = node.volume
            prev_est_vol = node.est_volume

        new_states = unique_nodes[:w]
        for new_state in new_states:
            if new_state.est_volume > best_volume:
                best_volume = new_state.est_volume

            new_state.est_volume = 0

        all_states.extend(current_states)
        current_states = copy.deepcopy(new_states)


    all_states.clear()
    return best_volume


def ranking_spaces(input_space_list, bin, sf=FULL_SUPPORTED):
    """
    This function is now a wrapper that uses a Numba-jitted helper for performance.
    It prepares data, calls the fast helper, and then formats the output.
    Functionality remains identical to the original.
    """
    if not input_space_list:
        return []

    space_list = copy.deepcopy(input_space_list)

    assert len(bin.corners) == 8, "Bin must have 8 corners"

    # 1. Prepare data for Numba: extract coordinates into NumPy arrays
    spaces_corners_np = np.array([s.corners for s in space_list], dtype=np.float64)
    bin_corners_np = np.array(bin.corners, dtype=np.float64)

    # 2. Call the fast Numba-jitted helper function
    # Returns a list of (sorted_vector_tuple, anchor_idx, original_index)
    numba_results = _calculate_space_metrics_numba(spaces_corners_np, bin_corners_np, sf)

    # 3. Combine Numba results with Python objects to create a list of dictionaries for sorting
    space_metrics = []
    for sorted_vector, anchor_idx, original_index in numba_results:
        space = space_list[original_index]
        space_metrics.append({
            'space': space,
            'anchor_distance': sorted_vector,  # The pre-sorted tuple for sorting
            'volume': space.volume,
            'anchor_idx': anchor_idx
        })

    # 4. Sort the list of dictionaries based on the pre-calculated keys
    selected = sorted(space_metrics, key=lambda x: (x['anchor_distance'], -x['volume']))

    # 5. Build the final sorted list of Space objects and update their anchor_corner attribute
    result_list = []
    for item in selected:
        space_obj = item['space']
        anchor_idx = item['anchor_idx']
        # This side-effect is part of the original function's behavior
        space_obj.anchor_corner = space_obj.corners[anchor_idx]
        result_list.append(space_obj)

    return result_list


def fiter_legal_blocks(blocks: list[Block], s: Space, box_map):
    legal_blocks = []

    for b_block in blocks:

        flag = True

        if b_block.dimensions[0] > s.lx or b_block.dimensions[1] > s.ly or b_block.dimensions[2] > s.lz:
            continue

        typed_ids = [element.type_id for element in b_block.boxes if element.type_id is not None]
        dic = Counter(typed_ids)
        for key in dic.keys():
            data = box_map[key]
            if key in box_map.keys() and dic[key] > data[key]:
                flag = False
                break
        if flag:
            legal_blocks.append(b_block)
    return legal_blocks


def expand_state(state: State, input_blocks: list[Block], box_data, w: int, iter: int, case_number: int) -> list[State]:

    blocks = copy.deepcopy(input_blocks)
    cur_state = copy.deepcopy(state)
    # K3 select space
    cur_state.free_space_list = ranking_spaces(cur_state.free_space_list, cur_state.bin_type)
    count = 0
    while count == 0 and cur_state.free_space_list:
        space = cur_state.free_space_list[0]
        for block in blocks:
            block.ranking = 0
            if space.can_fit(block) and cur_state.can_load(block):
                count += 1
                block.ranking = f(block, space, box_data, cur_state.box_map)
            else:
                block.ranking = float('-inf')
        if count == 0:
            del cur_state.free_space_list[0]
            continue
    if count == 0:

        return []
    selected_space = cur_state.free_space_list[0]

    blocks.sort(key=lambda x: x.ranking, reverse=True)

    if w >= count:
        top_blocks = blocks[:count]
    else:
        top_blocks = blocks[:w]
    # update info
    child_nodes = []
    child_id = 0
    for child_block in top_blocks:
        child_state = copy.deepcopy(cur_state)
        new_pos = position_transform(selected_space, child_block)
        x, y, z = new_pos
        action = Action(x, y, z, child_block, selected_space)

        child_state.update(action, state)

        greedy(input_blocks, child_state, box_data, iter, child_id, case_number)
        child_nodes.append(child_state)
        child_id += 1
    return child_nodes


def main(input_case: dict, general_blocks: list[Block], timeout=TIMEOUT):
    start = time.time()
    case = copy.deepcopy(input_case)

    box_data = case.get("boxes")
    bin_size = case.get("container")
    case_number = case.get("case_number")
    available_boxes = case.get("ava_box")
    bin_type = BinType(bin_size[0], bin_size[1], bin_size[2])

    w = 4
    best_volume = 0
    while time.time() - start < timeout and w < 50000:
        print(f"----------------case {case_number} new iter---------------")
        print(f"w={w}")
        cur_volume = beam_search(bin_type, general_blocks, box_data, available_boxes, w, case_number, start)

        w = int(math.ceil(math.sqrt(2) * w))
        if cur_volume > best_volume:
            best_volume = cur_volume
        print(
            f'--------------- case {case_number} : current iter rate={best_volume / bin_type.volume * 100:.5f}%-------------')
    rate = best_volume / bin_type.volume

    print(f'case {case_number} : final rate={rate * 100:.2f}%')
    return rate


def greedy(input_blocks: list[Block], state: State, box_data, iter, id, case_number):
    node = copy.deepcopy(state)
    blocks = copy.deepcopy(input_blocks)

    while node.free_space_list:
        node.free_space_list = ranking_spaces(node.free_space_list, node.bin_type)
        fspace = node.free_space_list[0]
        count = 0

        for block in blocks:
            block.ranking = 0
            if fspace.can_fit(block) and node.can_load(block):
                count += 1
                block.ranking = f(block, fspace, box_data, node.box_map)
            else:
                block.ranking = float('-inf')
        if count != 0:
            blocks.sort(key=lambda x: x.ranking, reverse=True)
            selected_block = blocks[0]
            new_pos = position_transform(fspace, selected_block)
            x, y, z = new_pos
            action = Action(x, y, z, selected_block, fspace)
            node.update(action, None)
        else:

            del node.free_space_list[0]

    state.est_volume = node.volume
    print(f'cur_case: {case_number}---greedy est_volume: {node.volume / node.bin_type.volume * 100:.2f}')


if __name__ == '__main__':
    # Numba JIT compilation happens on the first call.
    # To avoid measuring compilation time as part of the main run,
    # you could add a small "warm-up" call here on dummy data if needed.

    benchmark = "BR"
    benchmark_num = 4
    cases = read_container_loading_file(f"data/offline/br/{benchmark}{benchmark_num}.txt")
    case = cases[0]
    blocks = BlockJSONHandler.load_blocks(f'data/blocks/{benchmark}{benchmark_num}/case{1}_blocks.json')


    # arg2 is timeout
    res = main(case, blocks, 3000)
    print(f'{res * 100:.2f}')