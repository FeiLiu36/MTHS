import multiprocessing
import os

from data_process import position_transform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from fbr import f
import copy
import math
import time

from blocks_generator import TIMEOUT, FULL_SUPPORTED, BlockJSONHandler, generate_general_blocks
from entity import *
from readData import read_container_loading_file, read_pt_data


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
    # 搜索过程
    iter = 1
    while current_states and time.time() - start < TIMEOUT:
        print(f'----------case:{case_number}-cur_iter={iter}---------')
        # 遍历当前层的所有节点并拓展节点
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
        # 剪枝策略
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
        # 还有去掉类型数量相同的node
        # 保留beam_width个最佳节点
        new_states = unique_nodes[:w]
        for new_state in new_states:
            if new_state.est_volume > best_volume:
                best_volume = new_state.est_volume
            # 这里更新最优后要重置一下评估值
            new_state.est_volume = 0
        # 当前层
        all_states.extend(current_states)
        current_states = copy.deepcopy(new_states)

    # 清理内存
    all_states.clear()
    return best_volume


def ranking_spaces(input_space_list, bin, sf=FULL_SUPPORTED):
    """对free_space_list进行排序，返回最合适的space"""

    space_list = copy.deepcopy(input_space_list)
    # 检查容器是否提供8个角点
    assert len(bin.corners) == 8, "容器必须提供8个角点"
    # 先根据曼哈顿距离选择space
    space_metrics = []
    for space in space_list:
        distances = [manhattan_distance(s_corner, c_corner) for s_corner, c_corner in zip(space.corners, bin.corners)]
        # 如果要full-supported就取四个底角点
        if sf:
            distances = distances[:4]
        # 找到锚角和曼哈顿距离
        min_distance = min(distances)
        anchor_idx = distances.index(min_distance)
        space.anchor_corner = space.corners[anchor_idx]
        # 获取曼哈顿距离元组
        vector = manhattan_vector(space.corners[anchor_idx], bin.corners[anchor_idx])
        vector.sort(key=lambda x: x)
        space_metrics.append({
            'space': space,
            'anchor_distance': tuple(vector),
            'volume': space.volume,
        })
    selected = sorted(space_metrics, key=lambda x: (x['anchor_distance'], -x['volume']))
    return [selected[i]['space'] for i in range(len(selected))]


def fiter_legal_blocks(blocks: list[Block], s: Space, box_map):
    legal_blocks = []
    # 数量和尺寸合法检查筛选
    for b_block in blocks:
        # 数量合法检查
        flag = True
        # 尺寸合法检查
        if b_block.dimensions[0] > s.lx or b_block.dimensions[1] > s.ly or b_block.dimensions[2] > s.lz:
            continue
        # 计算block的box的类型和数量
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
    # 扩展节点
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
        # 所有空间都不行，表示节点走到尽头无法拓展
        return []
    selected_space = cur_state.free_space_list[0]
    # 从高到低筛选
    blocks.sort(key=lambda x: x.ranking, reverse=True)
    # 选择要选择可选的：w可能大于count
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
        # 更新当前子节点
        child_state.update(action, state)
        # 计算子节点的greedy体积评估
        greedy(input_blocks, child_state, box_data, iter, child_id, case_number)
        child_nodes.append(child_state)
        child_id += 1
    return child_nodes


def main(input_case: dict, general_blocks: list[Block], timeout=TIMEOUT):
    start = time.time()
    case = copy.deepcopy(input_case)
    # 从case中获取数据
    box_data = case.get("boxes")
    bin_size = case.get("container")
    case_number = case.get("case_number")
    available_boxes = case.get("ava_box")
    bin_type = BinType(bin_size[0], bin_size[1], bin_size[2])
    # 搜索集束数目
    w = 4
    best_volume = 0
    while time.time() - start < timeout and w < 50000:
        print(f"----------------case {case_number} new iter---------------")
        print(f"w={w}")
        cur_volume = beam_search(bin_type, general_blocks, box_data, available_boxes, w, case_number, start)
        # 更新搜索努力因子
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
    # 循环处理所有自由空间，直到没有自由空间为止
    while node.free_space_list:
        node.free_space_list = ranking_spaces(node.free_space_list, node.bin_type)
        fspace = node.free_space_list[0]
        count = 0
        # 检查哪些块可以放入当前自由空间
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
            # 如果当前空间装不下任何块，删去
            del node.free_space_list[0]
    # 更新该节点的评估体积
    state.est_volume = node.volume
    print(f'cur_case: {case_number}---greedy est_volume: {node.volume / node.bin_type.volume * 100:.2f}')


if __name__ == '__main__':

    benchmark = "BR"
    benchmark_num = 4
    cases = read_container_loading_file(f"data/offline/br/{benchmark}{benchmark_num}.txt")
    case = cases[0]
    blocks = BlockJSONHandler.load_blocks(f'data/blocks/{benchmark}{benchmark_num}/case{1}_blocks.json')


    # arg2 is timeout
    res = main(case, blocks, 3000)
    print(f'{res * 100:.2f}')


