import math
import operator
import time
from random import randint
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt
import pandas as pd

# 定义全局变量
CELL_WIDTH = 1  # 单元格宽度
CELL_HEIGHT = 1  # 单元格长度
BLOCK_NUM = 60  # 地图中的障碍物数量
BLOCK_BIG_NUM = 8  # 地图中大障碍物数量
BLOCK_BORD = (20, 40)  # 地图中障碍块最大边长
V0 = 15  # 初速度
COST_RATE = 1  # 转弯之后的速度变为原来的0.9  如果损失设置太小方向会转不过来


def draw_path_from_csv(mapsize):
    plt.figure(figsize=(100, 100))  # 为了防止x,y轴间隔不一样长，影响最后的表现效果，所以手动设定等长
    plt.xlim(-1, mapsize[0])
    plt.ylim(-1, mapsize[1])
    my_x_ticks = np.arange(0, mapsize[0], 1)
    my_y_ticks = np.arange(0, mapsize[1], 1)
    plt.xticks(my_x_ticks)  # 我理解为竖线的位置与间隔
    plt.yticks(my_y_ticks)
    plt.grid(True)  # 开启栅格

    route1 = np.array(read_list_from_file("./route_rate1.csv"))
    route0dot9 = np.array(read_list_from_file("./route_rate0.9.csv"))
    route0dot6 = np.array(read_list_from_file("./route_rate0.6.csv"))
    blocklist = read_list_from_file("./block.csv")
    sub = np.array([(0.5, 0.5)])
    blocklist = np.subtract(np.array(blocklist), sub)

    plt.plot(route1[:, 0], route1[:, 1], linewidth=2, label="no velocity loss")
    plt.plot(route0dot9[:, 0], route0dot9[:, 1], linewidth=2, label="loss_rate: 0.9")
    plt.plot(route0dot6[:, 0], route0dot6[:, 1], linewidth=2, label="loss_rate: 0.6")
    plt.scatter(blocklist[:, 0], blocklist[:, 1], s=600, c='k', marker='s')
    plt.scatter(route1[0][0], route1[0][1], s=600, label="start_point")
    plt.scatter(route1[len(route1) - 1][0], route1[len(route1) - 1][1], s=600, label="end_point")
    plt.title("comparison")
    plt.legend()
    plt.savefig('route_comparison.png', dpi=100)
    plt.clf()


def draw_path(mapsize, blocklist, routelist, turnlist):
    plt.figure(figsize=(100, 100))  # 为了防止x,y轴间隔不一样长，影响最后的表现效果，所以手动设定等长
    plt.xlim(-1, mapsize[0])
    plt.ylim(-1, mapsize[1])
    my_x_ticks = np.arange(0, mapsize[0], 1)
    my_y_ticks = np.arange(0, mapsize[1], 1)
    plt.xticks(my_x_ticks)  # 我理解为竖线的位置与间隔
    plt.yticks(my_y_ticks)
    plt.grid(True)  # 开启栅格
    sub = np.array([(0.5, 0.5)])
    routelist = np.subtract(np.array(routelist), sub)
    blocklist = np.subtract(np.array(blocklist), sub)
    turnlist = np.subtract(np.array(turnlist), sub)
    plt.plot(routelist[:, 0], routelist[:, 1], linewidth=2)
    plt.plot(turnlist[:, 0], turnlist[:, 1], linewidth=2)
    plt.scatter(blocklist[:, 0], blocklist[:, 1], s=600, c='k', marker='s')
    plt.title("grid map simulation ")
    plt.show()

    # save_to_file(routelist, "route_rate1.csv")
    # save_to_file(turnlist, "turn_rate1.csv")
    # plt.savefig('result_rate1.png', dpi=100)
    # plt.clf()


# 由两个点求得直线公式：AX+BY+C=0
def get_line(node1, node2):
    A = node2[1] - node1[1]
    B = node1[0] - node2[0]
    C = node2[0] * node1[1] - node1[0] * node2[1]
    return A, B, C


# 判断node1和node2之间有没有存在障碍物,障碍物是个矩形，所以判断直线和该矩形是否相交：四个点带入直线是否同号
def block_exist(node1, node2, blocklist):
    for cur_node in blocklist:
        # 障碍物夹在两个点中间
        if cur_node[0] in range(int(min(node1[0], node2[0])), int(max(node1[0], node2[0]) + 1)) and cur_node[
            1] in range(
            int(min(node1[1], node2[1])), int(max(node1[1], node2[1]) + 1)):  # range左闭右开
            # 相交返回True
            # print(node1, node2, cur_node)
            A, B, C = get_line(node1, node2)
            # 用点到直线的距离计算
            distance = np.abs(A * cur_node[0] + B * cur_node[1] + C) / (np.sqrt(A ** 2 + B ** 2))
            if distance <= math.sqrt(2) * CELL_WIDTH / 2:
                return True

    return False


def init_orientation(snode, enode):
    x_ori = 1 if enode[0] - snode[0] > 0 else 0 if enode[0] - snode[0] == 0 else -1
    y_ori = 1 if enode[1] - snode[1] > 0 else 0 if enode[1] - snode[1] == 0 else -1
    print(x_ori, y_ori)
    return (x_ori, y_ori)


class Color(Enum):
    ''' 颜色 '''
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    @staticmethod
    def random_color():
        '''设置随机颜色'''
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        return (r, g, b)


class Map(object):
    def __init__(self, mapsize):
        self.mapsize = mapsize

    def generate_cell(self, cell_width, cell_height):
        '''
        定义一个生成器，用来生成地图中的所有节点坐标
        :param cell_width: 节点宽度
        :param cell_height: 节点长度
        :return: 返回地图中的节点
        '''
        x_cell = -cell_width
        for num_x in range(self.mapsize[0] // cell_width):
            y_cell = -cell_height
            x_cell += cell_width
            for num_y in range(self.mapsize[1] // cell_height):
                y_cell += cell_height
                yield (x_cell, y_cell)


class Node(object):
    def __init__(self, pos, offset):
        self.pos = pos
        self.offset = offset
        self.father = None
        self.gvalue = 0
        self.fvalue = 0
        self.velocity = None

    def set_vel(self, vel):
        self.velocity = vel

    def compute_fx(self, enode, father):
        if father == None:
            print('未设置当前节点的父节点！')

        if self.offset != father.offset:  # 如果当前节点需要改变方向，则速度损失，如果不需要改变方向，则保持原速度
            self.velocity = father.velocity * COST_RATE
        else:
            self.velocity = father.velocity
        gx_father = father.gvalue
        # 采用欧式距离计算父节点到当前节点的距离
        gx_f2n = math.sqrt((father.pos[0] - self.pos[0]) ** 2 + (father.pos[1] - self.pos[1]) ** 2)
        # print(gx_f2n,self.velocity)
        gvalue = gx_f2n / self.velocity + gx_father  # gvalue代表累加距离
        # print("father:", father.pos, father.offset, father.velocity, "cur:", self.pos, self.offset, self.velocity)
        hx_n2enode = math.sqrt(
            (self.pos[0] - enode.pos[0]) ** 2 + (self.pos[1] - enode.pos[1]) ** 2)  # hx_n2enode代表当前点距离目标点的距离
        # hx_n2enode = math.sqrt(2) * max(abs(self.pos[0] - enode.pos[0]), abs(self.pos[1] - enode.pos[1]))  # 启发函数改为切比雪夫
        # fvalue = gvalue + hx_n2enode  # 计算总距离

        fvalue = gvalue + hx_n2enode / self.velocity  # fvalue代表所用时间 时间=走过距离+估算的欧式距离/当前速度
        # print(self.velocity,fvalue)
        return gvalue, fvalue, (self.pos[0] - father.pos[0], self.pos[1] - father.pos[1]), self.velocity

    def set_fx(self, enode, father):
        self.gvalue, self.fvalue, self.offset, self.velocity = self.compute_fx(enode, father)
        self.father = father

    def update_fx(self, enode, father):
        gvalue, fvalue, self.offset, self.velocity = self.compute_fx(enode, father)
        if fvalue < self.fvalue:
            self.gvalue, self.fvalue = gvalue, fvalue
            self.father = father


class AStar(object):
    def __init__(self, mapsize, pos_sn, pos_en):
        self.mapsize = mapsize  # 表示地图的投影大小，并非屏幕上的地图像素大小
        self.openlist, self.closelist, self.blocklist = [], [], []
        self.snode = Node(pos_sn, init_orientation(pos_sn, pos_en))  # 用于存储路径规划的起始节点
        self.snode.set_vel(V0)  # 设置初始速度
        self.enode = Node(pos_en, None)  # 用于存储路径规划的目标节点
        self.cnode = self.snode  # 用于存储当前搜索到的节点
        self.open_num = 0

    def run(self):
        self.openlist.append(self.snode)
        while (len(self.openlist) > 0):
            # 查找openlist中fx最小的节点
            fxlist = list(map(lambda x: x.fvalue, self.openlist))
            index_min = fxlist.index(min(fxlist))
            self.cnode = self.openlist[index_min]
            del self.openlist[index_min]
            self.closelist.append(self.cnode)

            # 扩展当前fx最小的节点，并进入下一次循环搜索
            self.extend(self.cnode)
            # 如果openlist列表为空，或者当前搜索节点为目标节点，则跳出循环
            if len(self.openlist) == 0 or self.cnode.pos == self.enode.pos:
                break

        if self.cnode.pos == self.enode.pos:
            self.enode.father = self.cnode.father
            return 1
        else:
            return -1

    def get_minroute(self):
        minroute = []
        current_node = self.enode

        while (True):
            minroute.append(current_node.pos)
            current_node = current_node.father
            if current_node.pos == self.snode.pos:
                break

        minroute.append(self.snode.pos)
        minroute.reverse()
        print("openlist max size:", self.open_num)
        return minroute

    def extend(self, cnode):
        nodes_neighbor = self.get_neighbor(cnode)
        for node in nodes_neighbor:
            # 判断节点node是否在closelist和blocklist中，因为closelist和blocklist中元素均为Node类，所以要用map函数转换为坐标集合
            if node.pos in list(map(lambda x: x.pos, self.closelist)) or node.pos in self.blocklist:
                continue
            else:
                if node.pos in list(map(lambda x: x.pos, self.openlist)):
                    node.update_fx(self.enode, cnode)
                else:
                    node.set_fx(self.enode, cnode)
                    self.openlist.append(node)
                    if (len(self.openlist) > self.open_num):
                        self.open_num = len(self.openlist)

    def setBlock(self, blocklist):
        '''
        获取地图中的障碍物节点，并存入self.blocklist列表中
        注意：self.blocklist列表中存储的是障碍物坐标，不是Node类
        :param blocklist:
        :return:
        '''
        self.blocklist.extend(blocklist)
        # for pos in blocklist:
        #     block = Node(pos)
        #     self.blocklist.append(block)

    def get_neighbor(self, cnode):
        offsets = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
        nodes_neighbor = []
        x, y = cnode.pos[0], cnode.pos[1]
        for os in offsets:
            x_new, y_new = x + os[0], y + os[1]
            pos_new = (x_new, y_new)
            # 判断是否在地图范围内,超出范围跳过
            if x_new < 0 or x_new > self.mapsize[0] - 1 or y_new < 0 or y_new > self.mapsize[1]:
                continue
            nodes_neighbor.append(Node(pos_new, os))

        return nodes_neighbor


def get_turn(routelist):
    node1, node2, turnlist, indexlist = routelist[0], routelist[1], [], []
    if len(routelist) <= 2: return
    turnlist.append(routelist[0])  # 加入起始点
    indexlist.append(0)
    i = 1
    for cur_node in routelist[2:]:
        # 判断第三个点是否与前两个点共线
        if node1[0] * node2[1] - node2[0] * node1[1] + node2[0] * cur_node[1] - cur_node[0] * node2[1] + cur_node[0] * \
                node1[1] - cur_node[1] * node1[0] != 0:
            # 如果不共线,中间点就是拐点,重新定义启示两个点
            turnlist.append(node2)
            indexlist.append(i)
            node1 = node2
            node2 = cur_node
        else:  # 如果共线 记录共线的最末点
            node2 = cur_node
        i += 1  # 索引值
    if routelist[len(routelist) - 1] not in turnlist: turnlist.append(routelist[len(routelist) - 1])  # 加入末尾点
    print("turnlist", turnlist)
    indexlist.append(len(routelist) - 1)
    return turnlist, indexlist


# 获得两条线之间的焦点
def get_crosspoint(line1, line2):
    a0, b0, c0 = line1
    a1, b1, c1 = line2
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return x, y


def find_new_turn(routelist, blocklist):
    turnlist, indexlist = get_turn(routelist)
    # print(indexlist)
    i = 0
    while i + 2 < len(indexlist):
        G1, G2, G3 = turnlist[i], turnlist[i + 1], turnlist[i + 2]
        # 判断相隔拐点之间是否联通: 联通->不做任何操作
        if block_exist(G1, G3, blocklist) == True:
            # 不连通->寻找新的拐点
            G1toG2 = routelist[indexlist[i] + 1: indexlist[i + 1] + 1]
            G2toG3 = routelist[indexlist[i + 1]: indexlist[i + 2]]
            node1, node3 = [], []
            # 如果没有p1=拐点
            for p1 in G1toG2:
                if block_exist(p1, G3, blocklist) == False:
                    node1 = p1
                    break
            for p3 in list(reversed(G2toG3)):
                if block_exist(p3, G1, blocklist) == False:
                    node3 = p3
            if node1 and node3:
                turnlist[i + 1] = get_crosspoint(get_line(node1, G3), get_line(node3, G1))

            # print(i, G1toG2, list(reversed(G2toG3)), turnlist[i + 1])
            i += 1
        else:
            turnlist.remove(G2)
            indexlist.remove(indexlist[i + 1])
    print("find_new_turn:", turnlist)
    return turnlist


def main():
    t = time.time()
    mapsize = tuple(map(int, input('请输入地图大小，以逗号隔开：').split(',')))
    pos_snode = tuple(map(int, input('请输入起点坐标，以逗号隔开：').split(',')))
    pos_enode = tuple(map(int, input('请输入终点坐标，以逗号隔开：').split(',')))
    myAstar = AStar(mapsize, pos_snode, pos_enode)
    blocklist = gen_blocks(mapsize[0], mapsize[1], pos_snode, pos_enode) #生成密且小的障碍物
    # blocklist = gen_blocks_big(mapsize[0], mapsize[1], pos_snode, pos_enode)  # 生成少且大的障碍物
    # blocklist = read_list_from_file("block.csv")
    myAstar.setBlock(blocklist)
    if myAstar.run() == 1:
        routelist = myAstar.get_minroute()
        # turnlist, indexlist = get_turn(routelist)
        turnlist = find_new_turn(routelist, blocklist)
        print(time.time() - t)
        draw_path(mapsize, blocklist, routelist, turnlist)
    else:
        print('路径规划失败！')


# 生成障碍物 禁飞区 栅格地图cell边长为1
def gen_blocks(width, height, snode, enode):
    '''
    随机生成障碍物
    :param width: 地图宽度
    :param height: 地图高度
    :return:返回障碍物坐标集合
    '''
    i, blocklist = 0, []
    while (i < BLOCK_NUM):
        block = (randint(0, width - 1), randint(0, height - 1))
        if operator.eq(block, enode) or operator.eq(block, snode):
            continue
        if block not in blocklist:
            blocklist.append(block)
            i += 1

    return blocklist


def is_inrect(rect, point):
    # resct:左下,_,_右上
    print(rect, point)
    if point[0] in range(rect[0][0], rect[3][0] + 1) and \
            point[1] in range(rect[0][1], rect[3][1] + 1):
        return True
    return False


def save_to_file(list, filename):
    df = pd.DataFrame(list)
    df.to_csv(filename)
    print("保存成功")


# 仅限二维点
def read_list_from_file(filenmae):
    df = pd.read_csv(filenmae, dtype=float)
    dct_data = np.array(df.loc[:, :])
    pxs = dct_data[:, 1]
    pys = dct_data[:, 2]

    list = [(x, y) for (x, y) in zip(pxs, pys)]
    return list


def gen_blocks_big(width, height, snode, enode):
    '''
    随机生成障碍物
    :param width: 地图宽度
    :param height: 地图高度
    :return:返回障碍物坐标集合
    '''
    i, blocklist = 0, []
    while (i < BLOCK_BIG_NUM):
        block = (randint(0, width - 1), randint(0, height - 1))
        bord = randint(BLOCK_BORD[0], BLOCK_BORD[1])
        if bord % 2 != 0: continue
        if block[0] - bord / 2 >= 0 and block[0] + bord / 2 <= width \
                and block[1] - bord / 2 >= 0 and block[1] + bord / 2 <= height:
            rect = [
                (block[0] - int(bord / 2), block[1] - int(bord / 2)),
                (block[0] + int(bord / 2), block[1] - int(bord / 2)),
                (block[0] - int(bord / 2), block[1] + int(bord / 2)),
                (block[0] + int(bord / 2), block[1] + int(bord / 2))
            ]
            if is_inrect(rect, snode) or is_inrect(rect, enode):
                continue
            else:
                print(rect)
                for x in range(int(block[0] - bord / 2), int(block[0] + bord / 2 + 1)):
                    for y in range(int(block[1] - bord / 2), int(block[1] + bord / 2 + 1)):
                        blocklist.append((x, y))
                i += 1

    # blocklist = np.array(blocklist)
    # plt.figure(figsize=(100, 100))
    # plt.scatter(blocklist[:, 0], blocklist[:, 1], s=2800, c='k', marker='s')
    # plt.savefig("block.png")
    # save_to_file(blocklist, "block.csv")
    # xxx = read_list_from_file("block.csv")
    return blocklist


if __name__ == '__main__':
    main()
