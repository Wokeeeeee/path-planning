import math
from random import randint
import numpy as np
from enum import Enum
import operator
from matplotlib import pyplot as plt

# 定义全局变量：地图中节点的像素大小
CELL_WIDTH = 1  # 单元格宽度
CELL_HEIGHT = 1  # 单元格长度
BLOCK_NUM = 150  # 地图中的障碍物数量


def draw_path(mapsize, blocklist, routelist, turnlist):
    plt.figure(figsize=(mapsize[0],mapsize[1]))  # 为了防止x,y轴间隔不一样长，影响最后的表现效果，所以手动设定等长
    plt.xlim(-1, mapsize[0])
    plt.ylim(-1, mapsize[1])
    my_x_ticks = np.arange(0, mapsize[0], 1)
    my_y_ticks = np.arange(0, mapsize[1], 1)
    plt.xticks(my_x_ticks)  # 我理解为竖线的位置与间隔
    plt.yticks(my_y_ticks)
    plt.grid(True)  # 开启栅格
    routelist = np.array(routelist)
    blocklist = np.array(blocklist)
    turnlist = np.array(turnlist)
    plt.plot(routelist[:, 0], routelist[:, 1], linewidth=3)
    plt.plot(turnlist[:, 0], turnlist[:, 1], linewidth=3)
    plt.scatter(blocklist[:, 0], blocklist[:, 1], s=2700, c='k', marker='s')
    plt.title("grid map simulation ")

    plt.show()


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
        if cur_node[0] in range(min(node1[0], node2[0]), max(node1[0], node2[0])+1) and cur_node[1] in range(
                min(node1[1], node2[1]), max(node1[1], node2[1])+1):#range左闭右开
            # 相交返回True
            # print(node1, node2, cur_node)
            A, B, C = get_line(node1, node2)
            peaks = []
            peaks.append(np.array((cur_node[0] - CELL_WIDTH / 2, cur_node[1] - CELL_HEIGHT / 2, 1)))
            peaks.append(np.array((cur_node[0] + CELL_WIDTH / 2, cur_node[1] - CELL_HEIGHT / 2, 1)))
            peaks.append(np.array((cur_node[0] - CELL_WIDTH / 2, cur_node[1] + CELL_HEIGHT / 2, 1)))
            peaks.append(np.array((cur_node[0] + CELL_WIDTH / 2, cur_node[1] + CELL_HEIGHT / 2, 1)))
            weights = np.array((A, B, C))
            # print("judge", np.dot(peaks[0], weights), np.dot(peaks[1], weights), np.dot(peaks[2], weights),
            #       np.dot(peaks[3], weights))
            #四个顶点代入直线是否同号
            if np.dot(peaks[0], weights) <= 0 and np.dot(peaks[1], weights) <= 0 and np.dot(peaks[2],
                                                                                            weights) <= 0 and np.dot(
                peaks[3], weights) <= 0 or \
                    np.dot(peaks[0], weights) >= 0 and np.dot(peaks[1], weights) >= 0 and np.dot(peaks[2],
                                                                                                 weights) >= 0 and np.dot(
                peaks[3], weights) >= 0:
                pass
            else:
                return True

    return False


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
    def __init__(self, pos):
        self.pos = pos
        self.father = None
        self.gvalue = 0
        self.fvalue = 0

    def compute_fx(self, enode, father):
        if father == None:
            print('未设置当前节点的父节点！')

        gx_father = father.gvalue
        # 采用欧式距离计算父节点到当前节点的距离
        gx_f2n = math.sqrt((father.pos[0] - self.pos[0]) ** 2 + (father.pos[1] - self.pos[1]) ** 2)
        gvalue = gx_f2n + gx_father

        hx_n2enode = math.sqrt((self.pos[0] - enode.pos[0]) ** 2 + (self.pos[1] - enode.pos[1]) ** 2)
        fvalue = gvalue + hx_n2enode  #a*算法的启发函数
        return gvalue, fvalue

    def set_fx(self, enode, father):
        self.gvalue, self.fvalue = self.compute_fx(enode, father)
        self.father = father

    def update_fx(self, enode, father):
        gvalue, fvalue = self.compute_fx(enode, father)
        if fvalue < self.fvalue:
            self.gvalue, self.fvalue = gvalue, fvalue
            self.father = father


class AStar(object):
    def __init__(self, mapsize, pos_sn, pos_en):
        self.mapsize = mapsize  # 表示地图的投影大小，并非屏幕上的地图像素大小
        self.openlist, self.closelist, self.blocklist = [], [], []
        self.snode = Node(pos_sn)  # 用于存储路径规划的起始节点
        self.enode = Node(pos_en)  # 用于存储路径规划的目标节点
        self.cnode = self.snode  # 用于存储当前搜索到的节点

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
            nodes_neighbor.append(Node(pos_new))

        return nodes_neighbor


def get_turn(routelist):
    node1, node2, turnlist = routelist[0], routelist[1], []
    if len(routelist) <= 2: return
    turnlist.append(routelist[0])  # 加入起始点
    for cur_node in routelist[2:]:
        # 判断第三个点是否与前两个点共线
        if node1[0] * node2[1] - node2[0] * node1[1] + node2[0] * cur_node[1] - cur_node[0] * node2[1] + cur_node[0] * \
                node1[1] - cur_node[1] * node1[0] != 0:
            # 如果不共线,中间点就是拐点,重新定义启示两个点
            turnlist.append(node2)
            node1 = node2
            node2 = cur_node
        else:  # 如果共线 记录共线的最末点
            node2 = cur_node
    if routelist[len(routelist) - 1] not in turnlist: turnlist.append(routelist[len(routelist) - 1])  # 加入末尾点
    print("turnlist",turnlist)
    return turnlist


def reduce_turn(turnlist, blocklist):
    # 首先判断turn可不可以直接连线，中间也没有障碍物的
    i = 0
    while i + 3 <= len(turnlist):
        node1, mid, node2 = turnlist[i:i + 3]
        # print(i, node1, mid, node2)
        if block_exist(node1, node2, blocklist) == False:
            # print(i, "no block")
            turnlist.remove(mid)
        else:
            i += 1
    print("turnlist after reduction:",turnlist)


def main():
    mapsize = tuple(map(int, input('请输入地图大小，以逗号隔开：').split(',')))
    pos_snode = tuple(map(int, input('请输入起点坐标，以逗号隔开：').split(',')))
    pos_enode = tuple(map(int, input('请输入终点坐标，以逗号隔开：').split(',')))
    myAstar = AStar(mapsize, pos_snode, pos_enode)
    blocklist = gen_blocks(mapsize[0], mapsize[1], pos_snode, pos_enode)
    myAstar.setBlock(blocklist)
    routelist = []  # 记录搜索到的最优路径
    if myAstar.run() == 1:
        routelist = myAstar.get_minroute()
        print(routelist)
        turnlist = get_turn(routelist)
        reduce_turn(turnlist, blocklist)
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



if __name__ == '__main__':
    main()
