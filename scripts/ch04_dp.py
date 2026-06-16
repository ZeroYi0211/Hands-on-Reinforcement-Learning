"""
Chapter 4: Dynamic Programming Algorithm
"""
import copy

class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()
    
    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 起点定义在左上角(0, 0)
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow): # 行
            for j in range(self.ncol): # 列
                for a in range(4):
                    # 位置在悬崖或者终点, 因为无法继续交互, 任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                     True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, j + change[a][1]))
                    next_state = next_y * self.ncol + next_x # 状态索引 = 行 * 12 + 列
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1 # 下一个位置在悬崖
                        reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.nrow * self.env.ncol # 初始化所有状态的价值为0
        self.pi = [[0.25, 0.25, 0.25,0.25]
                   for i in range(self.env.ncol * self.env.nrow)] # 初始化为均匀随即策略
        self.theta = theta
        self.gamma = gamma
