import math
from utils import *
import numpy as np
import cv2


class Status:
    # G
    g = 9.801
    # 预测阶段白噪声
    # w_p,w_v,w_ap,w_bg,w_ba = [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]

    # ------------------------------------------------------------需要调参--------------------------------------------------------------
    # 测量协方差矩阵R -> 越小越信任量测，稳态噪声（重要！）
    r_v, r_ap, r_vl, r_p, r_a, r_m = [0.15, 0.15, 0.15], [0.12, 0.12, 0.12], [0.1, 0.1, 0.05], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]

    # 预测协方差矩阵Q -> 越小越信任模型（重要！） 如果没有先验信息，应当适当增大Q的取值
    covariance_q = np.eye(9)

    # 状态delta_k的协方差矩阵P -> 决定瞬态过程收敛速率，稳态过程中的P由QR决定
    covariance = np.eye(9)
    # ---------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba):
        self.position = array2matrix(position)
        self.velocity = array2matrix(velocity)
        alpha = 0.09355373656573884 / 2
        self.q = np.matrix([math.cos(alpha), 0, 0, math.sin(alpha)]).T
        # self.q = np.matrix([1, 0, 0, 0]).T
        self.B2N_matrix = np.matrix([
            [1 - 2 * self.q[2, 0] * self.q[2, 0] - 2 * self.q[3, 0] * self.q[3, 0],
             2 * self.q[1, 0] * self.q[2, 0] - 2 * self.q[0, 0] * self.q[3, 0],
             2 * self.q[1, 0] * self.q[3, 0] + 2 * self.q[0, 0] * self.q[2, 0]],
            [2 * self.q[1, 0] * self.q[2, 0] + 2 * self.q[0, 0] * self.q[3, 0],
             1 - 2 * self.q[1, 0] * self.q[1, 0] - 2 * self.q[3, 0] * self.q[3, 0],
             2 * self.q[2, 0] * self.q[3, 0] - 2 * self.q[0, 0] * self.q[1, 0]],
            [2 * self.q[1, 0] * self.q[3, 0] - 2 * self.q[0, 0] * self.q[2, 0],
             2 * self.q[2, 0] * self.q[3, 0] + 2 * self.q[0, 0] * self.q[1, 0],
             1 - 2 * self.q[1, 0] * self.q[1, 0] - 2 * self.q[2, 0] * self.q[2, 0]]
        ])

        # self.delta_p = delta_p
        # self.delta_v = delta_v
        # self.delta_ap = delta_ap
        # self.bg = bg
        # self.ba = ba
        self.delta_k = np.matrix([
            delta_p[0], delta_p[1], delta_p[2],
            delta_v[0], delta_v[1], delta_v[2],
            delta_ap[0], delta_ap[1], delta_ap[2]]).T
        self.bias = np.matrix([ba[0], ba[1], ba[2], bg[0], bg[1], bg[2]]).T

    def get_pos(self):
        return self.position

    def get_rotation_matrix(self):
        return self.B2N_matrix

    def next(self, delta_t, frame, exp, sl):
        # w - bg
        gyros = frame.get_gyros()
        tmp = [gyros[0], gyros[1], gyros[2]]
        # C = C + C * cross_product[(w - bg) * delta_t]
        th = np.linalg.norm(tmp) * delta_t
        omg = np.matrix([[0, -tmp[0], -tmp[1], -tmp[2]],
                         [tmp[0], 0, tmp[2], -tmp[1]],
                         [tmp[1], -tmp[2], 0, tmp[0]],
                         [tmp[2], tmp[1], -tmp[0], 0]])
        if th != 0:
            self.q = (np.eye(4) * math.cos(0.5 * th) + delta_t * omg * math.sin(0.5 * th) / th) * self.q
            self.q = self.q / np.linalg.norm(self.q)
        # self.q = self.q + 0.5 * omg * self.q * delta_t
        # self.q = self.q / np.linalg.norm(self.q)


        # exp.add_gyro(self.B2N_matrix * array2matrix(frame.get_gyros()))

        exp.add_angle([math.atan2(2 * (self.q[0, 0] * self.q[1, 0] + self.q[2, 0] * self.q[3, 0]), (1 - 2 * (self.q[1, 0] * self.q[1, 0] + self.q[2, 0] * self.q[2, 0]))) * 180 / math.pi,
                       math.asin(2 * (self.q[0, 0] * self.q[2, 0] - self.q[1, 0] * self.q[3, 0])) * 180 / math.pi,
                       math.atan2(2 * (self.q[0, 0] * self.q[3, 0] + self.q[1, 0] * self.q[2, 0]), (1 - 2 * (self.q[2, 0] * self.q[2, 0] + self.q[3, 0] * self.q[3, 0]))) * 180 / math.pi])

        self.B2N_matrix = np.matrix([
            [1 - 2*self.q[2,0]*self.q[2,0] - 2*self.q[3,0]*self.q[3,0], 2*self.q[1,0]*self.q[2,0] - 2*self.q[0,0]*self.q[3,0], 2*self.q[1,0]*self.q[3,0] + 2*self.q[0,0]*self.q[2,0]],
            [2*self.q[1,0]*self.q[2,0] + 2*self.q[0,0]*self.q[3,0], 1 - 2*self.q[1,0]*self.q[1,0] - 2*self.q[3,0]*self.q[3,0], 2*self.q[2,0]*self.q[3,0] - 2*self.q[0,0]*self.q[1,0]],
            [2*self.q[1,0]*self.q[3,0] - 2*self.q[0,0]*self.q[2,0], 2*self.q[2,0]*self.q[3,0] + 2*self.q[0,0]*self.q[1,0], 1 - 2*self.q[1,0]*self.q[1,0] - 2*self.q[2,0]*self.q[2,0]]
        ])
        # print(self.B2N_matrix * self.B2N_matrix.T)

        # # f - ba
        # accs = frame.get_accs()
        # tmp = [accs[0], accs[1], accs[2]]
        # # v = v + [C * (f - ba) -g] * delta_t
        # if sp != 0:
        #     self.velocity = self.B2N_matrix * array2matrix([0, sp, 0])
        # self.velocity = self.velocity + (self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])) * delta_t

        # p = p + v * delta_t
        if sl != 0:
            self.position = self.position + self.B2N_matrix * np.matrix([0, sl, 0]).T
        # print(self.position)

        print()
        print('p ' + str(self.position.T))
        print('v ' + str(self.velocity.T))
        print('a ' + str(array2matrix(frame.get_accs()).T))
        print('rotation' + str(self.B2N_matrix))
        # print('a ' + str(accs))
        # print('a-ba ' + str(array2matrix(tmp)))
        print('C*a' + str((self.B2N_matrix * array2matrix(frame.get_accs())).T))
        print('C*a ' + str((self.B2N_matrix * array2matrix(tmp)).T))
        print('ad ' + str((self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])).T))
        print()
        exp.add_pos(self.position[0, 0] * 12, self.position[1, 0] * 12)
        exp.add_debug_v(self.delta_k, self.velocity, (self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])))

