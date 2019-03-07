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
    # 测量协方差矩阵R -> 越小越信任观测，稳态噪声（重要！）
    r_v, r_ap, r_vl, r_p, r_a, r_m = [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.2, 0.2, 0.2], [0.01, 0.1, 0.1], [0.01, 0.01, 0.01]

    # 预测协方差矩阵Q -> 越小越信任模型（重要！） 如果没有先验信息，应当适当增大Q的取值
    covariance_q = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 3, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 3, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 3, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0.03, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0.03, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0.03]])

    # 状态delta_k的协方差矩阵P -> 决定瞬态过程收敛速率，稳态过程中的P由QR决定
    covariance = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]]) * 0.1
    # ---------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba):
        self.position = array2matrix(position)
        self.velocity = array2matrix(velocity)
        alpha = 0.038 / 2
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
        # self.B2N_matrix = rotation_matrix

        # self.delta_p = delta_p
        # self.delta_v = delta_v
        # self.delta_ap = delta_ap
        # self.bg = bg
        # self.ba = ba
        self.delta_k = np.matrix([
            delta_p[0], delta_p[1], delta_p[2],
            delta_v[0], delta_v[1], delta_v[2],
            delta_ap[0], delta_ap[1], delta_ap[2]]).T
        self.bias = np.matrix([bg[0], bg[1], bg[2], ba[0], ba[1], ba[2]]).T

    def get_pos(self):
        return self.position

    def get_rotation_matrix(self):
        return self.B2N_matrix

    def next(self, delta_t, frame, exp):
        # w - bg
        gyros = frame.get_gyros()
        tmp = [gyros[0] - self.bias[0, 0], gyros[1] - self.bias[1, 0], gyros[2] - self.bias[2, 0]]
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

        # rotation matrix of attitude error
        error_vector = array2matrix([self.delta_k[6, 0], self.delta_k[7, 0], self.delta_k[8, 0]])
        error_mod = np.linalg.norm(error_vector)
        if error_mod != 0:
            error_vector = error_vector / error_mod
            error_q = np.matrix([math.cos(error_mod / 2),
                       error_vector[0, 0] * math.sin(error_mod / 2),
                       error_vector[1, 0] * math.sin(error_mod / 2),
                       error_vector[2, 0] * math.sin(error_mod / 2)]).T
            error_q_v = [error_q[0, 0], error_q[1, 0], error_q[2, 0], error_q[3, 0]]
            # self.q = error_q_v * self.q
            self.q = np.matrix([error_q_v[0] * self.q[0, 0] - error_q_v[1] * self.q[1, 0] - error_q_v[2] * self.q[2, 0] - error_q_v[3] * self.q[3, 0],
                                   error_q_v[0] * self.q[1, 0] + error_q_v[1] * self.q[0, 0] + error_q_v[2] * self.q[3, 0] - error_q_v[3] * self.q[2, 0],
                                   error_q_v[0] * self.q[2, 0] - error_q_v[1] * self.q[3, 0] + error_q_v[2] * self.q[0, 0] + error_q_v[3] * self.q[1, 0],
                                   error_q_v[0] * self.q[3, 0] + error_q_v[1] * self.q[2, 0] - error_q_v[2] * self.q[1, 0] + error_q_v[3] * self.q[0, 0]]).T
            self.q = self.q / np.linalg.norm(self.q)

        # exp.add_gyro(self.B2N_matrix * array2matrix(frame.get_gyros()))

        exp.add_angle([math.atan2(2 * (self.q[0, 0] * self.q[1, 0] + self.q[2, 0] * self.q[3, 0]), (1 - 2 * (self.q[1, 0] * self.q[1, 0] + self.q[2, 0] * self.q[2, 0]))) * 180 / math.pi,
                       math.asin(2 * (self.q[0, 0] * self.q[2, 0] - self.q[1, 0] * self.q[3, 0])) * 180 / math.pi,
                       math.atan2(2 * (self.q[0, 0] * self.q[3, 0] + self.q[1, 0] * self.q[2, 0]), (1 - 2 * (self.q[2, 0] * self.q[2, 0] + self.q[3, 0] * self.q[3, 0]))) * 180 / math.pi])

        self.B2N_matrix = np.matrix([
            [1 - 2*self.q[2,0]*self.q[2,0] - 2*self.q[3,0]*self.q[3,0], 2*self.q[1,0]*self.q[2,0] - 2*self.q[0,0]*self.q[3,0], 2*self.q[1,0]*self.q[3,0] + 2*self.q[0,0]*self.q[2,0]],
            [2*self.q[1,0]*self.q[2,0] + 2*self.q[0,0]*self.q[3,0], 1 - 2*self.q[1,0]*self.q[1,0] - 2*self.q[3,0]*self.q[3,0], 2*self.q[2,0]*self.q[3,0] - 2*self.q[0,0]*self.q[1,0]],
            [2*self.q[1,0]*self.q[3,0] - 2*self.q[0,0]*self.q[2,0], 2*self.q[2,0]*self.q[3,0] + 2*self.q[0,0]*self.q[1,0], 1 - 2*self.q[1,0]*self.q[1,0] - 2*self.q[2,0]*self.q[2,0]]
        ])
        print(self.B2N_matrix * self.B2N_matrix.T)

        # f - ba
        accs = frame.get_accs()
        tmp = [accs[0] - self.bias[3, 0], accs[1] - self.bias[4, 0], accs[2] - self.bias[5, 0]]
        # v = v + [C * (f - ba) -g] * delta_t
        self.velocity = self.velocity + (self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])) * delta_t
        self.velocity = self.velocity - array2matrix([self.delta_k[3, 0], self.delta_k[4, 0], self.delta_k[5, 0]])

        # p = p + v * delta_t
        self.position = self.position + self.velocity * delta_t
        print(self.position)
        self.position = self.position - array2matrix([self.delta_k[0, 0], self.delta_k[1, 0], self.delta_k[2, 0]])

        # self.delta_k = np.matrix([0,0,0,0,0,0,0,0,0]).T
        print()
        print('p ' + str(self.position.T))
        print('v ' + str(self.velocity.T))
        print('rotation' + str(self.B2N_matrix))
        # print('a ' + str(accs))
        # print('a-ba ' + str(array2matrix(tmp)))
        # print('C*a' + str((self.B2N_matrix * array2matrix(frame.get_accs())).T))
        print('a ' + str(array2matrix(frame.get_accs()).T))
        print('C*a ' + str((self.B2N_matrix * array2matrix(tmp)).T))
        print('ad ' + str((self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])).T))
        print()
        exp.add_pos(self.position[0, 0] * 12, self.position[1, 0] * 12)
        exp.add_debug_v(self.delta_k, self.velocity, (self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])))

    def next_delta(self, delta_t, frame):
        # (f_nX) * delta_t
        fnt = cross_product(self.B2N_matrix * array2matrix(frame.get_accs())) * delta_t
        # C * delta_t
        ct = self.B2N_matrix * delta_t

        # 15 * 15
        updater = np.matrix([
            # [1, 0, 0,  delta_t, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0],
            # [0, 1, 0,  0, delta_t, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0],
            # [0, 0, 1,  0, 0, delta_t,  0, 0, 0,  0, 0, 0,  0, 0, 0],
            #
            # [0, 0, 0,  1, 0, 0,  fnt[0, 0], fnt[0, 1], fnt[0, 1],  0, 0, 0,  ct[0, 0], ct[0, 1], ct[0, 2]],
            # [0, 0, 0,  0, 1, 0,  fnt[1, 0], fnt[1, 1], fnt[1, 2],  0, 0, 0,  ct[1, 0], ct[1, 1], ct[1, 2]],
            # [0, 0, 0,  0, 0, 1,  fnt[2, 0], fnt[2, 1], fnt[2, 2],  0, 0, 0,  ct[2, 0], ct[2, 1], ct[2, 2]],
            #
            # [0, 0, 0,  0, 0, 0,  1, 0, 0,  -ct[0, 0], -ct[0, 1], -ct[0, 2],  0, 0, 0],
            # [0, 0, 0,  0, 0, 0,  0, 1, 0,  -ct[1, 0], -ct[1, 1], -ct[1, 2],  0, 0, 0],
            # [0, 0, 0,  0, 0, 0,  0, 0, 1,  -ct[2, 0], -ct[2, 1], -ct[2, 2],  0, 0, 0],
            #
            # [0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 0, 0,  0, 0, 0],
            # [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0,  0, 0, 0],
            # [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1,  0, 0, 0],
            #
            # [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 0, 0],
            # [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0],
            # [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1]
            [0, 0, 0, delta_t, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, delta_t, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, delta_t, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, fnt[0, 0], fnt[0, 1], fnt[0, 1]],
            [0, 0, 0, 0, 0, 0, fnt[1, 0], fnt[1, 1], fnt[1, 2]],
            [0, 0, 0, 0, 0, 0, fnt[2, 0], fnt[2, 1], fnt[2, 2]],

            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bias_updater = np.matrix([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, ct[0, 0], ct[0, 1], ct[0, 2]],
            [0, 0, 0, ct[1, 0], ct[1, 1], ct[1, 2]],
            [0, 0, 0, ct[2, 0], ct[2, 1], ct[2, 2]],
            [-ct[0, 0], -ct[0, 1], -ct[0, 2], 0, 0, 0],
            [-ct[1, 0], -ct[1, 1], -ct[1, 2], 0, 0, 0],
            [-ct[2, 0], -ct[2, 1], -ct[2, 2], 0, 0, 0],
        ])

        self.delta_k = updater * self.delta_k + bias_updater * self.bias
        self.covariance = updater * self.covariance * updater.T + self.covariance_q
        print(self.delta_k.T)

    def correct_by_zupt(self):
        H = np.matrix([
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
        ])
        # 量测噪声
        covariance_r = np.diag([self.r_v[0], self.r_v[1], self.r_v[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)

        # 测量值
        y = self.velocity - np.matrix([0, 0, 0]).T

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(9) - K * H) * self.covariance
        print("zupt " + str(self.delta_k.T))

    def correct_by_zaru(self, first_epoch_rotation):
        # 描述非常模糊，且使用到ins计算出的航向角，但是系统中并没有对角速度积分

        H = np.matrix([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])
        # 量测噪声
        covariance_r = np.diag([self.r_ap[0], self.r_ap[1], self.r_ap[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)

        # 测量值
        rotation = cv2.Rodrigues(self.get_rotation_matrix() * first_epoch_rotation.I)[0]

        self.delta_k = self.delta_k + K * (rotation - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance
        print("zaru " + str(self.delta_k.T))

    def correct_by_velocity(self, step_speed):
        v_pdr = step_speed

        # 通过导航坐标系旋转矩阵旋转计算行人坐标系旋转矩阵
        # angle = math.atan2(self.velocity[1, 0], self.velocity[0, 0])

        # B2L_matrix = self.B2N_matrix * np.matrix([
        #     [math.cos(angle), math.sin(angle), 0],
        #     [-math.sin(angle), math.cos(angle), 0],
        #     [0, 0 ,1]
        # ])
        B2L_matrix = np.eye(3)

        # N -> B
        N2B_matrix = self.B2N_matrix.T
        
        tmp = B2L_matrix * N2B_matrix
        tmp2 = B2L_matrix * N2B_matrix * cross_product([self.velocity[0, 0], self.velocity[1, 0], self.velocity[2, 0]])

        H = np.matrix([
            [0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2], -tmp2[0, 0], -tmp2[0, 1], -tmp2[0, 2]],
            [0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2], -tmp2[1, 0], -tmp2[1, 1], -tmp2[1, 2]],
            [0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2], -tmp2[2, 0], -tmp2[2, 1], -tmp2[2, 2]]
        ])

        # 量测噪声
        covariance_r = np.diag([self.r_vl[0], self.r_vl[1], self.r_vl[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)

        # 测量值
        print("speed: " + str(self.B2N_matrix * array2matrix([0, v_pdr, 0])))
        y = B2L_matrix * N2B_matrix * self.velocity - array2matrix([0, v_pdr, 0])

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(9) - K * H) * self.covariance
        print("step-v " + str(self.delta_k.T))

    def correct_by_step_length(self, step_length, pos, rotation):
        pos = pos + rotation * array2matrix([0, step_length, 0])

        H = np.matrix([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        # 量测噪声
        covariance_r = np.diag([self.r_p[0], self.r_p[1], self.r_p[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)
        
        # 卡尔曼增益
        K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)

        # 测量值
        y = self.position - pos

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(9) - K * H) * self.covariance
        print("step-l " + str(self.delta_k.T))

    def correct_by_gravity(self, frame):
        fb = array2matrix(frame.get_accs())
        
        tmp = self.B2N_matrix * fb

        # 测量值
        # y = tmp - array2matrix([self.delta_k[12, 0], self.delta_k[13, 0], self.delta_k[14, 0]]) - array2matrix([0, 0, self.g])
        y = tmp - array2matrix([0, 0, self.g])

        tmp = cross_product([tmp[0, 0], tmp[1, 0], tmp[2, 0]])

        H = np.matrix([
            # [0, 0, 0, 0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2], 0, 0, 0, self.B2N_matrix[0, 0], self.B2N_matrix[0, 1], self.B2N_matrix[0, 2]],
            # [0, 0, 0, 0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2], 0, 0, 0, self.B2N_matrix[1, 0], self.B2N_matrix[1, 1], self.B2N_matrix[1, 2]],
            # [0, 0, 0, 0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2], 0, 0, 0, self.B2N_matrix[2, 0], self.B2N_matrix[2, 1], self.B2N_matrix[2, 2]]
            [0, 0, 0, 0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2]],
            [0, 0, 0, 0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2]],
            [0, 0, 0, 0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2]]
        ])

        # 量测噪声
        covariance_r = np.diag([self.r_a[0], self.r_a[1], self.r_a[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)
        
        # 卡尔曼增益
        K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(9) - K * H) * self.covariance
        print("g " + str(self.delta_k.T))

    def correct_by_mag(self, frame, mag):
        mb = array2matrix(frame.get_mags())

        tmp = self.B2N_matrix * mb

        # 测量值
        y = tmp - mag
        if abs(y[0]) > 10 or abs(y[1]) > 10 or abs(y[2]) > 10:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(y)
        tmp = cross_product([tmp[0, 0], tmp[1, 0], tmp[2, 0]])

        H = np.matrix([
            [0, 0, 0, 0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2], 0, 0, 0, 0, 0, 0]
        ])

        # 量测噪声
        covariance_r = np.diag([self.r_m[0], self.r_m[1], self.r_m[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance
        print("mag " + str(self.delta_k.T))

