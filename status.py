import math
from utils import *
import numpy as np
import cv2
from collections import deque
from math import cos, sin, pi
import matplotlib.pyplot as plt

class Status:
    # G
    g = 9.801

    global_m = [0, 22, -44]
    # 预测阶段白噪声
    # w_p,w_v,w_ap,w_bg,w_ba = [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]

    # ------------------------------------------------------------需要调参--------------------------------------------------------------
    # 测量协方差矩阵R -> 越小越信任观测，稳态噪声（重要！）
    r_v, r_ap, r_vl, r_p, r_a, r_m = [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.2, 0.2, 0.2], [1, 1, 1], [3, 3, 3]

    # 预测协方差矩阵Q -> 越小越信任模型（重要！） 如果没有先验信息，应当适当增大Q的取值
    covariance_q = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0.001, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0.001, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0.001]])

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

    gravity = None

    def __init__(self, position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba):
        self.position = array2matrix(position)
        self.velocity = array2matrix(velocity)
        alpha = 0.022 / 2
        self.q = np.matrix([math.cos(alpha), 0, 0, math.sin(alpha)]).T
        # self.q = np.matrix([1, 0, 0, 0]).T
        self.B2N_matrix = q2R(self.q)

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
        self.WIN_SIZE = 200
        self.delta_t = 1 / 100

        self.init_mag = None
        self.qa_win = deque()
        self.m_win = deque()
        self.rk = np.eye(3) * 0.1

    def get_pos(self):
        return self.position

    def get_rotation_matrix(self):
        return self.B2N_matrix

    def add_sensor_data(self, delta_t, frame):
        gyros = frame.get_gyros()
        tmp = [gyros[0] - self.bias[0, 0], gyros[1] - self.bias[1, 0], gyros[2] - self.bias[2, 0]]
        th = np.linalg.norm(tmp) * delta_t
        sint = tmp / np.linalg.norm(tmp) * sin(th / 2)
        tmpq = np.mat([cos(th / 2), -sint[0], -sint[1], -sint[2]]).T

        if len(self.qa_win) == self.WIN_SIZE:
            self.qa_win.popleft()
            self.m_win.popleft()
        self.qa_win.append(tmpq)
        self.m_win.append(frame.get_mags())

    def next(self, delta_t, frame, exp, begin):
        # w - bg
        gyros = frame.get_gyros()
        tmp = [gyros[0] - self.bias[0, 0], gyros[1] - self.bias[1, 0], gyros[2] - self.bias[2, 0]]

        th = np.linalg.norm(tmp) * delta_t
        sint = tmp / np.linalg.norm(tmp) * sin(th / 2)
        tmpq = np.mat([cos(th / 2), sint[0], sint[1], sint[2]]).T
        self.q = multiple_q(self.q, tmpq)

        # self.m_win.append(np.array(frame.get_mags()) / np.linalg.norm(np.array(frame.get_mags())))

        # C = C + C * cross_product[(w - bg) * delta_t]
        # omg = np.matrix([[0, -tmp[0], -tmp[1], -tmp[2]],
        #                  [tmp[0], 0, tmp[2], -tmp[1]],
        #                  [tmp[1], -tmp[2], 0, tmp[0]],
        #                  [tmp[2], tmp[1], -tmp[0], 0]])
        # if th != 0:
        #     self.q = (np.eye(4) * math.cos(0.5 * th) + delta_t * omg * math.sin(0.5 * th) / th) * self.q
        #     self.q = self.q / np.linalg.norm(self.q)

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
            # self.q = error_q_v * self.q
            self.q = multiple_q(error_q, self.q)

        exp.add_gyro(self.B2N_matrix * array2matrix(frame.get_gyros()))

        euler_ang = np.array(frame.get_angle()) * 180 / math.pi
        exp.add_mag_ang(euler_ang)
        # calculate the angle
        # if self.gravity is None:
        #     self.gravity = np.array(frame.get_accs())
        # else:
        #     self.gravity = self.gravity * 0.7 + 0.3 * np.array(frame.get_accs())
        # self.gravity = self.gravity / np.linalg.norm(self.gravity)
        # frame_mag = np.array(frame.get_mags())
        # gravity_mag = self.gravity * np.dot(frame_mag, self.gravity)
        # direct_mag = frame_mag - gravity_mag
        # direct_mag = direct_mag / np.linalg.norm(direct_mag)
        # mag_ang = q2ang(vec_2q(direct_mag, np.array([0, 1, 0])))

        if begin:
            my_angle = q2ang(self.q)
            exp.add_angle(my_angle)
            exp.add_yaw_angle(euler_ang[2], my_angle[2])
        else:
            exp.add_angle(euler_ang)

        self.B2N_matrix = q2R(self.q)

        # f - ba
        accs = frame.get_accs()
        tmpa = [accs[0] - self.bias[3, 0], accs[1] - self.bias[4, 0], accs[2] - self.bias[5, 0]]
        # v = v + [C * (f - ba) -g] * delta_t
        check = self.B2N_matrix * array2matrix(tmpa) - array2matrix([0, 0, self.g])
        # check2 = (self.B2N_matrix * array2matrix(tmpa) - array2matrix([0, 0, self.g])) * delta_t
        if np.linalg.norm(check) > 1:
            self.velocity = self.velocity + (self.B2N_matrix * array2matrix(tmpa) - array2matrix([0, 0, self.g])) * delta_t
        self.velocity = self.velocity - array2matrix([self.delta_k[3, 0], self.delta_k[4, 0], self.delta_k[5, 0]])

        # p = p + v * delta_t
        self.position = self.position + self.velocity * delta_t
        self.position = self.position - array2matrix([self.delta_k[0, 0], self.delta_k[1, 0], self.delta_k[2, 0]])

        # self.delta_k = np.matrix([0,0,0,0,0,0,0,0,0]).T
        # print()
        # print('p ' + str(self.position.T))
        # print('v ' + str(self.velocity.T))
        # print('rotation' + str(self.B2N_matrix))
        # print('a ' + str(array2matrix(frame.get_accs()).T))
        # print('C*a ' + str((self.B2N_matrix * array2matrix(tmp)).T))
        # print('ad ' + str((self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])).T))
        # print()
        exp.add_pos(self.position[0, 0] * 12, self.position[1, 0] * 12)
        exp.add_velocity([self.velocity[0, 0], self.velocity[1, 0], self.velocity[2, 0]])
        exp.add_debug_v(self.delta_k, self.velocity, (self.B2N_matrix * array2matrix(tmp) - array2matrix([0, 0, self.g])))

        if np.linalg.norm(np.array(frame.get_mags())) > 20:
            if self.init_mag is None:
                self.init_mag = np.mat(frame.get_mags()).T
            else:
                tmpq = np.mat([cos(th / 2), -sint[0], -sint[1], -sint[2]]).T
                tmpR = q2R(tmpq)
                self.init_mag = tmpR * self.init_mag
            exp.add_raw_mag(frame.get_mags())
            exp.add_cal_mag(self.init_mag)


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
        # print(self.delta_k.T)

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
        print("speed: " + str(self.B2N_matrix * array2matrix([0, v_pdr, 0])) + '\n' + str(N2B_matrix * self.velocity))
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

        tmp = cross_product(tmp)

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

    def correct_by_mag(self, mb):

        # mb = array2matrix(frame.get_mags())
        # gyros = frame.get_gyros()
        # tmp = [gyros[0] - self.bias[0, 0], gyros[1] - self.bias[1, 0], gyros[2] - self.bias[2, 0]]
        # tmp = [x * 0.01 for x in tmp]
        # R = np.matrix([
        #     [cos(tmp[0]) * cos(tmp[1]), cos(tmp[0]) * sin(tmp[1]) * sin(tmp[2]) - sin(tmp[0]) * cos(tmp[2]),
        #      cos(tmp[0]) * sin(tmp[1]) * cos(tmp[2]) + sin(tmp[0]) * sin(tmp[2])],
        #     [sin(tmp[0]) * cos(tmp[1]), cos(tmp[0]) * cos(tmp[2]) + sin(tmp[0]) * sin(tmp[1]) * sin(tmp[2]),
        #      sin(tmp[0]) * sin(tmp[1]) * cos(tmp[2]) - cos(tmp[2]) * sin(tmp[2])],
        #     [-sin(tmp[1]), cos(tmp[1]) * sin(tmp[2]), cos(tmp[1]) * cos(tmp[2])]
        # ])
        # self.sliding_win_r.append(R)
        # self.sliding_win.append(mb)
        # if len(self.sliding_win_r) == self.WIN_SIZE:
        #     first_mag = self.sliding_win.popleft()
        #     f = first_mag
        #     self.sliding_win_r.popleft()
        #
        #     for r in self.sliding_win_r:
        #         first_mag = r * first_mag
        #     # c = first_mag.T.dot(mb) / (np.linalg.norm(first_mag) * np.linalg.norm(mb))
        #     # c = f.T.dot(mb) / (np.linalg.norm(f) * np.linalg.norm(mb))
        #     c = mb.T.dot(f) / (np.linalg.norm(mb) * np.linalg.norm(f))
        #
        #     exp.add_mag_ang(math.acos(c) / math.pi * 180)


        # mb = array2matrix(frame.get_mags())
        # if self.init_mag is None and np.linalg.norm(mb) > 30:
        #     self.init_mag = mb
        #     self.ma = mb
        # elif self.init_mag is not None:
        #     gyros = frame.get_gyros()
        #     tmp = [gyros[0] - self.bias[0, 0], gyros[1] - self.bias[1, 0], gyros[2] - self.bias[2, 0]]
        #     tmp = [x * 0.01 for x in tmp]
        #     R = np.matrix([
        #         [cos(tmp[0])*cos(tmp[1]), cos(tmp[0])*sin(tmp[1])*sin(tmp[2])-sin(tmp[0])*cos(tmp[2]), cos(tmp[0])*sin(tmp[1])*cos(tmp[2])+sin(tmp[0])*sin(tmp[2])],
        #         [sin(tmp[0])*cos(tmp[1]), cos(tmp[0])*cos(tmp[2])+sin(tmp[0])*sin(tmp[1])*sin(tmp[2]), sin(tmp[0])*sin(tmp[1])*cos(tmp[2])-cos(tmp[2])*sin(tmp[2])],
        #         [-sin(tmp[1]), cos(tmp[1])*sin(tmp[2]), cos(tmp[1])*cos(tmp[2])]
        #     ])
        #     self.init_mag = R * self.init_mag
        #     c = self.init_mag.T.dot(mb) / (np.linalg.norm(self.init_mag) * np.linalg.norm(mb))
        #     exp.add_mag_ang(math.acos(c) / math.pi * 180)

        # ang = frame.get_angle()
        # self.exp.add_mag_ang(ang / math.pi * 180)
        self.smooth()

        # tmp = self.B2N_matrix * mb
        #
        # earth_mag = array2matrix([-32.187, -4.562, -30.625])
        # earth_mag = earth_mag / np.linalg.norm(earth_mag)
        #
        # # 测量值
        # y = tmp - earth_mag
        #
        # tmp = cross_product([tmp[0, 0], tmp[1, 0], tmp[2, 0]])
        # H = np.matrix([
        #     [0, 0, 0, 0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2]],
        #     [0, 0, 0, 0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2]],
        #     [0, 0, 0, 0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2]]
        # ])
        #
        # # 量测噪声
        # covariance_r = np.diag([self.r_m[0], self.r_m[1], self.r_m[2]])
        #
        # # 卡尔曼增益
        # K = self.covariance * H.T * np.linalg.pinv(H * self.covariance * H.T + covariance_r)
        #
        # self.delta_k = self.delta_k + K * (y - H * self.delta_k)
        #
        # self.covariance = (np.eye(9) - K * H) * self.covariance
        # print("mag " + str(self.delta_k.T))

    def smooth(self):
        # mag point to north
        global_m = self.global_m
        global_m = global_m / np.linalg.norm(global_m)

        covariance = np.eye(3) * 0.1
        q_cov = np.eye(3) * 0.01
        # r_cov = np.eye(3) * 0.01
        q = self.q
        # global_m = np.mat([1, 0, 0])

        # th_win contains rotation matrix among the window
        self.qa_win.reverse()
        self.m_win.reverse()
        for i in range(self.WIN_SIZE):
            qa = self.qa_win[i]
            qa = -qa
            qa[0, 0] = -qa[0, 0]
            q = multiple_q(q, qa)

            # calculate the rotation(qm) between device_m ang global m
            # qm = vec_2q(self.m_win[i], global_m)

            # B2N = q2R(q)
            # tmp_mag = B2N * np.mat(self.m_win[i]).T
            # H = cross_product(tmp_mag)

            # covariance = covariance + q_cov
            # k = covariance * H.T * np.linalg.pinv(H * covariance * H.T + self.rk)
            # fi = k * (tmp_mag - global_m)
            # norm = np.linalg.norm(fi)
            # error_q = np.mat([cos(norm/2), fi[0] * sin(norm/2), fi[1] * sin(norm/2), fi[2] * sin(norm/2)]).T
            # q = multiple_q(error_q, q)
            # covariance = (np.eye(3) - k * H) * covariance

        self.qa_win.reverse()
        self.m_win.reverse()
        self.sage_husa(q, self.WIN_SIZE)

    def sage_husa(self, q, win_size):
        # mag point to north
        # global_m = np.mat([[3.62236081, -27.84858828, -32.88457262]]).T

        # global_m = np.mat([[-32.187, -4.562, -30.625]]).T
        global_m = np.mat(self.global_m).T
        # global_m = global_m / np.linalg.norm(global_m)

        # adaptive factor
        beta = 1
        b = 0.98
        rk = self.rk

        covariance = np.zeros([3, 3])

        q_cov = np.eye(3) * 0.1
        cal_mag = []
        delta_vec = np.mat([0, 0, 0]).T
        for i in range(win_size):
            # calculate rk


            # tmp_mag = (np.eye(3) - cross_product(delta_vec)) * B2N * np.mat(self.m_win[i]).T
            ang = q2ang(multiple_q(vec2q(delta_vec), q))
            no_bias_R = q2R(multiple_q(vec2q(delta_vec), q))
            tmp_mag = no_bias_R * np.mat(self.m_win[i]).T
            cal_mag.append(tmp_mag)
            H = cross_product(no_bias_R * np.mat(self.m_win[i]).T)

            zk = tmp_mag - global_m - H * delta_vec
            tmpr = zk * zk.T - H * covariance * H.T

            rk = self.merge(rk, tmpr, beta)
            beta = beta / (beta + b)

            # ekf
            covariance = covariance + q_cov
            k = covariance * H.T * np.linalg.pinv(H * covariance * H.T + rk)
            covariance = (np.eye(3) - k * H) * covariance
            delta_vec = delta_vec + k * zk

            qa = self.qa_win[i]
            q = multiple_q(q, qa)
            # q = multiple_q(error_q, q)

        # fig = plt.figure('x')
        # t = np.arange(0, len(cal_mag) * 0.01, 0.01)
        # plt.plot(t, np.array(cal_mag)[:, 0], 'r:')
        # plt.plot(t, np.array(self.m_win)[:, 0], "b:")
        # fig = plt.figure('y')
        # t = np.arange(0, len(cal_mag) * 0.01, 0.01)
        # plt.plot(t, np.array(cal_mag)[:, 1], 'r:')
        # plt.plot(t, np.array(self.m_win)[:, 1], "b:")
        # fig = plt.figure('z')
        # t = np.arange(0, len(cal_mag) * 0.01, 0.01)
        # plt.plot(t, np.array(cal_mag)[:, 2], 'r:')
        # plt.plot(t, np.array(self.m_win)[:, 2], "b:")
        # plt.show()

        # # ekf
        # B2N = q2R(q)
        # tmp_mag = B2N * np.mat(self.m_win[-1]).T
        # H = cross_product(tmp_mag)
        # zk = tmp_mag - global_m

        # covariance = covariance + q_cov
        # k = covariance * H.T * np.linalg.pinv(H * covariance * H.T + rk)
        # delta_vec = k * zk
        # norm = np.linalg.norm(delta_vec)
        # delta_vec = delta_vec / norm
        # error_q = np.mat([cos(norm / 2), delta_vec[0] * sin(norm / 2), delta_vec[1] * sin(norm / 2),
        #                   delta_vec[2] * sin(norm / 2)]).T
        # q = multiple_q(error_q, q)
        # covariance = (np.eye(3) - k * H) * covariance
        # self.rk = rk
        error_q = vec2q(delta_vec)
        q = multiple_q(error_q, q)
        self.q = q

    def merge(self, rk, newRk, beta):
        r_max = 81
        r_min = 0.01
        if rk is None:
            rk = newRk

        for i in range(3):
            for j in range(3):
                if i != j:
                    rk[i, j] = 0
                else:
                    if newRk[i, j] > r_max:
                        rk[i, j] = r_max
                    elif newRk[i, j] < r_min:
                        rk[i, j] = (1 - beta) * rk[i, j] + beta * r_min
                    else:
                        rk[i, j] = (1 - beta) * rk[i, j] + beta * newRk[i, j]
        return rk


