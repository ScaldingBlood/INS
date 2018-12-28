import math
from utils import *
import cv2


class Status:
    # G
    g = 9.801
    # 预测阶段白噪声
    # w_p,w_v,w_ap,w_bg,w_ba = [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]
    # 纠正阶段白噪声
    v_v, v_ap, v_vl, v_p, v_a, v_m = [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.015, 0.015, 0.015], [0.01, 0.01, 0.01], [0.015, 0.015, 0.015], [0.01, 0.01, 0.01]
    # 干扰协方差矩阵Q 
    covariance_q = np.eye(15) * 0.01


    def __init__(self, position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba):
        self.position = array2matrix(position)
        self.velocity = array2matrix(velocity)
        self.B2N_matrix = rotation_matrix

        # self.delta_p = delta_p
        # self.delta_v = delta_v
        # self.delta_ap = delta_ap
        # self.bg = bg
        # self.ba = ba
        self.delta_k = np.matrix([
            delta_p[0], delta_p[1], delta_p[2],
            delta_v[0], delta_v[1], delta_v[2],
            delta_ap[0], delta_ap[1], delta_ap[2],
            bg[0], bg[1], bg[2],
            ba[0], ba[1], ba[2]]).T
        
        # 状态delta_k的协方差矩阵
        self.covariance = np.zeros((15, 15))

    
    def get_pos(self):
        return self.position

    def get_rotation_matrix(self):
        return self.B2N_matrix


    def next(self, delta_t, frame):
        # w - bg
        gyros = frame.get_gyros()
        tmp = [gyros[0] - self.delta_k[9, 0], gyros[1] - self.delta_k[10, 0],gyros[2] - self.delta_k[11, 0]]
        # C = C + C * cross_product[(w - bg) * delta_t]
        self.B2N_matrix = self.B2N_matrix + self.B2N_matrix * cross_product(tmp) * delta_t
        # C = (I - cross_product(delta_ap)) * C
        self.B2N_matrix = (np.eye(3) - cross_product([self.delta_k[6, 0], self.delta_k[7, 0], self.delta_k[8, 0]])) * self.B2N_matrix

        # f - ba
        accs = frame.get_accs()
        tmp = [accs[0] - self.delta_k[12, 0], accs[1] - self.delta_k[13, 0], accs[2] - self.delta_k[14, 0]]
        # v = v + [C * (f - ba) -g] * delta_t
        self.velocity = self.velocity + (self.B2N_matrix * array2matrix(tmp) - np.matrix([0, 0, -9.801])) * delta_t
        self.velocity = self.velocity - array2matrix([self.delta_k[3, 0], self.delta_k[4, 0], self.delta_k[5, 0]])

        # p = p + v * delta_t
        self.position = self.position + self.velocity * delta_t
        self.position = self.position - array2matrix([self.delta_k[0, 0], self.delta_k[1, 0], self.delta_k[2, 0]])


    def next_delta(self, delta_t, frame):
        # (f_nX) * delta_t
        fnt = cross_product(self.B2N_matrix * array2matrix(frame.get_accs())) * delta_t
        # C * delta_t
        ct = self.B2N_matrix * delta_t

        # 15 * 15
        updater = np.matrix([
            [1, 0, 0,  delta_t, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0],
            [0, 1, 0,  0, delta_t, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0],
            [0, 0, 1,  0, 0, delta_t,  0, 0, 0,  0, 0, 0,  0, 0, 0],

            [0, 0, 0,  1, 0, 0,  fnt[0, 0], fnt[0, 1], fnt[0, 1],  0, 0, 0,  ct[0, 0], ct[0, 1], ct[0, 2]],
            [0, 0, 0,  0, 1, 0,  fnt[1, 0], fnt[1, 1], fnt[1, 2],  0, 0, 0,  ct[1, 0], ct[1, 1], ct[1, 2]],
            [0, 0, 0,  0, 0, 1,  fnt[2, 0], fnt[2, 1], fnt[2, 2],  0, 0, 0,  ct[2, 0], ct[2, 1], ct[2, 2]],

            [0, 0, 0,  0, 0, 0,  1, 0, 0,  -ct[0, 0], -ct[0, 1], -ct[0, 2],  0, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 1, 0,  -ct[1, 0], -ct[1, 1], -ct[1, 2],  0, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 1,  -ct[2, 0], -ct[2, 1], -ct[2, 2],  0, 0, 0],

            [0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 0, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1,  0, 0, 0],

            [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1]
        ])

        # # current delta
        # delta_k = np.matrix([
        #     delta_p[0], delta_p[1], delta_p[2],
        #     delta_v[0], delta_v[1], delta_v[2],
        #     delta_ap[0], delta_ap[1], delta_ap[2],
        #     bg[0], bg[1], bg[2],
        #     ba[0], ba[1], ba[2]]).T

        # self.delta_k = updater * self.delta_k + np.matrix([w_p[0],w_p[1],w_p[2],w_v[0],w_v[1],w_v[2],w_ap[0],w_ap[1],w_ap[2],w_bg[0],w_bg[1],w_bg[2],w_ba[0],w_ba[1],w_ba[2]])
        self.delta_k = updater * self.delta_k
        self.covariance = updater * self.covariance * updater.T + self.covariance_q


    # def update_delta(delta_k):
    #     self.delta_p = [delta_k[0, 0], delta_k[1, 0], delta_k[2, 0]]
    #     self.delta_v = [delta_k[3, 0], delta_k[4, 0], delta_k[5, 0]]
    #     self.delta_ap = [delta_k[6, 0], delta_k[7, 0], delta_k[8, 0]]
    #     self.bg = [delta_k[9, 0], delta_k[10, 0], delta_k[11, 0]]
    #     self.ba = [delta_k[12, 0], delta_k[13, 0], delta_k[14, 0]]


    def correct_by_zupt(self):
        H = np.matrix([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        # 量测噪声
        covariance_r = np.diag([self.v_v[0], self.v_v[1], self.v_v[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)
        
        # 卡尔曼增益
        K = self.covariance * H.T * ((H * self.covariance * H.T + covariance_r).I)

        # 测量值
        y = self.velocity - np.matrix([0, 0, 0]).T

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance

    
    def correct_by_zaru(self, first_epoch_rotation):
        # 描述非常模糊，且使用到ins计算出的航向角，但是系统中并没有对角速度积分

        H = np.matrix([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])
        # 量测噪声
        covariance_r = np.diag([self.v_ap[0], self.v_ap[1], self.v_ap[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * ((H * self.covariance * H.T + covariance_r).I)

        # 测量值
        rotation = cv2.Rodrigues(self.get_rotation_matrix() * (first_epoch_rotation.I))[0]

        self.delta_k = self.delta_k + K * (rotation - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance


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
        N2B_matrix = self.B2N_matrix.I
        
        tmp = B2L_matrix * N2B_matrix
        tmp2 = B2L_matrix * N2B_matrix * cross_product([self.velocity[0, 0], self.velocity[1, 0], self.velocity[2, 0]])

        H = np.matrix([
            [0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2], -tmp2[0, 0], -tmp2[0, 1], -tmp2[0, 2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2], -tmp2[1, 0], -tmp2[1, 1], -tmp2[1, 2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2], -tmp2[2, 0], -tmp2[2, 1], -tmp2[2, 2], 0, 0, 0, 0, 0, 0]
        ])

        # 量测噪声
        covariance_r = np.diag([self.v_vl[0], self.v_vl[1], self.v_vl[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * (H * self.covariance * H.T + covariance_r).I

        # 测量值
        y = self.velocity - np.matrix([v_pdr, 0, 0]).T

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance


    def correct_by_step_length(self, step_length, pos):
        if pos is None:
            return self.position
        # angle = math.atan2(self.velocity[1, 0], self.velocity[0, 0])
        angle = 0
        pos = pos + np.matrix([step_length * math.cos(angle), step_length * math.sin(angle), 0]).T

        H = np.matrix([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        # 量测噪声
        covariance_r = np.diag([self.v_p[0], self.v_p[1], self.v_p[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)
        
        # 卡尔曼增益
        K = self.covariance * H.T * (H * self.covariance * H.T + covariance_r).I

        # 测量值
        y = self.position - pos

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance


    def correct_by_gravity(self, frame):
        fb = array2matrix(frame.get_accs())
        
        tmp = self.B2N_matrix * fb

        # 测量值
        y = tmp - array2matrix([0, 0, -9.801])

        tmp = cross_product([tmp[0, 0], tmp[1, 0], tmp[2, 0]])

        H = np.matrix([
            [0, 0, 0, 0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2], 0, 0, 0, self.B2N_matrix[0, 0], self.B2N_matrix[0, 1], self.B2N_matrix[0, 2]],
            [0, 0, 0, 0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2], 0, 0, 0, self.B2N_matrix[1, 0], self.B2N_matrix[1, 1], self.B2N_matrix[1, 2]],
            [0, 0, 0, 0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2], 0, 0, 0, self.B2N_matrix[2, 0], self.B2N_matrix[2, 1], self.B2N_matrix[2, 2]]
        ])

        # 量测噪声
        covariance_r = np.diag([self.v_a[0], self.v_a[1], self.v_a[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)
        
        # 卡尔曼增益
        K = self.covariance * H.T * (H * self.covariance * H.T + covariance_r).I

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance


    def correct_by_mag(self, frame, mag):
        mb = array2matrix(frame.get_mags())

        tmp = self.B2N_matrix * mb

        # 测量值
        y = tmp - array2matrix(mag)

        tmp = cross_product([tmp[0, 0], tmp[1, 0], tmp[2, 0]])

        H = np.matrix([
            [0, 0, 0, 0, 0, 0, tmp[0, 0], tmp[0, 1], tmp[0, 2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, tmp[1, 0], tmp[1, 1], tmp[1, 2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, tmp[2, 0], tmp[2, 1], tmp[2, 2], 0, 0, 0, 0, 0, 0]
        ])

        # 量测噪声
        covariance_r = np.diag([self.v_m[0], self.v_m[1], self.v_m[2]])
        covariance_r = np.multiply(covariance_r, covariance_r)

        # 卡尔曼增益
        K = self.covariance * H.T * (H * self.covariance * H.T + covariance_r).I

        self.delta_k = self.delta_k + K * (y - H * self.delta_k)

        self.covariance = (np.eye(15) - K * H) * self.covariance
