import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img
from utils import *
import math
import pandas as pd


class Experiment:
    avg_acc_list = []
    sum_delta = []  # 1e-5
    sum_gryo = []  # 1e-7
    res_to_judge = []

    avg_gyo_list = []  # 0.00072
    mag_gyo_list = []  # -0.03
    mag_res_to_judge = []

    pos_x = []
    pos_y = []

    acc_x = []
    acc_y = []

    delta_v = []
    a = []
    v_minus_delta_v = []

    angle_a = []
    angle_b = []
    angle_c = []

    gyro_z = []

    low_dynamic_judge = []

    mag_ang = []

    mag_ang_delta = []
    mag_ang_var = []

    raw_mag_x = []
    raw_mag_y = []
    raw_mag_z = []
    cal_mag_x = []
    cal_mag_y = []
    cal_mag_z = []

    mag_correct_p = []

    velocity = []

    yaw_angle_raw = []
    yaw_angle_my = []

    def add_avg_acc(self, acc):
        self.avg_acc_list.append(acc)

    def add_sum_delta(self, delta):
        self.sum_delta.append(delta)

    def add_sum_gryo(self, g):
        self.sum_gryo.append(g)

    def add_res_to_judge(self, res):
        self.res_to_judge.append(res)

    def add_avg_gyo(self, gyo):
        self.avg_gyo_list.append(gyo)

    def add_mag_gyo(self, gyo):
        self.mag_gyo_list.append(gyo)

    def add_mag_res(self, res):
        self.mag_res_to_judge.append(res)

    flag = True
    init_x = 0
    def add_pos(self, x, y):
        if self.flag:
            self.init_x = x
            self.flag = False
        self.pos_x.append(self.init_x - (x - self.init_x))
        self.pos_y.append(y)

    def add_acc(self, x, y):
        self.acc_x.append(x)
        self.acc_y.append(y)

    def add_debug_v(self, delta_v, v, a):
        self.delta_v.append(delta_v)
        self.v_minus_delta_v.append(v)
        self.a.append(a)

    def add_angle(self, a):
        self.angle_a.append(a[0])
        self.angle_b.append(a[1])
        self.angle_c.append(a[2])

    def add_gyro(self, g):
        self.gyro_z.append(g[2, 0])

    def add_low_dynamic_judge(self, m):
        self.low_dynamic_judge.append(m)

    def add_mag_ang(self, angle):
        self.mag_ang.append(angle)

    def add_angle_delta(self, delta):
        self.mag_ang_delta.append(delta)

    def add_angle_var(self, var):
        self.mag_ang_var.append(var)

    def add_raw_mag(self, mag):
        # mag = np.array(mag)
        # mag = mag / np.linalg.norm(mag)
        self.raw_mag_x.append(mag[0])
        self.raw_mag_y.append(mag[1])
        self.raw_mag_z.append(mag[2])

    def add_cal_mag(self, mag):
        self.cal_mag_x.append(mag[0, 0])
        self.cal_mag_y.append(mag[1, 0])
        self.cal_mag_z.append(mag[2, 0])

    def add_mag_correct_p(self, p):
        self.mag_correct_p.append(p)

    def add_velocity(self, v):
        self.velocity.append(v)

    def add_yaw_angle(self, y1, y2):
        self.yaw_angle_raw.append(y1)
        self.yaw_angle_my.append(y2)

    def show(self):
        # plt.figure()
        # t = np.arange(0, len(self.res_to_judge) * 0.01, 0.01)
        # plt.plot(t, self.sum_delta, 'r-', t, self.sum_gryo, 'g:', t, self.res_to_judge, 'b-.')
        # plt.figure(2)
        # mag_t = np.arange(0, len(self.mag_res_to_judge) * 0.01, 0.01)
        # plt.plot(mag_t, self.avg_gyo_list, 'r-', mag_t, self.mag_gyo_list, 'g:', mag_t, self.mag_res_to_judge, 'b-.')
        # plt.plot(mag_t, self.avg_gyo_list, 'r--', mag_t, self.mag_gyo_list, 'g--')
        # plt.show()

        # for i in range(len(self.res_to_judge)):
        #     print(str(self.sum_delta[i]) + " " + str(self.sum_gryo[i]) + " " + str(self.res_to_judge[i]) + " " + str(self.res_to_judge[i] < 0.3))


        # plt.figure('merge')
        # lent = len(self.mag_ang_var) - 10
        # t = np.arange(0, lent * 0.01, 0.01)
        # plt.plot(t, self.raw_mag_x[200:200+lent], 'r:', label='raw-mag-x')
        # plt.plot(t, self.cal_mag_x[200:200+lent], 'g:', label='calculate-mag-x')
        # plt.plot(t, self.mag_ang_var[0:lent], "b:", label='variance')
        # plt.legend()
        # plt.show()

        # map

        fig = plt.figure(figsize=(8, 8))
        plt.axis([0, 1008, 1009, 0])
        bgimg = img.imread('data/f7.png')
        imgplot = plt.imshow(bgimg)

        plt.plot(self.pos_x, self.pos_y, 'r-')
        plt.show()

        # gyro of axis-z
        # plt.figure('gyro-z')
        # t = np.arange(0, len(self.gyro_z) * 0.01, 0.01)
        # plt.plot(t, self.gyro_z, "r:")
        # plt.show()

        # raw mag
        plt.figure('mag')
        t = np.arange(0, len(self.raw_mag_x) *0.01, 0.01)
        plt.plot(t, self.raw_mag_x, 'c:', label='raw-x')
        plt.plot(t, self.raw_mag_y, 'm:', label='raw-y')
        plt.plot(t, self.raw_mag_z, 'y:', label='raw-z')

        plt.plot(t, self.cal_mag_x, lw=0.5, color='r', ls='-', label='cal-x')
        plt.plot(t, self.cal_mag_y, lw=0.5, color='g', ls='-', label='cal-y')
        plt.plot(t, self.cal_mag_z, lw=0.5, color='b', ls='-', label='cal-z')
        plt.ylabel('Magnetic Field/uT')
        plt.xlabel('Time/s')
        plt.legend()

        # raw_mag_angle = []
        # first_mag = np.array([self.raw_mag_x[0], self.raw_mag_y[0], self.raw_mag_z[0]])
        # for i in range(1, len(self.raw_mag_x)):
        #     raw_mag_angle.append(calculate_angle(first_mag, np.array([self.raw_mag_x[i], self.raw_mag_y[i], self.raw_mag_z[i]])))
        # plt.figure('raw-mag-angle')
        # t = np.arange(0, len(raw_mag_angle) * 0.01, 0.01)
        # plt.plot(t, raw_mag_angle, 'r:')

        # delta and var of mag-ang
        # plt.figure('delta')
        # t = np.arange(0, len(self.mag_ang_delta) * 0.01, 0.01)
        # # plt.plot(t, [180 for t in range(len(t))], "k--")
        # plt.plot(t, self.mag_ang_delta, "r:", label='max - min')
        # plt.show()
        plt.figure('var')
        t = np.arange(7, 7 + len(self.mag_ang_var)/10, 0.1)
        # plt.plot(t, [180 for t in range(len(t))], "k--")
        # print(np.mean(self.mag_ang_var[255:355]))
        plt.plot(t, self.mag_ang_var, "r:", label='variance')
        self.mag_correct_p = np.array(self.mag_correct_p) * (np.max(self.mag_ang_var) * 1.1)
        for i in range(len(self.mag_ang_var)):
            plt.vlines(7 + i/10, 0, self.mag_correct_p[i], lw=0.5, color='g', linestyles='--')
        # plt.plot(t, self.mag_correct_p, lw=0.5, color='g', ls='--')
        plt.xlabel('Time/s')
        plt.legend()
        # plt.show()

        # velocity of three axises
        # plt.figure('velocity')
        # t = np.arange(0, len(self.velocity) * 0.01, 0.01)
        # self.velocity = np.array(self.velocity)
        # plt.plot(t, self.velocity[:,0], 'r:')
        # plt.plot(t, self.velocity[:,1], 'g:')
        # plt.plot(t, self.velocity[:,2], 'b:')

        # angle of three axises
        plt.figure('angle')
        t = np.arange(0, len(self.angle_a) * 0.01, 0.01)

        self.mag_ang = np.array(self.mag_ang)
        # for i in range(len(self.mag_ang) -1):
        #     for j in range(3):
        #         self.mag_ang[i + 1, j] = 0.7 * self.mag_ang[i, j] + 0.3 * self.mag_ang[i+1, j]
        self.mag_ang = np.array([[x if x > -120 else x + 360 for x in a] for a in self.mag_ang])
        plt.plot(t, self.mag_ang[:, 0], "c:", label='raw-pitch')
        plt.plot(t, self.mag_ang[:, 1], "m:", label='raw-roll')
        plt.plot(t, self.mag_ang[:, 2], "y:", label='raw-yaw')

        self.angle_a = [a if a > -120 else a + 360 for a in self.angle_a]
        self.angle_b = [a if a > -120 else a + 360 for a in self.angle_b]
        self.angle_c = [a if a > -120 else a + 360 for a in self.angle_c]
        # plt.plot(t, [0 for t in range(len(t))], "k--")
        # plt.plot(t, [90 for t in range(len(t))], "k--")
        # plt.plot(t, [-90 for t in range(len(t))], "k--")
        # plt.plot(t, [180 for t in range(len(t))], "k--")
        plt.plot(t, self.angle_a, "r:", label='cal-pitch')
        plt.plot(t, self.angle_b, "g:", label='cal-roll')
        plt.plot(t, self.angle_c, "b:", label='cal-yaw')

        plt.xlabel('Time/s')
        plt.ylabel('Degree/°')
        plt.legend()

        plt.show()

        np.savetxt('data/Yaw_Data_raw.txt', self.yaw_angle_raw)
        np.savetxt('data/Yaw_Data_my.txt', self.yaw_angle_my)

        # judge low dynamic situation
        # plt.figure('low')
        # t = np.arange(0, len(self.low_dynamic_judge) * 0.01, 0.01)
        # plt.plot(t, self.low_dynamic_judge, "r:")
        # plt.show()


        # pure INS
        # plt.figure()
        # t = np.arange(0, len(self.acc_x) * 0.01, 0.01)
        # plt.plot(t, self.acc_x, 'r:', t, self.acc_y, 'b:')
        # plt.figure()
        # v_x, v_y = [0], [0]
        # for a in self.acc_x:
        #     v_x.append(v_x[len(v_x) - 1] + a * 0.01)
        # for a in self.acc_y:
        #     v_y.append(v_y[len(v_y) - 1] + a * 0.01)
        # plt.plot(t, v_x[1:], "r:", t, v_y[1:], "b:")
        # p_x, p_y = [0], [0]
        # plt.figure()
        # for v in v_x[1:]:
        #     p_x.append(p_x[len(p_x) - 1] + v * 0.01)
        # for v in v_y[1:]:
        #     p_y.append(p_y[len(p_y) - 1] + v * 0.01)
        # plt.plot(t, p_x[1:], "r:", t, p_y[1:], "b:")
        # plt.show()


        # debug delta_v
        # plt.figure()
        # t = np.arange(0, len(self.delta_v)* 0.01, 0.01)
        # plt.plot(t, [item[3, 0] for item in self.delta_v], 'r:', t, [item[4, 0] for item in self.delta_v], 'g:'
        #          , t, [item[5, 0] for item in self.delta_v], 'b:')
        #
        # plt.figure()
        # t = np.arange(0, len(self.a) * 0.01, 0.01)
        # plt.plot(t, [item[0, 0] for item in self.a], 'r:', t, [item[1, 0] for item in self.a], 'g:'
        #          , t, [item[2, 0] for item in self.a], 'b:')
        #
        # plt.figure()
        # plt.plot(t, [item[0, 0] for item in self.v_minus_delta_v], 'r:', t, [item[1, 0] for item in self.v_minus_delta_v], 'g:',
        #          t, [item[2, 0] for item in self.v_minus_delta_v], 'b:')
        # plt.show()


