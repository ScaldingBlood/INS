import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img


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

    def add_pos(self, x, y):
        self.pos_x.append(x)
        self.pos_y.append(y)

    def add_acc(self, x, y):
        self.acc_x.append(x)
        self.acc_y.append(y)

    def add_debug_v(self, delta_v, v, a):
        self.delta_v.append(delta_v)
        self.v_minus_delta_v.append(v)
        self.a.append(a)

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


        # map

        fig = plt.figure(figsize=(8, 8))
        plt.axis([0, 1008, 1009, 0])
        bgimg = img.imread('data/f7.png')
        imgplot = plt.imshow(bgimg)

        plt.plot(self.pos_x, self.pos_y, 'r:')

        plt.show()


        # pure INS
        plt.figure()
        t = np.arange(0, len(self.acc_x) * 0.01, 0.01)
        plt.plot(t, self.acc_x, 'r:', t, self.acc_y, 'b:')
        plt.figure()
        v_x, v_y = [0], [0]
        for a in self.acc_x:
            v_x.append(v_x[len(v_x) - 1] + a * 0.01)
        for a in self.acc_y:
            v_y.append(v_y[len(v_y) - 1] + a * 0.01)
        plt.plot(t, v_x[1:], "r:", t, v_y[1:], "b:")
        p_x, p_y = [0], [0]
        plt.figure()
        for v in v_x[1:]:
            p_x.append(p_x[len(p_x) - 1] + v * 0.01)
        for v in v_y[1:]:
            p_y.append(p_y[len(p_y) - 1] + v * 0.01)
        plt.plot(t, p_x[1:], "r:", t, p_y[1:], "b:")
        plt.show()


        # debug delta_v
        plt.figure()
        t = np.arange(0, len(self.delta_v)* 0.01, 0.01)
        plt.plot(t, [item[3, 0] for item in self.delta_v], 'r:', t, [item[4, 0] for item in self.delta_v], 'g:'
                 , t, [item[5, 0] for item in self.delta_v], 'b:')

        plt.figure()
        t = np.arange(0, len(self.a) * 0.01, 0.01)
        plt.plot(t, [item[0, 0] for item in self.a], 'r:', t, [item[1, 0] for item in self.a], 'g:'
                 , t, [item[2, 0] for item in self.a], 'b:')

        plt.figure()
        plt.plot(t, [item[0, 0] for item in self.v_minus_delta_v], 'r:', t, [item[1, 0] for item in self.v_minus_delta_v], 'g:',
                 t, [item[2, 0] for item in self.v_minus_delta_v], 'b:')
        plt.show()


