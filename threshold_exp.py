import matplotlib.pyplot as plt
import numpy as np


class ThresholdEXP:
    avg_acc_list = []
    sum_delta = []  # 1e-5
    sum_gryo = []  # 1e-7
    res_to_judge = []

    avg_gyo_list = []  # 0.00072
    mag_gyo_list = []  # -0.03
    mag_res_to_judge = []

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

    def show(self):
        plt.figure()
        t = np.arange(0, len(self.res_to_judge) * 0.01, 0.01)
        plt.plot(t, self.sum_delta, 'r-', t, self.sum_gryo, 'g:', t, self.res_to_judge, 'b-.')
        plt.figure(2)
        mag_t = np.arange(0, len(self.mag_res_to_judge) * 0.01, 0.01)
        plt.plot(mag_t, self.avg_gyo_list, 'r--', mag_t, self.mag_gyo_list, 'g--', mag_t, self.mag_res_to_judge, 'b--')
        # plt.plot(mag_t, self.avg_gyo_list, 'r--', mag_t, self.mag_gyo_list, 'g--')
        plt.show()
        # for i in range(len(self.res_to_judge)):
        #     print(str(self.sum_delta[i]) + " " + str(self.sum_gryo[i]) + " " + str(self.res_to_judge[i]) + " " + str(self.res_to_judge[i] < 0.3))
