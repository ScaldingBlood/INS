from collections import deque
from step_detector import StepDetector
import numpy as np
import math
from utils import *


class Judgment:
    # quasi static judgment window length
    N = 30
    
    # accelerator noise
    acc_noise = 0.5
    # gyroscope noise
    gyro_noise = 0.5
    # quasi static judgment parameter
    gama = 2 / N
    
    # variance of acceleration
    va = 7

    # window size of mag
    Win_size = 700
    # threshold to judge quasi static mag
    T = 2

    is_first_step = True
    first_mag = np.mat([0, 0, 1]).T

    def __init__(self, delta_t, exp):
        self.N_frames = deque()
        self.Step_acc_frames = deque()
        self.rotations = deque()
        self.Win_mag_frames = deque()
        self.Win_gyr_frames = deque()
        self.step_detector = StepDetector()
        self.delta_t = delta_t
        self.exp = exp
        # self.angles = deque()

    def judge(self, frame):
        self.frame = frame

        self.Step_acc_frames.append(frame.get_accs())

        # if len(self.Win_mag_frames) == self.Win_size:
        #     self.Win_mag_frames.popleft()
        #     self.Win_gyr_frames.popleft()
        self.Win_mag_frames.append(frame.get_mags())
        self.Win_gyr_frames.append(frame.get_gyros())

        if len(self.N_frames) == self.N:
            self.N_frames.popleft()

        # if len(self.angles) == 200:
        #     self.angles.popleft()
        # self.angles.append((frame.get_angle() if frame.get_angle() > 0 else frame.get_angle() + math.pi * 2) * 180 / math.pi)
        self.N_frames.append(frame)

    def quasi_static_state(self):
        if len(self.N_frames) < self.N:
            return False
        else:
            avg_acc = [sum([frame.get_accs()[j] for frame in self.N_frames]) / self.N for j in range(3)]

            sum_acc_delta = 0
            for frame in self.N_frames:
                sum_acc_delta += sum(math.pow(frame.get_accs()[i] - avg_acc[i], 2) for i in range(3))
            self.exp.add_sum_delta(sum_acc_delta)

            sum_gyro = 0
            for frame in self.N_frames:
                sum_gyro += sum(math.pow(frame.get_gyros()[i], 2) for i in range(3))
            self.exp.add_sum_gryo(sum_gyro)

            self.exp.add_res_to_judge((sum_acc_delta / math.pow(self.acc_noise, 2) + sum_gyro / math.pow(self.gyro_noise, 2)) / self.N)
            if (sum_acc_delta / math.pow(self.acc_noise, 2) + sum_gyro / math.pow(self.gyro_noise, 2)) / self.N < self.gama:
                return True
            else:
                return False

    def new_step(self):
        accs = self.frame.get_accs()
        step_length = 0
        step_speed = 0
        # peak detection algorithm
        if self.step_detector.step_detect(accs):
            # if self.is_first_step:
            #     self.is_first_step = False
            # else:
            step_length = self.calculate_step_length()
            # step_length = 0.6
            is_swing = self.in_a_swing()
            if not is_swing:
                during = len(self.Step_acc_frames) * self.delta_t
                during = 1 if during > 1 else during
                step_speed = step_length / during
            step_speed = step_speed if step_speed < 2 else 1
            self.Step_acc_frames.clear()
        return step_length, step_speed

    def calculate_step_length(self):
        max_value = np.linalg.norm(np.array(self.Step_acc_frames[0]))
        min_value = np.linalg.norm(np.array(self.Step_acc_frames[0]))
        for item in self.Step_acc_frames:
            v = np.linalg.norm(item)
            min_value = v if v < min_value else min_value
            max_value = v if v > max_value else max_value
        return 0.2* math.pow(max_value-min_value, 1/3)/3 + 0.58

    def in_a_swing(self):
        mos = np.array([math.sqrt(sum(pow(x[i], 2) for i in range(3))) for x in self.Step_acc_frames])
        # print(np.var(mos))
        # return np.var(mos) > self.va
        return False

    def low_dynamic(self):
        # accs = self.frame.get_accs()
        # mo = math.sqrt(sum(pow(accs[i], 2) for i in range(3)))
        mo = np.linalg.norm(np.array(self.frame.get_accs()))
        # self.exp.add_low_dynamic_judge(math.fabs(mo - 9.801))
        return math.fabs(mo - 9.801) < 0.02

    def quasi_static_magnetic(self):
        # if len(self.Win_gyr_frames) < self.Win_size:
        #     return False

        mag0 = np.array(self.Win_mag_frames[0])
        # mag0 = mag0 / np.linalg.norm(mag0) if np.linalg.norm(mag0) > 0 else mag0
        q = np.mat([1, 0, 0, 0]).T
        delta_t = 1 / 100
        delta_angle = []
        for i in range(self.Win_size):
            new_mag = np.mat(self.Win_mag_frames[i + 1])
            new_mag = new_mag / np.linalg.norm(new_mag)

            tmp = self.Win_gyr_frames[i]
            th = np.linalg.norm(tmp) * delta_t
            tmp = tmp / np.linalg.norm(tmp) * sin(th / 2)
            tmpq = np.mat([cos(th / 2), -tmp[0], -tmp[1], -tmp[2]]).T
            q = multiple_q(q, tmpq)

            R = q2R(q)
            cal_mag = R * np.mat(mag0).T
            cal_mag = cal_mag / np.linalg.norm(cal_mag)

            delta_angle.append((math.acos(new_mag * cal_mag)) / math.pi * 180)
            # tmp = (new_mag * (last_mag.T if last_mag is not None else new_mag.T))[0, 0]
            # delta_angle.append(math.acos(tmp if tmp <= 1 else 1) / math.pi * 180)
            # last_mag = new_mag


        # delta_val = max(delta_angle)
        variance = np.var(delta_angle)
        # self.exp.add_angle_delta(delta_val)
        self.exp.add_angle_var(variance)

        for i in range(10):
            self.Win_mag_frames.popleft()
            self.Win_gyr_frames.popleft()

        if variance < 1.5:
            # avg = np.mean(self.Win_mag_frames, 0)
            return True
        else:
            return False


    # def quasi_static_magnetic(self, rotation_matrix, first_epoch_mag):
    #     self.rotations.append(rotation_matrix)
    #     if len(self.Win_gyr_frames) < self.Win_size:
    #         return False
    #
    #     gyros = []
    #     for i in range(len(self.Win_gyr_frames)):
    #         gyros.append((self.rotations[i] * np.matrix(self.Win_gyr_frames[i]).T)[2, 0])
    #     avg = np.mean(np.array(gyros))
    #
    #     self.exp.add_avg_gyo(avg)
    #
    #     tmp = list(self.Win_angle_frames)
    #     diff = (tmp[self.Win_size - 1] - tmp[0]) / ((self.Win_size - 1) * self.delta_t)
    #
    #     self.exp.add_mag_gyo(diff)
    #
    #     self.exp.add_mag_res(math.fabs(avg - diff))
    #     print('list' + str(tmp))
    #     print(str(diff))
    #     print(first_epoch_mag)
    #     print(rotation_matrix * array2matrix(self.Win_mag_frames[self.Win_size - 1]))
    #     print(self.T - math.fabs(avg - diff))
    #     return self.T - math.fabs(avg - diff) > 0
