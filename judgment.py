from collections import deque
from step_detector import StepDetector
import numpy as np
import math


class Judgment:
    # quasi static judgment window length
    N = 30
    
    # accelerator noise
    acc_noise = 0.01
    # gyroscope noise
    gyro_noise = 0.01
    # quasi static judgment parameter
    gama = 10
    
    # variance of acceleration
    va = 3

    # window size of mag
    Win_size = 10
    # threshold to judge quasi static mag
    T = 1

    is_first_step = True

    def __init__(self, delta_t, threshold_exp):
        self.N_frames = deque()
        self.Step_acc_frames = deque()
        self.Win_mag_frames = deque()
        self.Win_gry_frames = deque()
        self.Win_angle_frames = deque()
        self.step_detector = StepDetector()
        self.delta_t = delta_t
        self.threshold_exp = threshold_exp

    def judge(self, frame):
        self.frame = frame

        self.Step_acc_frames.append(frame.get_accs())

        if len(self.Win_mag_frames) == self.Win_size:
            self.Win_mag_frames.popleft()
            self.Win_gry_frames.popleft()
            self.Win_angle_frames.popleft()
        self.Win_mag_frames.append(frame.get_mags())
        self.Win_gry_frames.append(frame.get_gyros())
        self.Win_angle_frames.append(frame.get_angle())

        if len(self.N_frames) == self.N:
            self.N_frames.popleft()
        self.N_frames.append(frame)

    def quasi_static_state(self):
        if len(self.N_frames) < self.N:
            return False
        else:
            avg_acc = [sum([frame.get_accs()[j] for frame in self.N_frames]) / self.N for j in range(3)]

            sum_acc_delta = 0
            for frame in self.N_frames:
                sum_acc_delta += sum(math.pow(frame.get_accs()[i] - avg_acc[i], 2) for i in range(3))
            self.threshold_exp.add_sum_delta(sum_acc_delta)

            sum_gyro = 0
            for frame in self.N_frames:
                sum_gyro += sum(math.pow(frame.get_gyros()[i], 2) for i in range(3))
            self.threshold_exp.add_sum_gryo(sum_gyro)

            self.threshold_exp.add_res_to_judge(sum_acc_delta / math.pow(self.acc_noise, 2) + sum_gyro / math.pow(self.gyro_noise, 2))
            if (sum_acc_delta / math.pow(self.acc_noise, 2) + sum_gyro / math.pow(self.gyro_noise, 2)) / self.N < self.gama:
                return True
            else:
                return False

    def new_step(self):
        accs = self.frame.get_accs()
        self.Step_acc_frames.append(accs)

        step_length = 0
        step_speed = 0
        # peak detection algorithm
        if self.step_detector.step_detect(accs):
            # if self.is_first_step:
            #     self.is_first_step = False
            # else:
            step_length = self.calculate_step_length()
            is_swing = self.in_a_swing()
            if is_swing:
                step_speed = step_length / (len(self.Step_acc_frames) * self.delta_t)
            self.Step_acc_frames.clear()
        return step_length, step_speed

    def calculate_step_length(self):
        max_value = self.Step_acc_frames[0]
        min_value = self.Step_acc_frames[0]
        for v in self.Step_acc_frames:
            min_value = v if v < min_value else min_value
            max_value = v if v > max_value else max_value
        return math.pow(max_value-min_value, 1/4)

    def in_a_swing(self):
        mos = np.array([math.sqrt(sum(pow(x[i], 2) for i in range(3))) for x in self.Step_acc_frames])
        return np.var(mos) > self.va

    def low_dynamic(self):
        # accs = self.frame.get_accs()
        # mo = math.sqrt(sum(pow(accs[i], 2) for i in range(3)))
        mo = np.linalg.norm(np.array(self.frame.get_accs()))
        return math.fabs(mo - 9.801) < 0.5

    def quasi_static_magnetic(self, rotation_matrix):
        if len(self.Win_gry_frames) < self.Win_size:
            return False

        gyros = []
        for g in self.Win_gry_frames:
            gyros.append((np.matrix(g) * rotation_matrix)[0, 2])
        avg = np.mean(np.array(gyros))

        self.threshold_exp.add_avg_gyo(avg)

        tmp = list(self.Win_angle_frames)
        diff = (tmp[self.Win_size - 1] - tmp[0]) / ((self.Win_size - 1) * self.delta_t)

        self.threshold_exp.add_mag_gyo(diff)

        self.threshold_exp.add_mag_res(math.fabs(avg - diff))
        return self.T - math.fabs(avg - diff) > 0
