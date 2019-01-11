from status import Status
from data_access import DataAccess
from judgment import Judgment
import numpy as np
from threshold_exp import ThresholdEXP
from utils import *

print('hello c-ins!')

# sample interval
delta_t = 1 / 100

last_step_time = 0
pos = None
first_epoch_mag = None
first_epoch_rotation = None


def process(status, frame, judgment):
    global pos
    global first_epoch_mag
    global first_epoch_rotation

    # predict
    status.next_delta(delta_t, frame)

    # correct
    judgment.judge(frame)
    if judgment.quasi_static_state():
        status.correct_by_zupt()
        if first_epoch_rotation is None:
            first_epoch_rotation = status.get_rotation_matrix()
        else:
            status.correct_by_zaru(first_epoch_rotation)
        pos = status.get_pos()
    else:
        first_epoch_rotation = None
        step_length, step_speed = judgment.new_step()
        if step_length > 0:
            if judgment.in_a_swing():
                pos = status.correct_by_step_length(step_length, pos)
            else:
                # can we also update pos here ?
                status.correct_by_velocity(step_speed)

    if judgment.low_dynamic():
        status.correct_by_gravity(frame)

    # if judgment.quasi_static_magnetic(status.get_rotation_matrix(), first_epoch_mag):
    #     if first_epoch_mag is None:
    #         first_epoch_mag = status.get_rotation_matrix() * array2matrix(frame.get_mags())
    #     else:
    #         status.correct_by_mag(frame, first_epoch_mag)
    # else:
    #     first_epoch_mag = None
    
    # feedback
    status.next(delta_t, frame)


if __name__ == '__main__':
    threshold_exp = ThresholdEXP()
    judgment = Judgment(delta_t, threshold_exp)
    data_access = DataAccess()

    # initial status
    x, y = data_access.get_start_pos()
    position = [x, y, 1]
    velocity = [0, 0, 0]
    rotation_matrix = np.eye(3)
    delta_p = [0, 0, 0]
    delta_v = [0, 0, 0]
    delta_ap = [0, 0, 0]
    bg = [0, 0, 0]
    ba = [0, 0, 0]

    status = Status(position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba)

    flag = 0
    while True:
        frame = data_access.get_frame()
        if frame is None:
            break
        elif np.linalg.norm(np.array(frame.get_accs())) > 9.5:
            flag = 1
        if flag == 1:
            process(status, frame, judgment)

    threshold_exp.show()
