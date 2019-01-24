from status import Status
from data_access import DataAccess
from judgment import Judgment
import numpy as np
from experiment import Experiment
from utils import *

print('hello c-ins!')

# sample interval
delta_t = 1 / 100

last_step_time = 0
pos = None
first_epoch_mag = None
first_epoch_rotation = None
step_count = 0


def process(status, frame, judgment):
    global pos
    global first_epoch_mag
    global first_epoch_rotation
    global step_count

    exp.add_acc(frame.get_accs()[0], frame.get_accs()[1])
    # predict
    status.next_delta(delta_t, frame)

    # correct
    judgment.judge(frame)
    if judgment.quasi_static_state():
        status.correct_by_zupt()
        # if first_epoch_rotation is None:
        #     first_epoch_rotation = status.get_rotation_matrix()
        # else:
        #     status.correct_by_zaru(first_epoch_rotation)
    else:
        first_epoch_rotation = None
        step_length, step_speed = judgment.new_step()
        if step_length > 0:
            if step_speed == 0:
                if pos == 0:
                    pos = status.get_pos()
                else:
                    pos = status.correct_by_step_length(step_length, pos)
            else:
                step_count = step_count + 1
                pos = 0
                # can we also update pos here ?
                status.correct_by_velocity(step_speed)
        else:
            pos = 0

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
    status.next(delta_t, frame, exp)


if __name__ == '__main__':
    exp = Experiment()
    judgment = Judgment(delta_t, exp)
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
    ba = [0, 0, 0.03]

    status = Status(position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba)

    flag = 0
    while True:
        frame = data_access.get_frame()
        if frame is None:
            break
        elif np.linalg.norm(np.array(frame.get_accs())) > 9:
            flag = 1
        if flag == 1:
            process(status, frame, judgment)

    print("Step Count: " + str(step_count))
    exp.show()
