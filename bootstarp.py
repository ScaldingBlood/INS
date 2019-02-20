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
    # status.next_delta(delta_t, frame)

    # correct
    judgment.judge(frame)
    step_length, step_speed = judgment.new_step()
    
    # feedback
    status.next(delta_t, frame, exp, step_speed)


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
    exp.add_pos(x * 12, y * 12)

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
