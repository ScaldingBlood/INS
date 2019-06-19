from status import Status
from status2 import Status2
from data_access import DataAccess
from judgment import Judgment
from judgment2 import Judgment2
import numpy as np
from experiment import Experiment
from utils import *

print('hello c-ins!')

# sample interval
delta_t = 1 / 100

last_step_time = 0
pos = None
last_rotation = None
first_epoch_mag = None
first_epoch_rotation = None
step_count = 0
new_step = False
mag_count = 0
begin = False

def cal_distance(pos1, pos2):
    if np.linalg.norm((pos1 - pos2)[0:1, 0]) < 1:
        return True
    else:
        return False
    return np.linalg.norm(pos1 - pos2) < 0.8

def process(status, frame, judgment):
    global pos
    global last_rotation
    global new_step
    global first_epoch_mag
    global first_epoch_rotation
    global step_count
    global mag_count
    global begin


    exp.add_acc(frame.get_accs()[0], frame.get_accs()[1])
    # predict
    status.next_delta(delta_t, frame)

    # correct
    judgment.judge(frame)
    if judgment.quasi_static_state():
        status.correct_by_zupt()
        status.correct_by_gravity(frame)
        # if first_epoch_rotation is None:
        #     first_epoch_rotation = status.get_rotation_matrix()
        # else:
        #     status.correct_by_zaru(first_epoch_rotation)
    else:
        if judgment.low_dynamic():
            status.correct_by_gravity(frame)

        first_epoch_rotation = None
        step_length, step_speed = judgment.new_step()
        if step_length > 0:
            new_step = True
            step_count = step_count + 1

            # if pos is not None and cal_distance(pos, status.get_pos()):
            # # if pos is not None:
            #     status.correct_by_step_length(step_length, pos, last_rotation)

            if step_speed > 0:
                status.correct_by_velocity(step_speed)
        else:
            new_step = False

    if len(judgment.Win_mag_frames) > judgment.Win_size:
        if judgment.quasi_static_magnetic():
            mag_count += 1
            if mag_count == 5:

                if not begin:
                    # status.q = vec_2q(frame.get_mags(), status.global_m)
                    eular_ang = frame.get_angle()
                    eular_ang[0] = 0
                    eular_ang[1] = 0
                    status.q = angle2q(eular_ang)
                    begin = True

                exp.add_mag_correct_p(1)
                status.correct_by_mag(array2matrix(frame.get_mags()))
                mag_count = 0
            else:
                exp.add_mag_correct_p(0)
        else:
            mag_count = 0
            exp.add_mag_correct_p(0)
    # else:
    #     exp.add_mag_correct_p(0)

    # feedback
    status.add_sensor_data(delta_t, frame)

    status.next(delta_t, frame, exp, begin)

    # set new step
    # if len(judgment.Step_acc_frames) == 0:
    if new_step:
        pos = status.get_pos()
        last_rotation = status.get_rotation_matrix()


def process2(status, frame, judgment):
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
    if step_length > 0:
        step_count += 1

    # feedback
    status.next(delta_t, frame, exp, step_length)

if __name__ == '__main__':
    exp = Experiment()
    judgment = Judgment(delta_t, exp)
    judgment2 = Judgment2(delta_t, exp)
    data_access = DataAccess()

    # initial status
    x, y = data_access.get_start_pos()
    position = [x, y, 1]
    velocity = [0, 1, 0]
    rotation_matrix = np.eye(3)
    delta_p = [0, 0, 0]
    delta_v = [0, 0, 0]
    delta_ap = [0, 0, 0]
    bg = [0.0003, 0.0003, 0.0003]
    ba = [0.02, 0.15, -0.17]

    status = Status(position, velocity, rotation_matrix, delta_p, delta_v, delta_ap, bg, ba)

    position2 = [x, y, 1]
    velocity2 = [0, 1, 0]
    rotation_matrix2 = np.eye(3)
    delta_p2 = [0, 0, 0]
    delta_v2 = [0, 0, 0]
    delta_ap2 = [0, 0, 0]
    bg2 = [0.0003, 0.0003, 0.0003]
    ba2 = [0.02, 0.15, -0.17]
    status2 = Status2(position2, velocity2, rotation_matrix2, delta_p2, delta_v2, delta_ap2, bg2, ba2)
    exp.add_pos(x * 12, y * 12)
    exp.add_pos2(x * 12, y * 12)

    flag = 0
    while True:
        frame = data_access.get_frame()
        if frame is None:
            break
        elif np.linalg.norm(np.array(frame.get_accs())) > 9:
            flag = 1
        if flag == 1:
            process(status, frame, judgment)
            process2(status2, frame, judgment2)

    print("Step Count: " + str(step_count))
    exp.show()