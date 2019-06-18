import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    yaw_angle_raw = np.loadtxt('data/Yaw_Data_raw.txt')[59:]
    yaw_angle_raw = np.array([x if x > 0 else 360 + x for x in yaw_angle_raw])
    yaw_angle_my = np.loadtxt('data/Yaw_Data_my.txt')[59:]
    yaw_angle_my = np.array([x if x > 0 else 360 + x for x in yaw_angle_my])
    yaw_angle_pdr = np.abs(pd.read_csv('data/Yaw_Data1560485859602.txt').values.T[0, :])[:]
    yaw_angle_pdr = 360 - yaw_angle_pdr - 38

    t = np.arange(0, len(yaw_angle_raw)/100, 0.01)
    t_p = np.arange(0, len(yaw_angle_raw)/100, len(yaw_angle_raw) / len(yaw_angle_pdr)/100)
    fig = plt.figure()
    plt.plot(t, yaw_angle_raw, 'y', label='Raw')
    plt.plot(t_p, yaw_angle_pdr, 'g', label='PCA')
    plt.plot(t, yaw_angle_my, 'b', label='Proposed')
    plt.xlabel('Time/s')
    plt.ylabel('Error/°')
    plt.legend()
    plt.show()

    
    print(np.mean(yaw_angle_raw))
    print(np.mean(yaw_angle_pdr))
    print(np.mean(yaw_angle_my))
    
    yaw_angle_raw = (yaw_angle_raw - 168)
    yaw_angle_raw = np.array([x if x < 180 else 360 - x for x in yaw_angle_raw])
    yaw_angle_raw = np.array([x if x > -180 else x + 360 for x in yaw_angle_raw])

    yaw_angle_my = (yaw_angle_my - 170)
    yaw_angle_my = np.array([x if x < 180 else 360 - x for x in yaw_angle_my])
    yaw_angle_my = np.array([x if x > -180 else x + 360 for x in yaw_angle_my])

    yaw_angle_pdr = (yaw_angle_pdr - 150)
    
    print("\nmean")
    print(np.mean(yaw_angle_raw))
    print(np.mean(yaw_angle_pdr))
    print(np.mean(yaw_angle_my))
    print("median")
    print(np.median(yaw_angle_raw))
    print(np.median(yaw_angle_pdr))
    print(np.median(yaw_angle_my))
    print('dev')
    print(math.sqrt(np.var(yaw_angle_raw)))
    print(math.sqrt(np.var(yaw_angle_pdr)))
    print(math.sqrt(np.var(yaw_angle_my)))
    print('max')
    print(np.max(yaw_angle_raw))
    print(np.max(yaw_angle_pdr))
    print(np.max(yaw_angle_my))

    yaw_angle_raw = np.sort(yaw_angle_raw)
    yaw_angle_pdr = np.sort(yaw_angle_pdr)
    yaw_angle_my = np.sort(yaw_angle_my)
    p1 = 1. * np.arange(len(yaw_angle_raw)) / (len(yaw_angle_raw) -1)
    p2 = 1. * np.arange(len(yaw_angle_pdr)) / (len(yaw_angle_pdr) -1)
    p3 = 1. * np.arange(len(yaw_angle_my)) / (len(yaw_angle_my) -1)

    # fig = plt.figure()
    # plt.grid(ls='--')
    # plt.plot(yaw_angle_raw, p1, "r", label='Raw')
    # yaw_angle_pdr = np.insert(yaw_angle_pdr, 0, -45)
    # yaw_angle_pdr = np.append(yaw_angle_pdr, 55)
    # p2 = np.insert(p2, 0, 0)
    # p2 = np.append(p2, 1)
    # plt.plot(yaw_angle_pdr, p2, "g", label='PCA')
    # yaw_angle_my = np.insert(yaw_angle_my, 0, -40)
    # yaw_angle_my = np.append(yaw_angle_my, 50)
    # p3 = np.insert(p3, 0, 0)
    # p3 = np.append(p3, 1)
    # plt.plot(yaw_angle_my, p3, "b", label='Proposed')
    # plt.legend()
    # plt.xlabel('Error(°)')
    # plt.ylabel('$p$(Error(°))')
    # plt.show()

