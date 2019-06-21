import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    yaw_angle_raw = np.loadtxt('data/Yaw_Data_raw.txt')[2800:]
    yaw_angle_raw = np.array([x if x > 0 else 360 + x for x in yaw_angle_raw])
    yaw_angle_my = np.loadtxt('data/Yaw_Data_my.txt')[2800:]
    yaw_angle_my = np.array([x if x > 0 else 360 + x for x in yaw_angle_my])
    yaw_angle_ahrs = np.abs(pd.read_csv('data/Yaw_Data_AHRS.csv').values.T[0, :])[3000:] + 90
    yaw_angle_pdr = np.abs(pd.read_csv('data/Yaw_Data_PCA.txt').values.T[0, :])[:]
    yaw_angle_pdr = yaw_angle_pdr + 30

    t = np.arange(0, len(yaw_angle_raw)/100, 0.01)
    t_a = np.arange(0, len(yaw_angle_ahrs)/100, 0.01)
    t_p = np.arange(0, len(yaw_angle_raw)/100, len(yaw_angle_raw) / len(yaw_angle_pdr)/100)
    fig = plt.figure()
    plt.plot(t, yaw_angle_raw, 'y', label='Raw')
    plt.plot(t_p, yaw_angle_pdr, 'g', label='PCA')
    plt.plot(t_a, yaw_angle_ahrs, 'r', label='AHRS')
    plt.plot(t, yaw_angle_my, 'b', label='Proposed')
    plt.xlabel('Time/s')
    plt.ylabel('Error/°')
    plt.legend()
    plt.show()

    yaw_angle_raw = np.loadtxt('data/Yaw_Data_raw.txt')[2800:]
    yaw_angle_my = np.loadtxt('data/Yaw_Data_my.txt')[2800:]
    yaw_angle_pdr = np.abs(pd.read_csv('data/Yaw_Data_PCA.txt').values.T[0, :])[30:]
    yaw_angle_ahrs = np.abs(pd.read_csv('data/Yaw_Data_AHRS.csv').values.T[0, :])[:]

    print(np.mean(yaw_angle_raw))
    print(np.mean(yaw_angle_pdr))
    print(np.mean(yaw_angle_my))
    print(np.mean(yaw_angle_ahrs))

    yaw_angle_raw = (yaw_angle_raw - 172)
    yaw_angle_raw = np.array([x if x < 180 else 360 - x for x in yaw_angle_raw])
    yaw_angle_raw = np.array([x if x > -180 else x + 360 for x in yaw_angle_raw])

    yaw_angle_my = (yaw_angle_my - 170)
    yaw_angle_my = np.array([x if x < 180 else 360 - x for x in yaw_angle_my])
    yaw_angle_my = np.array([x if x > -180 else x + 360 for x in yaw_angle_my])

    yaw_angle_pdr = (yaw_angle_pdr - 150)

    yaw_angle_ahrs = yaw_angle_ahrs - 82

    print("\nmean")
    print(np.mean(abs(yaw_angle_raw)))
    print(np.mean(abs(yaw_angle_pdr)))
    print(np.mean(abs(yaw_angle_ahrs)))
    print(np.mean(abs(yaw_angle_my)))
    print("median")
    print(np.median(abs(yaw_angle_raw)))
    print(np.median(abs(yaw_angle_pdr)))
    print(np.median(abs(yaw_angle_ahrs)))
    print(np.median(abs(yaw_angle_my)))
    print('dev')
    print(math.sqrt(np.var(abs(yaw_angle_raw))))
    print(math.sqrt(np.var(abs(yaw_angle_pdr))))
    print(math.sqrt(np.var(abs(yaw_angle_ahrs))))
    print(math.sqrt(np.var(abs(yaw_angle_my))))
    print('max')
    print(np.max(abs(yaw_angle_raw)))
    print(np.max(abs(yaw_angle_pdr)))
    print(np.max(abs(yaw_angle_ahrs)))
    print(np.max(abs(yaw_angle_my)))

    yaw_angle_raw = np.sort(yaw_angle_raw)
    yaw_angle_pdr = np.sort(yaw_angle_pdr)
    yaw_angle_ahrs = np.sort(yaw_angle_ahrs)
    yaw_angle_my = np.sort(yaw_angle_my)
    p1 = 1. * np.arange(len(yaw_angle_raw)) / (len(yaw_angle_raw) - 1)
    p2 = 1. * np.arange(len(yaw_angle_pdr)) / (len(yaw_angle_pdr) - 1)
    p3 = 1. * np.arange(len(yaw_angle_ahrs)) / (len(yaw_angle_ahrs) - 1)
    p4 = 1. * np.arange(len(yaw_angle_my)) / (len(yaw_angle_my) - 1)

    fig = plt.figure()
    plt.grid(ls='--')

    yaw_angle_my = np.insert(yaw_angle_my, 0, -47)
    yaw_angle_my = np.append(yaw_angle_my, 52)
    p4 = np.insert(p4, 0, 0)
    p4 = np.append(p4, 1)
    plt.plot(yaw_angle_my, p4, "b", label='Proposed')

    yaw_angle_ahrs = np.insert(yaw_angle_ahrs, 0, -47)
    yaw_angle_ahrs = np.append(yaw_angle_ahrs, 52)
    p3 = np.insert(p3, 0, 0)
    p3 = np.append(p3, 1)
    plt.plot(yaw_angle_ahrs, p3, 'y', label='AHRS')

    yaw_angle_pdr = np.insert(yaw_angle_pdr, 0, -47)
    yaw_angle_pdr = np.append(yaw_angle_pdr, 52)
    p2 = np.insert(p2, 0, 0)
    p2 = np.append(p2, 1)
    plt.plot(yaw_angle_pdr, p2, "g", label='PCA')

    plt.plot(yaw_angle_raw, p1, "r", label='Raw')
    plt.legend()
    plt.xlabel('Error/°')
    plt.ylabel('$p$(Error/°)')
    plt.show()

