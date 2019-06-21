from frame import Frame
import pandas as pd
import csv

class DataAccess:
    def __init__(self):
        self.df = pd.read_csv('data/expdata-twobuilding2.txt')
        self.loc = pd.read_csv('data/Location_Result1547455882986.txt')
        self.flag = True
        self.pos = 0
        self.sum = self.df.shape[0]

    def get_frame(self):
        if self.pos >= self.sum:
            return None
        if self.flag:
            while (self.df.iloc[self.pos, 1] == 0) | (self.df.iloc[self.pos, 2] == 0) | (self.df.iloc[self.pos, 3] == 0) |\
                    (self.df.iloc[self.pos + 1, 1] == 0) | (self.df.iloc[self.pos + 1, 2] == 0) | (self.df.iloc[self.pos + 1, 3] == 0) | \
                    (self.df.iloc[self.pos + 2, 1] == 0) | (self.df.iloc[self.pos + 2, 2] == 0) | (self.df.iloc[self.pos + 2, 3] == 0) | \
                    (self.df.iloc[self.pos + 3, 1] == 0):
                self.pos = self.pos + 4
            self.flag = False
        frame = Frame([self.df.iloc[self.pos, 1], self.df.iloc[self.pos, 2], self.df.iloc[self.pos, 3]],
                      [self.df.iloc[self.pos + 1, 1], self.df.iloc[self.pos + 1, 2], self.df.iloc[self.pos + 1, 3]],
                      [self.df.iloc[self.pos + 2, 1], self.df.iloc[self.pos + 2, 2], self.df.iloc[self.pos + 2, 3]],
                      [self.df.iloc[self.pos + 3, 1], self.df.iloc[self.pos + 3, 2], self.df.iloc[self.pos + 3, 3]], self.df.iloc[self.pos, 4])
        self.pos = self.pos + 4
        return frame

    def get_start_pos(self):
        return self.loc.iloc[0, 1], self.loc.iloc[0, 2]

    def get_end_pos(self):
        return self.loc.iloc[1, 1], self.loc.iloc[1, 2]

if __name__ == '__main__':
    da = DataAccess()
    head_info = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'yaw', 'time']
    csvFile = open('expdata-twobuildings.csv', 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head_info)
    while True:
        f = da.get_frame()
        if f is None:
            break
        acc = f.get_accs()
        gyr = f.get_gyros()
        mag = f.get_mags()
        y = f.get_angle()[2]
        new_row = [acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2], mag[0], mag[1], mag[2], y, f.get_time()]
        writer.writerow(new_row)
    csvFile.close()

