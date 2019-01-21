from frame import Frame
import pandas as pd


class DataAccess:
    def __init__(self):
        self.df = pd.read_csv('data/AHRS_Data1547624422305.txt')
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
                      self.df.iloc[self.pos + 3, 1])
        self.pos = self.pos + 4
        return frame

    def get_start_pos(self):
        return self.loc.iloc[0, 1], self.loc.iloc[0, 2]

    def get_end_pos(self):
        return self.loc.iloc[1, 1], self.loc.iloc[1, 2]
