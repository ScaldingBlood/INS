#encoding=utf-8
import math

class StepDetector:
    # 初始化数据
    isTrue = False

    value_A = 0.0
    value_B = 0.0
    low = [0, 0, 0]
    s = 0
    sums = 0
    x = 0
    y = 0
    z = 0
    w = 0
    b = 0
    c = 0
    d = 0
    newsum = 0
    time_s2 = 0
    biaoji = 0
    newsum2 = 0
    o = 0
    num = 0
    flag = 0

    def step_detect(self, accs):
        last_num = self.newsum
        # low-pass Filter
        FILTERING_VALUE = 0.84
        accs = [(self.low[i] * FILTERING_VALUE + accs[i] * (1 - FILTERING_VALUE)) if self.low[i] != 0 else accs[i] for i in range(3)]

        self.low = accs
        self.flag = 0
        self.value_A = math.sqrt(sum(pow(i, 2) for i in accs))
        # print(A)
        self.value_B = self.value_A - 9.8
        # print(self.value_B)
        self.num = self.num + 1
        if self.value_B > 6.0 or self.value_B < -5.0:
            self.s = 0
            self.num = 0
        else:
            if self.s == 0 and self.flag == 0:
                self.flag = 1
                self.biaoji = 0
                self.num = 0
                if self.value_B < 0.5:
                    self.s = 0
                else:
                    self.s = 1
            else:
                if self.s == 1 and self.flag == 0:
                    self.time_s2 = 0
                    self.flag = 1
                    if self.value_B < 0.9 and self.value_B >= 0.5:
                        self.s= 1
                    if self.value_B >= 0.9:
                        self.s= 2
                    if self.value_B < 0.5:
                        self.s= 4
                if self.s == 4 and self.flag == 0:
                    self.flag = 1
                    if self.biaoji >= 10:
                        self.s = 0
                    else:
                        if self.value_B >= 0.5:
                            self.s = 1
                        else:
                            self.biaoji = self.biaoji + 1

            if self.s == 2 and self.flag == 0:
                self.flag = 1
                self.time_s2 = self.time_s2 + 1
                if self.time_s2 > 100:
                    self.s = 0
                else:
                    if self.value_B >= 0.9:
                        self.s = 2
                    if self.value_B < -0.5:
                        self.s = 3

            if self.s == 3 and self.flag == 0:
                self.flag = 1
                self.s = 6

            if self.s == 6 and self.flag == 0:
                self.flag = 1
                self.s = 0
                self.sums = self.sums + 1

                if self.sums < 4:
                    self.newsum = self.newsum + 1
                    self.isTrue = False
                    if self.b == 0:
                        self.b = self.b + 1
                        self.y = self.x
                    else:
                        if self.c == 0:
                            self.c = self.c + 1
                            self.z = self.y
                        if self.d == 0:
                            self.d = self.d + 1
                            self.w = self.z
                        # else:
                        #     pass
                else:
                    if (self.w - self.x) < 300 & (self.w - self.x) > 20:
                        self.newsum = self.newsum + 1
                        self.isTrue = True
                        self.b = 0
                        self.x = self.w
                    else:
                        if (self.x - self.y) < 300 & (self.x - self.y) > 20:
                            self.newsum = self.newsum + 1
                            self.isTrue = True
                            self.c = 0
                            self.b = 1
                            self.y = self.x
                        if (self.y - self.z) < 300 & (self.y - self.z) > 20:
                            self.newsum = self.newsum + 1
                            self.isTrue = True
                            self.d = 0
                            self.c = 1
                            self.z = self.y
                        if (self.z - self.w) < 300 & (self.z - self.w) > 20:
                            self.newsum = self.newsum + 1
                            self.isTrue = True
                            self.w = self.z
                            self.d = 1
                        else:
                            self.newsum = self.newsum + 1
                            self.isTrue = False
                            self.sums = 1
                            self.b = 1
                            self.c = 0
                            self.d = 0
                            self.x = 40
                            self.y = 40
                            self.z = 0
                            self.w = 0
                if self.isTrue is not True:
                    self.o = 0
                else:
                    if self.o == 0:
                        self.newsum2 = self.newsum2 + 3
                    self.newsum2 = self.newsum2 + 1
                    self.o = 1
        return self.newsum > last_num
