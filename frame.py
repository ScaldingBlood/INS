class Frame:
    def __init__(self, accs, mags, gyros, angle, time):
        self.accs = accs
        self.gyros = gyros
        self.mags = mags
        self.angle = [-angle[0], -angle[1], -angle[2]]
        self.time = time
    
    def get_accs(self):
        return self.accs

    def get_gyros(self):
        return self.gyros
    
    def get_mags(self):
        return self.mags

    def get_angle(self):
        return self.angle

    def get_time(self):
        return self.time