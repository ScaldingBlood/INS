class Frame:
    def __init__(self, accs, mags, gyros, angle):
        self.accs = accs
        self.gyros = gyros
        self.mags = mags
        self.angle = angle
    
    def get_accs(self):
        return self.accs

    def get_gyros(self):
        return self.gyros
    
    def get_mags(self):
        return self.mags

    def get_angle(self):
        return self.angle
