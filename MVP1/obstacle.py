class BoundedObstacle:
    def __init__(self):
        self.x = None
        self.y = None
        self.width = None
        self.height = None
        self.angle = None
        self.corners = None  # Ex: [[x, y], [x, y], [x, y], [x, y]]
        self.distance = -1

    def setRect(self, rect):
        self.x = round(rect[0][0])
        self.y = round(rect[0][1])
        self.width = round(rect[1][0])
        self.height = round(rect[1][1])
        self.angle = round(rect[2])
    
    def setCorners(self, box):
        self.corners = box

    def print(self):
        print('Coordinates (x,y): ({}, {})'.format(self.x, self.y))
        print('Width: {}'.format(self.width))
        print('Height: {}'.format(self.height))
        print('Angle: {}'.format(self.angle))
        print('Corners: {}'.format(self.corners))
        print()

class TrackedObject:
    def __init__(self, np_points):
        self.corners = None
        self.points = np_points
        self.mask = None
        self.trackCount = 1