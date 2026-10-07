import math

def check_angle_rad(angle1, angle2, tol=0.8):
    diff = math.atan2(math.sin(angle1 - angle2), math.cos(angle1 - angle2))
    return abs(diff) <= tol