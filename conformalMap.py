def conformalMapping(zipped_x_y):
    conformal_x = []
    conformal_y = []
    for i, obj in enumerate(zipped_x_y):
        x = obj[0]
        y = obj[1]
        denom = x*x + y*y
        conformal_x.append(x / denom)
        conformal_y.append(y / denom)
    return conformal_x, conformal_y

def circleFunction(x0, y0, r0, t):
    from math import cos, sin
    
    x = r0 * cos(t)
    y = r0 * sin(t)
    return x + x0, y + y0

def inverseConformalMapping(lines):
    import numpy as np
    from math import cos, sin, sqrt
    
    circles = np.empty((0, 3))
    for line in lines:
        rho = line[0]
        theta = line[1]
        # Unsure about why this is happening...
        if rho == 0:
            continue
        curve = (sin(theta) / (2 * rho)) * np.array([cos(theta) / sin(theta), 1, sqrt(pow(-cos(theta) / sin(theta), 2) + 1)])
        circles = np.vstack([circles, curve])
        
    return circles