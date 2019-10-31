def createHoughSpace(points, minRho, maxRho, minTheta, maxTheta, nrBinsRho, nrBinsTheta):
    """
    Performs Hough transformation (returns the Hough space).
    TODO: UPDATE TO USE NUMPY MESHGRID

    :param points: The points to perform the Hough transformation on.
    :param minRho: min value for rho (length to line from origin)
    :param maxRho: max value for rho (length to line from origin)
    :param minTheta: min value for theta (angle of line)
    :param maxTheta: max value for theta (angle of line)
    :param nrBinsRho: the number of bins to be used for rho in the accumulator (Hough space).
    :param nrBinsTheta: the number of bins to be used for theta in the accumulator (Hough space).
    :return: returns the accumulator array as a numpy matrix with shape (nrBinsRho, nrBinsTheta)
    """
    import numpy as np
    import math
    
    thetas = np.linspace(minTheta, maxTheta, nrBinsTheta)
    rhoRange = maxRho - minRho
    accumulator = np.zeros((nrBinsRho, nrBinsTheta))
    for p in points:
        x = p[0]
        y = p[1]
        for i, theta in enumerate(thetas):
            rho = x * math.cos(theta) + y * math.sin(theta)
            #add with 1/2 nrBins so that negative values are in the bottom half of bins
            rhoIndex = ((rho / rhoRange) * nrBinsRho) + (nrBinsRho / 2) 
            rhoIndex = int(rhoIndex)
            accumulator[rhoIndex, i] += 1
            
    return accumulator, maxRho, minRho, nrBinsRho, nrBinsTheta, thetas


def visualizeHoughSpace3D(accumulator, minRho, maxRho, minTheta, maxTheta, nrBinsRho, nrBinsTheta):
    """
    Visualizes the Hough space in 3D

    :param accumulator: The Hough space.
    :param minRho: min value for rho (length to line from origin)
    :param maxRho: max value for rho (length to line from origin)
    :param minTheta: min value for theta (angle of line)
    :param maxTheta: max value for theta (angle of line)
    :param nrBinsRho: the number of bins to be used for rho in the accumulator (Hough space).
    :param nrBinsTheta: the number of bins to be used for theta in the accumulator (Hough space).
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')
    x = np.linspace(minRho, maxRho, nrBinsRho)
    y = np.linspace(minTheta, maxTheta, nrBinsTheta)
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, np.transpose(accumulator))
    plt.show()

    
def findMaxValues(accumulator, nrBinsRho, nrBinsTheta, threshhold):
    """
    Finds all max values in the accumulator array.
    Algorithm:
    For every point:
        if surrounding values are lower or equal:
            chose as one max value
    
    :param accumulator: The Hough space.
    :param nrBinsRho: the number of bins to be used for rho in the accumulator (Hough space).
    :param nrBinsTheta: the number of bins to be used for theta in the accumulator (Hough space).
    :return: returns points of coordiantes where the max values are. (rho, theta)
    """
    points = []
    for r in range(nrBinsRho):
        for t in range(nrBinsTheta):
            if checkIfMaxInSurronding(accumulator, r, t, nrBinsRho, nrBinsTheta):
                if accumulator[r, t] >= threshhold:
                    points.append((r, t))
            
    return points

def checkIfMaxInSurronding(accumulator, r, t, nrBinsRho, nrBinsTheta):
    """
    Check if this index (r, t) is the max among its values (i.e check if surrounding is less)

    :param accumulator: The Hough space.
    :param r: the rho index
    :param t: the theta index
    :param nrBinsRho: the number of bins to be used for rho in the accumulator (Hough space).
    :param nrBinsTheta: the number of bins to be used for theta in the accumulator (Hough space).
    :return: returns true if the values is the largest in the window and false otherwise
    """
    area = 1
    midValue = accumulator[r, t]
    for rt in range(r - area, r + area + 1):
        for tt in range(t - area, t + area + 1):
            if rt == r and t == tt:
                continue
            if 0 <= rt < nrBinsRho and 0 <= tt < nrBinsTheta and midValue < accumulator[rt, tt]:
                return False
    return True


def extractRhoAndTheta(accumulatorIndices, thetas, minRho, maxRho, nrBinsRho):
    """
    Get all rho and theta values from a set of indices in the accumulator

    :param accumulator: The Hough space.
    :param nrBinsRho: the number of bins to be used for rho in the accumulator (Hough space).
    :param nrBinsTheta: the number of bins to be used for theta in the accumulator (Hough space).
    :return: returns points of coordiantes where the max values are. (rho, theta)
    """
    
    import numpy as np
    
    maxRhos = np.array([])
    maxThetas = np.array([])

    for index in accumulatorIndices:
        r = index[0]
        t = index[1]
        rhoRange = maxRho - minRho
        rho = ((r - (nrBinsRho / 2)) * rhoRange) / nrBinsRho
        maxRhos = np.append(maxRhos, rho)
        maxThetas = np.append(maxThetas, thetas[t])
    
    return maxRhos, maxThetas
    
    
    
    
    
    
    