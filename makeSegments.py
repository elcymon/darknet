import numpy as np

def makeSegments(xPoints,yPoints):
    '''
    Takes in list of x coordinates and y coordinates and returns top-left and bottom-right xy coordinates from the lists
    '''
    topLeft_bottomRight = []
    for y in range(len(yPoints) - 1):
        for x in range(len(xPoints) - 1):
            topLeft_bottomRight.append((yPoints[y], xPoints[x], yPoints[y+1],xPoints[x+1]))
    
    return topLeft_bottomRight