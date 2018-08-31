import numpy as np

def IntegratedImage(img):
    row, col = img.shape
    result = np.zeros(shape = [row, col])

    #first, we have to initialize the first row
    for i in range(0, col):
        if i is 0:
            result[0, i] = img[0, i]
        else:
            result[0, i] = img[0, i] + result[0, i - 1]

    #initialize the first column
    for i in range(0, row):
        if i is 0:
            result[i, 0] = img[i, 0]
        else:
            result[i, 0] = img[i, 0] + result[i - 1, 0]

    #using dynamic programming to compute other pixels
    for i in range(1, row):
        for j in range(1, col):
            result[i, j] = result[i, j-1] + result[i-1, j] - result[i-1, j-1] + img[i, j]
    return result

#calculate the region
def sum_region(integratedimage, topleft, bottomright):
    i_topleft, j_topleft = topleft[0], topleft[1]
    i_bottomright, j_bottomright = bottomright[0], bottomright[1]
    i_topright, j_topright = topleft[0], bottomright[1]
    i_bottomleft, j_bottomleft = bottomright[0], topleft[1]
    area = integratedimage[i_bottomright, j_bottomright] - integratedimage[i_topright, j_topright] - integratedimage[i_bottomleft, j_bottomleft] + integratedimage[i_topleft, j_topleft]
    return area

