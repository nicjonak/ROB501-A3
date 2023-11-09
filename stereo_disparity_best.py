import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Code goes here...
   
    #Used similar SAD algorithm as before but read that smoothing improves 
    #performance and that adjusting window size can also improve performance.
    #I decided to set my window size to 8 and smooth my output with a median 
    #filter to attempt to get better results. When running through the test 
    #files this algorithm acheived RMSerrors and %bad within the acceptable 
    #ranges set in the assignment outline for both the cone/teddy data as 
    #well as for the kitti data

    ws = 8 #window size
    Id=np.zeros_like(Il) #output image arrray
    Irp=np.pad(Ir,ws,'constant',constant_values=0) #pad images for convolution with kernel size of ws
    Ilp=np.pad(Il,ws,'constant',constant_values=0)

    #loop through pixels in bounding box and check SAD score with other pixels in row from a disparity of -1 to maxd
    for j in range(bbox[1][0]+ws, bbox[1][1]+ws):
        for i in range(bbox[0][0]+ws, bbox[0][1]+ws):
            BestD = 0
            BestSAD = float('inf')
            for disparity in range(-1,maxd):
                if i-disparity-ws<0 or i-disparity+ws+1>Il.shape[1]: #check bounds
                    continue
                winL = Ilp[j-ws:j+ws+1, i-ws:i+ws+1]
                winR = Irp[j-ws:j+ws+1, i-disparity-ws:i-disparity+ws+1]
                SAD = np.sum(np.absolute(winL.flatten() - winR.flatten())) #compute SAD
                if SAD < BestSAD:
                    BestSAD = SAD
                    BestD = abs(disparity)
            Id[j-ws][i-ws] = BestD 

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")
    

    Id=median_filter(Id,10) #filter to smooth disparity map
    return Id
