# Edge Detection
Special Usecase of Convolution operation is "Edge Detection"
Edge - A boundary separating 2 heterogeneous areas. Heteroginity can be based on several criteria.
Edge Based On Criteria - Intensity Edge; Colour Edge; Texture Edge

# What is Edge Detection?
Intensity is vector value.

Image with half black and half white. 
    mat = np.zeros((10,20))
    mat[:, 10:] = 1

Rowwise Differentiation will give us edge point.
    kernet = [-1, 1]
    vec = mat[1,:]

    2D: kernel = np.array([[0,1],[-1,0]])

    Robert Cross Operator
    Some advance kernel - Prewitt Operator and Sobel Operator

# How Edge Detection Applied?
