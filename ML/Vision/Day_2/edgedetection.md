# Edge Detection
Special Usecase of Convolution operation is "Edge Detection"
Edge - A boundary separating 2 heterogeneous areas. Heteroginity can be based on several criteria.
Edge Based On Criteria - Intensity Edge; Colour Edge; Texture Edge

# What is Edge Detection?
Intensity is vector value.

Image with half black and half white. 
    mat = np.zeros((10,20))
    mat[:, 10:] = 1

    Now take first row, for 0-9 - it is 0 and 10-: it is one.
    Now what operation to find edge in 1st row.
    a[n+1] - a[n] = 1 then it is an edge. (differentiation.)
Rowwise Differentiation will give us edge point.

Row - 
    kernel = [-1, 1] // Same as differentiation.
    vec = mat[1,:]

2D: kernel = np.array([[0,1],[-1,0]])


Now complex image like bicycle with slant grills gate.
    Robert Cross Operator
    Some advance kernel - Prewitt Operator and Sobel Operator

        -1 0 1     -1 -1 -1
        -1 0 1      0  0  0   - prewitt operator - Verticle (3*3) and Horizontal (3*3)
        -1 0 1      1  1  1

        -1 0 1      -1 -2 -1
        -2 0 2       0  0  0    Sovbel Operator - Verticle (3*3) and Horizontal (3*3)
        -1 0 1       1  2  1 

It is for horizontal and vertical. (even diagnol one.)

other important filters -
https://blog.paperspace.com/filters-in-convolutional-neural-networks

# How Edge Detection Applied?
A tablets(capsule shape). Detect defective tablets. Tablet have a line in center.
Edge detection means edge whites.. other black.

Above filters detect edges but distortition due to tablet pata...

vertical edge detect medince in pouch (pata).
horizontale edge detect line on tablet.

How to take care of distortition? 
    Data Preprocessing
        Bluring Image Using Meidan > Threshold operation on blurred image.


