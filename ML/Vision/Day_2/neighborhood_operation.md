# Neighborhood Operations
Context
If we just see a pixel then we can't say anything about it.
Important - Observe part of image in context with other part.

# What is Neighborhood?
Neighborhood of a pixel is a group of pixels connect to or near to pixel.
Neighborhood = 3. Means 3 * 3 matrix and pixel at center.

Image > Apply operation like mean etc on neighborhood > New Image
    New Image is smaller in size.
    New Image is Blur Image
    New Image resemble the Original Image.

Neighborhood matrix is called kernel. we slide this kernel and provide operation based on weight of the kernel.

# Convolution Operation - Basic Building Block of CNNs.
In 2D Convolution operation, a small matrix called Kernel is slide over the image and foreach slide some operation is performed between kernel and the image. Resultant image vectors of the operations when stiched together in 2D create a resultant image which is smaller in dimensions to original image.
