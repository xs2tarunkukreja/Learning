# Neighborhood Operations
Context
If we just see a pixel then we can't say anything about it.
Important - Observe part of image in context with other part.

# What is Neighborhood?
Neighborhood of a pixel is a group of pixels connect to or near to pixel.
Neighborhood = 3. Means 3 * 3 matrix and pixel at center.

When we add neighbour, amount of information increase. Not only amount but quality as well. So, more meaningful information.

We can find max, min and mean of neighbour window.

5 * 5 matrix image; 3*3 neighbour window.

Output will be 3*3 matrix
O[0,0] = Operation(Image[0:3, 0:3])
O[0,1] = Operation(Image[0:3, 1:4])
O[0,2] = Operation(Image[0:3, 2:])

O[1,0] = Operation(Image[1:4, 0:3])
O[1,1] = Operation(Image[1:4, 1:4])
O[1,2] = Operation(Image[1:4, 2: ])

O[2,0] = Operation(Image[2: , 0:3])
O[2,1] = Operation(Image[2: , 1:4])
O[2,2] = Operation(Image[2: , 2: ])

This type of operation in convolutional operation. Operation is aggregation on that particular range.
We are getting same information as well. (Not 100%)

Image > Apply operation like mean etc on neighborhood > New Image
    New Image is smaller in size.
    New Image is Blur Image
    New Image resemble the Original Image.

Neighborhood matrix is called kernel. we slide this kernel and provide operation based on weight of the kernel.

max operation retain edges (also in term of sharpness). [mean = blur; min= edges merge]

# Convolution Operation - Basic Building Block of CNNs.
In 2D Convolution operation, a small matrix called Kernel is slide over the image and foreach slide some operation is performed between kernel and the image. Resultant image vectors of the operations when stiched together in 2D create a resultant image which is smaller in dimensions to original image.

Image Portion * kernel => operation can be any like *

If we want to retain original size then we have to pad the image.. Add columns left and rights and row in top and bottom. - Padding

Stride - Number of pixel the kernel move in each slide.

Data Augmentation - 

What analytics value from convolutional operation? It depends on kernel.

Spatial Dependency.

It do data preprocess in the start of pipeline.