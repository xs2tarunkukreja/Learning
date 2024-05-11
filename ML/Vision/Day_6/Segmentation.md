# Object Segmentation

## Why we needs?

# What is the problem with CNN in creating mask for Segmentation?
If we reduce dimensionality of an image, 
    Objective: Bounding mask to same image.
    Input Image Size should be equal to Output Image size.
    CNN - It keep reducing size/dimension of image.
Solution - padding so input and output size remain same. Padding can be issue.
    Adding extra information that is not actual part of image.
    So, it impact accuracy.

Approach 1 - Each layer maintain size.
Approach 2 - N Layers decrease size then make it of same size.

Approach 2 is better. We are not adding any extra information. So, let CNN extract detail as much as possible. 
    Encoding - Where we are reducing the size.
    Decoding - Again output size to input size. How? so that don't loss information.

7*7 > 5*5 > 3*3 > 2*2 - Encoding
7*7 < 5*5 < 3*3 < 2*2 - Decoding - We don't want to loss information.
    We can add actual image by using skip connection. Now we can increase size without lossing information.

## U-Net
Use same as encoding and then decoding.
    Transpose Convolutional - Name of approach in decoding
    Encode Output 2*2 * Filter 2*2 = Then Output 4 * 4 with stride 2.

0 1
2 3

Now suppose, an image with 3 object of same class. Problem of U-Net
    if we have neigbour object then mask will cover both. instead of 2 masks...

Instance Segmentation -
Assign different colors near by. How?
    2 ROI through Object Detection - chances are really high.

MaskFRCNN - Mask Faster RCNN
Another Approach - U-Net + Watershed model
    Intensity Plot of Image > Fill Water Here.. If water goes through connecting top of one/another from one hole to another... it means 2 images.
    We can use texture, color or anything change to differentiate.
