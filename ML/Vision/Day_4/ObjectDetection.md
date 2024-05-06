# Object Detection
## Sliding Window Algorithm
For each of this window, we would normally take window region and apply image classifier to determine if the window has an object that interest us. Problem -
    Object at different scales and orientation.
    Very Slow

## Image Pyramid
x*y image to /2 > /4 > /8 so on.
It is a multiscale representation of an image.
Utilizing it allow us to find object at different scale.
Image Pyramid + Slide Window

Problem -
    Slow
    Sensitive to Parameter choice.

## Object Detection Using Deep Learning
Input - Image
Output - Class + Object Bounding Box

### Evaluation Metrics
Classification - Accuracy: Percentage of correct prediction.
Object Detection and Segmentation - Intersection Over Union (IoU) = Area of Overlap/Area of Union

### Generalized Intersection Over Union
GIoU = IoU - |C\(A U B)|/|C|

## Bounding Box Regression

## Region Proposal
    Find blobby image region that are likely to contain object.
    Relative fast to run. e.g. Selective search give 1000  region proposals in few seconds on CPU

## Selective Search
    Uses the fact that image can be over-segmented to automatically identify location in an image that could contain images.
    Extract regions based on
        Color similarity
        Texture similarity
        Size similarity
        Meta Similarity - which is linear combination of the above.

# Region Based CNN (RCNN)
Input Image > Extract Region Proposal (~2K) > Compute CNN features > Classify Regions
Problems
    Training is slow, takes a lot of disk space.
    Inference is slow

# Fast RCNN
Pretain a CNN on Image classification task
Propose region by selective search
Alter pre-trained CNN
    Replace last max pooling layer with ROI Polling Layer
?

Problems
    Calculating region proposal is most time consiming.

# Faster RCNN

## Regional Proposal Network