# Object Detection
Image > Classification CAT > Classification and Localization - We identify cat and tell position in ractangle > Object Detection ( Localization when we have lots of animals/objects in image) > Instance segmentations (exact shape and position of objects)

Multiple Objects.

## Sliding Window Algorithm
For each of this window, we would normally take window region and apply image classifier to determine if the window has an object that interest us. Problem -
    Object at different scales and orientation.
    Very Slow

Now create window of differnet size.

Also we can reduce the image size.

## Image Pyramid
x*y image to /2 > /4 > /8 so on.
It is a multiscale representation of an image.
Utilizing it allow us to find object at different scale.
Image Pyramid + Slide Window

Problem -
    Slow
    Sensitive to Parameter choice - What should be initial size of window? Where should we stop?

## Object Detection Using Deep Learning
Input - Image
Output - Class + Object Bounding Box

### Evaluation Metrics
Classification - Accuracy: Percentage of correct prediction.
Object Detection and Segmentation - Intersection Over Union (IoU) = Area of Overlap/Area of Union

### What should be evaluation paraeter/matrix in term of Object Detection?

Here we are talking about 2 ractangle - one window wherr we found the object. (2) where actual object exist. 
Area of Intersection/Area of Union

In case the ractangles don't intersect than IoU = 0 in both cases if it is very close or very far. For that GIoI come into picture.

### Generalized Intersection Over Union
GIoU = IoU - |C\(A U B)|/|C|

## Bounding Box Regression
Here we run NN which predict x1, x2, y1, y2 for any object type. For e.g. airplane. It will search for airplance in image and predict box for the same.

## Region Proposal
    Find blobby image region that are likely to contain object.
    Relative fast to run. e.g. Selective search give 1000  region proposals in few seconds on CPU

    Mostly Used.

### Selective Search
    Uses the fact that image can be over-segmented to automatically identify location in an image that could contain images.
    Extract regions based on
        Color similarity
        Texture similarity
        Size similarity
        Meta Similarity - which is linear combination of the above.

    Contour - Connect all edges together.
    Find rectangle which contain contour.

# Region Based CNN (RCNN)
Input Image > Extract Region Proposal (~2K) > Compute CNN features (separate for each region) > Classify Regions
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

Single CNN layer.

## Regional Proposal Network
Here we are creating random size box.
