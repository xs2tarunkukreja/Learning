# Image Segmentation
We need to tell the boundary as well.

Car Drive
Stone Detection

## What is issue with Object Detection in automatic car?
Box having Road to drive. Some corner doesn't have road but our car treat it as road.

## WHat is Image Segmentation?
Segmentation an image means partitioning an image into different regions of homogenous pixels such that:
    Union of all region = The Image
    Each region is homogeneous. Homogeneity is based on critria based on problem one is trying to solve.
    Any pair of adjacent region is not homogeneous.
    Region should have low internal variance
    Region should be made of spatially contiguous pixels.

## How to separate out tumor from other?
### Non-Contextual
Threshold.
Create multiple threshold.

Issue - It is based on intensity. If 2 object have same color.
    How to select threshold.

How to select threshold?
    K-Mean
        5 clusters
    We can curve of variance.. and check for threshold.

### Contextual

    Region Growing
    Region Splitting
        We start with each point separately.  N points with N * N matrix.. It contain differences.. We keep adding if the difference is very small.

        Max and Min Intensity and Take the differnece
        If difference within threshold then same region else divide that into equal parts.

### Deep Learning Models
Semantic Segmentation - One color for one type of object.
    Issue - Can't count objects. Shelf - can't count a and b objects and space.
Instance Segmentation
Sallient Object Segmentation
Portrait Segmentation - Remove Background
Image Matting - keep background. Remove objects.

## Models
UNet
FCN