# Basic Operation
    Tunning its pixel intensity
    Affline Transformation -
        Translate
        Rotate
        Resize
        Crop
        Flip
    Spectral Operation - Threshold
    Logical Operation - Bitwise AND, OR, NOT

# Affline Transformation
That preserve collinearity(i.e. all points lying on a line initially still lie on the line after transformation)

## Translation
Shifting each pixel by some positions.
## Rotate
Tranforming pixels of an image by an angle and with a reference to a point.
## Resize
Scaling or converting it to different size. Interploting the original one.
## Crop
Removing the area of an image and only retaining a portion of image.
## Flip
Mirror Image - Horizontally or Vertically.

# Spectral Operation
Work on pixel intensity and spectral channel. Don't change the spatial positions.
## Threshold
Convert the image into binary image - two level of intensities.
E.g. - img > threshold ? 255 : 0

# Bitwise Operations
Masking Images
