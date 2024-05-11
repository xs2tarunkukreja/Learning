# Properties of Digital Image
Properties help to manipulate or extract some information.

Properties
    Image Resolution
    Image Brightness or Contrast
    Histogram

## Image Resolution
It means ability to distinguish two different and nearby/close entities. Smaller distance between 2 entities which we can separate, higher the resolution.
### Spatial Resolution
Number of pixels

It refer the ability to distinguish 2 close by objects in an image. It depends on "Size of Pixel". Smaller pixel size, finner the detail we can observe in.
Camera is improving this day by day.

Binning - Increaing size of pixel. Merging N pixel to 1 by taking average. Way to decrease the size of image - i/p size decrease - nodes per layer decrease but image have information.

High resolution means pixel size is small.

Low to High Resolution: How?
    AI by training from differen images.
    Can't do with this one image.

### Spectral Resolution
Ability to distinguish between different wavelength of electromagnetic spectrum. Help to distinguish between colors.

All crayon in grayscale image looks nearly same.

### Temportal Resolution
Temporal resolution is the ability to distinguish between 2 images taken at different time intervals.
Come into picture when sequence of images (as in movie)

Number of image per frame.

## Image Brightness or Contrast
Intensity level of the pixels.
### Brightness
Average Pixel Intensity level of image. Low means dark image. High means blown up image. Balanced means good image.

Increase brightness means difference between black and white increase.

New Pixel = Multiple each pixel with some value / Divide by Maximum.

### Contrast
Intensity difference between neighbouring objects. So, neighbouring objects are distinctly visible.

### Perceptual Contrast
Contrast or Brightness of a pixel in an image are effected by intensity level of neighboring pixels.

## Image Histogram
It represents the variation of different ranges of values present in a set of data. It is a graphical display of number of occurances of data points in different ranges.
### Histogram Stretch 
It is the process of mapping intensity level of image using a function such that a range of intensity levels in the image improves.

Contrast Enhancement happened.

#### Linear Stretch
 temp[:, :, i] = (image[:, :, i] - minI)*(((maxO - minO)/(maxI - minI)) + minO)
 temp = (image - minI)*(((maxO - minO)/(maxI - minI)) + minO)

### Logarithmic Stretch
    Higher intensity level are maintained whereas the lower level are given a boost.
    Enhance in dark area but not blow off in brighter area.
