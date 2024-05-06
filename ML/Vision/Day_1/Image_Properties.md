# Properties of Digital Image
Properties help to manipulate or extract some information.

Properties
    Image Resolution
    Image Brightness or Contrast
    Histogram

## Image Resolution
It means ability to distinguish two different and nearby/close entities. Smaller distance between 2 entities which we can separate, higher the resolution.
### Spatial Resolution
It refer the ability to distinguish 2 close by objects in an image. It depends on "Size of Pixel". Smaller pixel size, finner the detail we can observe in.
Camera is improving this day by day.

Binning - ?

### Spectral Resolution
Ability to distinguish between different wavelength of electromagnetic spectrum. Help to distinguish between colors.

All crayon in grayscale image looks nearly same.

### Temportal Resolution
Temporal resolution is the ability to distinguish between 2 images taken at different time intervals.
Come into picture when sequence of images (as in movie)

## Image Brightness or Contrast
Intensity level of the pixels.
### Brightness
Average Pixel Intensity level of image. Low means dark image. High means blown up image. Balanced means good image.

### Contrast
Intensity difference between neighbouring objects. So, neighbouring objects are distinctly visible.

### Perceptual Contrast
Contrast or Brightness of a pixel in an image are effected by intensity level of neighboring pixels.

## Image Histogram
It represents the variation of different ranges of values present in a set of data. It is a graphical display of number of occurances of data points in different ranges.

### Histogram Stretch 
It is the process of mapping intensity level of image using a function such that a range of intensity levels in the image improves.
#### Linear Stretch
 temp[:, :, i] = (image[:, :, i] - minI)*(((maxO - minO)/(maxI - minI)) + minO)
 temp = (image - minI)*(((maxO - minO)/(maxI - minI)) + minO)
### Logarithmic Stretch
    Higher intensity level are maintained whereas the lower level are given a boost.
    Enhance in dark area but not blow off in brighter area.
