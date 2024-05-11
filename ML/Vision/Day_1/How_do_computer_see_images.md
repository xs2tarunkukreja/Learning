# How to computer see images?
Difficulty - Computer work and understand digitally. So, image should be converted to digital form.

Convert light to digital image
    Light (7 Wavelengths) > Sensor > Sampling (Take RGB only) > Quantization (8 bit, 16 bit) > Encoding (JPEG, GIF) > Actual Image

    Sampling means small sample but true representation of pupulation i.e. image

90% cases grey scale (1 channel) is sifficient + 10% where color actual play important role like differentiate between layes packets.

pip install opencv-contrib-python

Pixel - Each number in a image matrix is called a pixel. It is basic unit of image. Pixel is level to which we can distinguish or perceive change in the "GrayScale" intensity change.  

RGB Image - 3D Vector per pixel - One Dimension for each of Red, Green and Blue color
    R + G + B = all top means white
    If all 0 means black.

Tools to read image -
    OpenCV 
    Pillow
    Scikit-Image
    MatPlotLib

imread() and imshow()

Must Read - https://towardsdatascience.com/image-manipulation-tools-for-python-6eb0908ed61f

Color Images -
    3D Matrices - 3 * 2D Matrices
    Color Channel.

OpenCV - By default, it read the image as BGR which needs to be converted to RGB.
    im_cv2_rgb = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)
    Why BGR not in RGB? - Very old when sensor where BGR based.

One more channel - Opacity.