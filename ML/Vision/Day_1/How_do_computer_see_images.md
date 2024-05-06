# How to computer see images?
Difficulty - Computer work and understand digitally. So, image should be converted to digital form.

Convert light to digital image
    Light > Sensor > Sampling > Quantization (8 bit, 16 bit) > Encoding (JPEG, GIF) > Actual Image

pip install opencv-contrib-python

Pixel - Each number in a image matrix is called a pixel. It is basic unit of image. Pixel is level to which we can distinguish or perceive change in the "GrayScale" intensity change.  

RGB Image - 3D Vector per pixel - One Dimension for each of Red, Green and Blue color

Tools to read image -
    OpenCV 
    Pillow
    Scikit-Image
    MatPlotLib

Must Read - https://towardsdatascience.com/image-manipulation-tools-for-python-6eb0908ed61f

Color Images -
    3D Matrices - 3 * 2D Matrices
    Color Channel.

OpenCV - By default, it read the image as BGR which needs to be converted to RGB.
    im_cv2_rgb = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)