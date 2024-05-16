# YOLO - Object Detection
Object Detection - Classification + Regression (2 stages)

Convert the same to 1 stage model.

anchor points; anchor box [cx, cy, h, w]
    Issue - Multiple Object of Different Size: It will fails. As vector will be N-Dimension. 
        Vector Size/Dimension will be different for different images.


Divide image into N equal size box.. Now share status based on each box.
    Issue - If same box have 2 objects.
        Solution 1 - Reduce box sie. But it impact performance.
        Solution 2 - Predict N classes... So, we can predict 2 classes for same box.

Yolo = Combine Solution 1 and 2. It takes only 2 objects per box.

Downoad yolo.h5 file