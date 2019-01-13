# 3D_detection
This work is inspired by image-to-3d-bbox(https://github.com/experiencor/image-to-3d-bbox), which is an an implementation of the paper "3D Bounding Box Estimation Using Deep Learning and Geometry" (https://arxiv.org/abs/1612.00496).

Instead of using kitti's 3-D truth, i mainly make two supplements compared to image-to-3d-bbox(https://github.com/experiencor/image-to-3d-bbox):    
1、Compute 3-D box center by 2-D box and network's output  
2、Compute theta_ray by 2-D box center  
Besides, I make some changes to the code structure.

By now, there are still several problems to be solved:  
1、The orientation loss is not consist with what was discussed the paper.  
2、The number of situations is 256 in this work, whereas it is 64 in the paper.  

Result on kitti:  
![000031.png](https://raw.githubusercontent.com/cersar/picture/master/000031.png)
## Useage:

If you want to train, run:
<pre><code>python3 train.py
</code></pre>
In this way, you can get your own weights file, or you can download from  https://1drv.ms/u/s!ApXgmQqTQot_hWWR4RDORU9jsxRP  
Then run detection:
<pre><code>python3 detection.py
</code></pre>


