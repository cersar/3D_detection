# 3D_detection
This work is inspired by image-to-3d-bbox(https://github.com/experiencor/image-to-3d-bbox), which is an an implementation of the paper "3D Bounding Box Estimation Using Deep Learning and Geometry" (https://arxiv.org/abs/1612.00496).

Instead of using kitti's 3-D truth, i mainly make two supplements:    
1、Compute 3-D box center by 2-D box and network's output  
2、Compute theta_ray by 2-D box center  
Besides, I make some changes to the code structure.

By now, there are still several problems to be solved, for example:  
1、The number of situations is 256 in this work, whereas it is 64 in the paper.  
2、When detecting, i use objects's truncated and occluded level in kitti's label file to decide whether to generate 3D box, whereas it is reasonable to generate these by the trained neural network.

This is just a raw version, welcome to share your ideas to improve it!

Result on kitti:  
![000254.jpg](https://github.com/cersar/3D_detection/blob/master/output/000254.jpg)  
![000074.jpg](https://github.com/cersar/3D_detection/blob/master/output/000074.jpg)  
![000154.jpg](https://github.com/cersar/3D_detection/blob/master/output/000154.jpg)  
## Useage:

If you want to train, after fixing paths in the train.py, just run:
<pre><code>python3 train.py
</code></pre>
In this way, you can get your own weights file, or you can download the pretrained file from  https://pan.cstcloud.cn/web/share.html?hash=7dct49xER5w  
In the detection time, after fixing paths in the detection.py, just run:
<pre><code>python3 detection.py
</code></pre>

