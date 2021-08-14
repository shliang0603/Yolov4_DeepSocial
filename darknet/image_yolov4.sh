

#./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights data/dog.jpg -i 0 -thresh 0.25

# 使用作者提供的预训练的模型
./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ../weights/DeepSocial.weights data/dog.jpg -i 0 -thresh 0.25



