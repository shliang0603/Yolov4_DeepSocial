
# 先克隆darknet的仓库
# !git clone https://github.com.cnpmjs.org/AlexeyAB/darknet




# 导入必要的库包
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from base64 import b64decode, b64encode




# 为了在Python代码中使用Yolov4，我们需要从`darknet.py` 中导入预构建的函数
# 函数的具体定义可以在darknet.py文件中查看

# from darknet.darknet import *
from darknet.darknet import load_network, network_width, network_height
from darknet.darknet import make_image, copy_image_from_bytes, free_image
from darknet.darknet import detect_image, bbox2points


# 测试yolov4网络
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("./darknet/cfg/yolov4.cfg", "./darknet/cfg/coco_deepsocial.data", "./weights/DeepSocial.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio





from src.sort import *
mot_tracker    = Sort(max_age=25, min_hits=4, iou_threshold=0.3)


#  
Input            = "./Images/OxfordTownCentreDataset.avi"
ReductionFactor  = 2
calibration      = [[180,162],[618,0],[552,540],[682,464]]


# DeepSocial Configration

from src.deepsocial import *

######################## Frame number
StartFrom  = 0 
EndAt      = 500                       #-1 for the end of the video
######################## (0:OFF/ 1:ON) Outputs
CouplesDetection    = 1                # Enable Couple Detection 
DTC                 = 1                # Detection, Tracking and Couples 
SocialDistance      = 1
CrowdMap            = 1
# MoveMap             = 0
# ViolationMap        = 0             
# RiskMap             = 0
######################## Units are Pixel
ViolationDistForIndivisuals = 28 
ViolationDistForCouples     = 31
####
CircleradiusForIndivsual    = 14
CircleradiusForCouples      = 17
######################## 
MembershipDistForCouples    = (16 , 10) # (Forward, Behind) per Pixel
MembershipTimeForCouples    = 35        # Time for considering as a couple (per Frame)
######################## (0:OFF/ 1:ON)
CorrectionShift  = 1                    # Ignore people in the margins of the video
HumanHeightLimit = 200                  # Ignore people with unusual heights
########################
Transparency        = 0.7
######################## Output Video's path
Path_For_DTC = os.getcwd() + "/result/DeepSOCIAL_DTC.avi"
Path_For_SocialDistance = os.getcwd() + "/result/DeepSOCIAL_Social_Distancing.avi"
Path_For_CrowdMap = os.getcwd() + "/result/DeepSOCIAL_Crowd_Map.avi"


def extract_humans(detections):
    detetcted = []
    if len(detections) > 0: # At least 1 detection in the image and check detection presence in a frame  
        idList = []
        id = 0
        for label, confidence, bbox in detections:
            if label == 'person': 
                xmin, ymin, xmax, ymax = bbox2points(bbox)
                id +=1
                if id not in idList: idList.append(id)
                detetcted.append([int(xmin), int(ymin), int(xmax), int(ymax), idList[-1]])
    return np.array(detetcted)

def centroid(detections, image, calibration, _centroid_dict, CorrectionShift, HumanHeightLimit):
    e = birds_eye(image.copy(), calibration)
    centroid_dict = dict()
    now_present = list()
    if len(detections) > 0:   
        for d in detections:
            p = int(d[4])
            now_present.append(p)
            xmin, ymin, xmax, ymax = d[0], d[1], d[2], d[3]
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w/2
            y = ymax - h/2
            if h < HumanHeightLimit:
                overley = e.image
                bird_x, bird_y = e.projection_on_bird((x, ymax))
                if CorrectionShift:
                    if checkupArea(overley, 1, 0.25, (x, ymin)):
                        continue
                e.setImage(overley)
                center_bird_x, center_bird_y = e.projection_on_bird((x, ymin))
                centroid_dict[p] = (
                            int(bird_x), int(bird_y),
                            int(x), int(ymax), 
                            int(xmin), int(ymin), int(xmax), int(ymax),
                            int(center_bird_x), int(center_bird_y))

                _centroid_dict[p] = centroid_dict[p]
    return _centroid_dict, centroid_dict, e.image

def ColorGenerator(seed=1, size=10):
    np.random.seed = seed
    color=dict()
    for i in range(size):
        h = int(np.random.uniform() *255)
        color[i]= h
    return color

def VisualiseResult(_Map, e):
    Map = np.uint8(_Map)
    histMap = e.convrt2Image(Map)
    visualBird = cv2.applyColorMap(np.uint8(_Map), cv2.COLORMAP_JET)
    visualMap = e.convrt2Image(visualBird)
    visualShow = cv2.addWeighted(e.original, 0.7, visualMap, 1 - 0.7, 0)
    return visualShow, visualBird, histMap




cap = cv2.VideoCapture(Input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
height, width = frame_height // ReductionFactor, frame_width // ReductionFactor
print("Video Reolution: ",(width, height))

if DTC: DTCVid = cv2.VideoWriter(Path_For_DTC, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
if SocialDistance: SDimageVid = cv2.VideoWriter(Path_For_SocialDistance, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
if CrowdMap: CrowdVid = cv2.VideoWriter(Path_For_CrowdMap, cv2.VideoWriter_fourcc(*"XVID"), 30.0, (width, height))

colorPool = ColorGenerator(size = 3000)
_centroid_dict = dict()
_numberOFpeople = list()
_greenZone = list()
_redZone = list()
_yellowZone = list()
_final_redZone = list()
_relation = dict()
_couples = dict()
_trackMap = np.zeros((height, width, 3), dtype=np.uint8)
_crowdMap = np.zeros((height, width), dtype=np.int) 
_allPeople = 0
_counter = 1
frame = 0

while True:
    print('-- Frame : {}'.format(frame))
    prev_time = time.time()
    ret, frame_read = cap.read()
    if not ret: break

    frame += 1
    if frame <= StartFrom: continue
    if frame != -1:
        if frame > EndAt: break
        
    frame_resized = cv2.resize(frame_read,(width, height), interpolation=cv2.INTER_LINEAR)
    image = frame_resized
    e = birds_eye(image, calibration)
    detections, width_ratio, height_ratio = darknet_helper(image, width, height)
    humans = extract_humans(detections)
    track_bbs_ids = mot_tracker.update(humans) if len(humans) != 0 else humans

    _centroid_dict, centroid_dict, partImage = centroid(track_bbs_ids, image, calibration, _centroid_dict, CorrectionShift, HumanHeightLimit)
    redZone, greenZone = find_zone(centroid_dict, _greenZone, _redZone, criteria=ViolationDistForIndivisuals)
    
    if CouplesDetection:
        _relation, relation = find_relation(e, centroid_dict, MembershipDistForCouples, redZone, _couples, _relation)
        _couples, couples, coupleZone = find_couples(image, _centroid_dict, relation, MembershipTimeForCouples, _couples)
        yellowZone, final_redZone, redGroups = find_redGroups(image, centroid_dict, calibration, ViolationDistForCouples, redZone, coupleZone, couples , _yellowZone, _final_redZone)
    else:
        couples = []
        coupleZone = []
        yellowZone = []
        redGroups = redZone
        final_redZone = redZone


    if DTC:
        DTC_image = image.copy()
        _trackMap = Apply_trackmap(centroid_dict, _trackMap, colorPool, 3)
        DTC_image = cv2.add(e.convrt2Image(_trackMap), image) 
        DTCShow = DTC_image
        for id, box in centroid_dict.items():
            center_bird = box[0], box[1]
            if not id in coupleZone:
                cv2.rectangle(DTCShow,(box[4], box[5]),(box[6], box[7]),(0,255,0),2)
                cv2.rectangle(DTCShow,(box[4], box[5]-13),(box[4]+len(str(id))*10, box[5]),(0,200,255),-1)
                cv2.putText(DTCShow,str(id),(box[4]+2, box[5]-2),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
        for coupled in couples:
            p1 , p2 = coupled
            couplesID = couples[coupled]['id']
            couplesBox = couples[coupled]['box']
            cv2.rectangle(DTCShow, couplesBox[2:4], couplesBox[4:], (0,150,255), 4)
            loc = couplesBox[0] , couplesBox[3]
            offset = len(str(couplesID)*5)
            captionBox = (loc[0] - offset, loc[1]-13), (loc[0] + offset, loc[1])
            cv2.rectangle(DTCShow,captionBox[0],captionBox[1],(0,200,255),-1)
            wc = captionBox[1][0] - captionBox[0][0]
            hc = captionBox[1][1] - captionBox[0][1]
            cx = captionBox[0][0] + wc // 2
            cy = captionBox[0][1] + hc // 2
            textLoc = (cx - offset, cy + 4)
            cv2.putText(DTCShow, str(couplesID) ,(textLoc),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
        DTCVid.write(DTCShow)


    if SocialDistance:
        SDimage, birdSDimage = Apply_ellipticBound(centroid_dict, image, calibration, redZone, greenZone, yellowZone, final_redZone, coupleZone, couples, CircleradiusForIndivsual, CircleradiusForCouples)
        SDimageVid.write(SDimage)


    if CrowdMap:
        _crowdMap, crowdMap = Apply_crowdMap(centroid_dict, image, _crowdMap)
        crowd = (crowdMap - crowdMap.min()) / (crowdMap.max() - crowdMap.min())*255
        crowd_visualShow, crowd_visualBird, crowd_histMap = VisualiseResult(crowd, e)
        CrowdVid.write(crowd_visualShow)


    cv2.waitKey(3)
print('::: Analysis Completed')

cap.release()
if DTC: DTCVid.release(); print("::: Video Write Completed : ", Path_For_DTC)
if SocialDistance: SDimageVid.release() ; print("::: Video Write Completed : ", Path_For_SocialDistance)
if CrowdMap: CrowdVid.release() ; print("::: Video Write Completed : ", Path_For_CrowdMap)
