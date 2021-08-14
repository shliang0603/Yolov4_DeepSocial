# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.12 |Anaconda, Inc.| (default, Sep  8 2020, 23:10:56) 
# [GCC 7.3.0]
# Embedded file name: deepsocial.py
# Compiled at: 2021-03-06 05:54:51
# Size of source mod 2**32: 14036 bytes
import cv2, numpy as np
from itertools import combinations

def find_zone(centroid_dict, _greenZone, _redZone, criteria):
    redZone = []
    greenZone = []
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
        distance = Euclidean_distance(p1[0:2], p2[0:2])
        if distance < criteria:
            if id1 not in redZone:
                redZone.append(int(id1))
            if id2 not in redZone:
                redZone.append(int(id2))

    for idx, box in centroid_dict.items():
        if idx not in redZone:
            greenZone.append(idx)

    return (
     redZone, greenZone)


def find_relation(e, centroid_dict, criteria, redZone, _couples, _relation):
    pairs = list()
    memberships = dict()
    for p1 in redZone:
        for p2 in redZone:
            if p1 != p2:
                distanceX, distanceY = Euclidean_distance_seprate(centroid_dict[p1], centroid_dict[p2])
                if p1 < p2:
                    pair = (
                     p1, p2)
                else:
                    pair = (
                     p2, p1)
                if _couples.get(pair):
                    distanceX = distanceX * 0.6
                    distanceY = distanceY * 0.6
                if distanceX < criteria[0]:
                    if distanceY < criteria[1]:
                        if memberships.get(p1):
                            memberships[p1].append(p2)
                        else:
                            memberships[p1] = [
                             p2]
                    if pair not in pairs:
                        pairs.append(pair)

    relation = dict()
    for pair in pairs:
        if _relation.get(pair):
            _relation[pair] += 1
            relation[pair] = _relation[pair]
        else:
            _relation[pair] = 1

    obligation = {}
    for p in memberships:
        top_relation = 0
        for secP in memberships[p]:
            if p < secP:
                pair = (
                 p, secP)
            else:
                pair = (
                 secP, p)
            if relation.get(pair) and top_relation < relation[pair]:
                top_relation = relation[pair]
                obligation[p] = secP

    couple = dict()
    for m1 in memberships:
        for m2 in memberships:
            if m1 != m2 and obligation.get(m1) and obligation.get(m2) and obligation[m1] == m2:
                if obligation[m2] == m1:
                    if m1 < m2:
                        pair = (
                         m1, m2)
                    else:
                        pair = (
                         m2, m1)
                couple[pair] = relation[pair]

    return (
     _relation, couple)


def find_couples(img, centroid_dict, relation, criteria, _couples):
    couples = dict()
    coupleZone = list()
    for pair in relation:
        proxTime = relation[pair]
        if proxTime > criteria:
            coupleZone.append(pair[0])
            coupleZone.append(pair[1])
            couplesBox = center_of_2box(centroid_dict[pair[0]], centroid_dict[pair[1]])
            if _couples.get(pair):
                couplesID = _couples[pair]['id']
                _couples[pair]['box'] = couplesBox
            else:
                couplesID = len(_couples) + 1
                _couples[pair] = {'id':couplesID,  'box':couplesBox}
            couples[pair] = _couples[pair]

    return (
     _couples, couples, coupleZone)


def find_redGroups(img, centroid_dict, calibration, criteria, redZone, coupleZone, couples, _yellowZone, _red_without_yellowZone):
    e = birds_eye(img, calibration)
    redGroups = list()
    for p1, p2 in couples:
        x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
        centerGroup_bird = e.projection_on_bird((x, ymax))
        for p, box in centroid_dict.items():
            if p != p1 and p != p2:
                center_bird = (
                 box[0], box[1])
                distance = Euclidean_distance(center_bird, centerGroup_bird)
                if distance < criteria:
                    redGroups.append(p1)
                    redGroups.append(p2)
                    redGroups.append(p)

    yellowZone = list()
    for p1, p2 in couples:
        if p1 not in redGroups and p2 not in redGroups:
            yellowZone.append(p1)
            yellowZone.append(p2)

    red_without_yellowZone = list()
    for id, box in centroid_dict.items():
        if id in redZone and id not in yellowZone:
            red_without_yellowZone.append(id)

    return (
     yellowZone, red_without_yellowZone, redGroups)


def Apply_ellipticBound(centroid_dict, img, calibration, red, green, yellow, final_redZone, coupleZone, couples, Single_radius, Couples_radius):
    RedColor = (0, 0, 255)
    GreenColor = (0, 255, 0)
    YellowColor = (0, 220, 255)
    BirdBorderColor = (255, 255, 255)
    BorderColor = (220, 220, 220)
    Transparency = 0.55
    e = birds_eye(img, calibration)
    overlay = e.img2bird()
    for idx, box in centroid_dict.items():
        center_bird = (
         box[0], box[1])
        if idx in green:
            cv2.circle(overlay, center_bird, Single_radius, GreenColor, -1)
        if idx in red and idx not in coupleZone:
            cv2.circle(overlay, center_bird, Single_radius, RedColor, -1)

    for p1, p2 in couples:
        x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
        centerGroup_bird = e.projection_on_bird((x, ymax))
        if p1 in yellow:
            if p2 in yellow:
                cv2.circle(overlay, centerGroup_bird, Couples_radius, YellowColor, -1)
        if p1 in final_redZone and p2 in final_redZone:
            cv2.circle(overlay, centerGroup_bird, Couples_radius, RedColor, -1)

    e.setBird(overlay)
    e.setImage(cv2.addWeighted(e.original, Transparency, e.bird2img(), 1 - Transparency, 0))
    overlay = e.image
    for idx, box in centroid_dict.items():
        birdseye_origin = (
         box[0], box[1])
        circle_points = e.points_projection_on_image(birdseye_origin, Single_radius)
        if idx not in coupleZone:
            for x, y in circle_points:
                cv2.circle(overlay, (x, y), 1, BorderColor, -1)

        ymin = box[5]
        ymax = box[7]
        origin = e.projection_on_image((box[0], box[1]))
        w = 3
        x = origin[0]
        top_left = (x - w, ymin)
        botton_right = (x + w, ymax)
        if idx in green:
            cv2.rectangle(overlay, top_left, botton_right, GreenColor, -1)
            cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)
        if idx in red and idx not in coupleZone:
            cv2.rectangle(overlay, top_left, botton_right, RedColor, -1)
            cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)

    for p1, p2 in couples:
        x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
        birdseye_origin = e.projection_on_bird((x, ymax))
        circle_points = e.points_projection_on_image(birdseye_origin, Couples_radius)
        for x, y in circle_points:
            cv2.circle(overlay, (x, y), 1, BorderColor, -1)

        origin = e.projection_on_image(birdseye_origin)
        w = 3
        x = origin[0]
        top_left = (x - w, ymin)
        botton_right = (x + w, ymax)
        if p1 in yellow:
            if p2 in yellow:
                cv2.rectangle(overlay, top_left, botton_right, YellowColor, -1)
                cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)
            if p1 in final_redZone and p2 in final_redZone:
                cv2.rectangle(overlay, top_left, botton_right, RedColor, -1)
                cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)

    e.setImage(overlay)
    return (
     e.image, e.bird)


def Apply_trackmap(centroid_dict, trackmap, colorPool, decay):
    trackmap = cv2.cvtColor(trackmap, cv2.COLOR_RGB2HSV)
    a = trackmap[:, :, 2]
    a[(a > 0)] -= decay
    trackmap[:, :, 2] = a
    for id, box in centroid_dict.items():
        center_bird = (
         box[0], box[1])
        color = colorPool[id]
        cv2.circle(trackmap, center_bird, 1, (color, 255, 255), -1)

    trackmap = cv2.cvtColor(trackmap, cv2.COLOR_HSV2RGB)
    return trackmap


def Apply_crowdMap(centroid_dict, img, _crowdMap):
    heat = np.zeros((img.shape[0], img.shape[1]), dtype=(np.int))
    for idx, box in centroid_dict.items():
        center_bird = (
         box[0], box[1])
        for i in range(1, 30, 3):
            new = np.zeros_like(heat)
            cv2.circle(new, center_bird, 2 * i, 10, -1)
            heat = cv2.add(heat, new)

    heat = cv2.blur(heat, (10, 10))
    _crowdMap = cv2.add(_crowdMap, heat)
    return (_crowdMap, heat)


def midPointCircleDraw(x_centre, y_centre, r):
    points = []
    x = r
    y = 0
    points.append((x + x_centre, y + y_centre))
    if r > 0:
        points.append((x + x_centre, -y + y_centre))
        points.append((y + x_centre, x + y_centre))
        points.append((-y + x_centre, x + y_centre))
    P = 1 - r
    while x > y:
        y += 1
        if P <= 0:
            P = P + 2 * y + 1
        else:
            x -= 1
            P = P + 2 * y - 2 * x + 1
        if x < y:
            break
        points.append((x + x_centre, y + y_centre))
        points.append((-x + x_centre, y + y_centre))
        points.append((x + x_centre, -y + y_centre))
        points.append((-x + x_centre, -y + y_centre))
        if x != y:
            points.append((y + x_centre, x + y_centre))
            points.append((-y + x_centre, x + y_centre))
            points.append((y + x_centre, -x + y_centre))
            points.append((-y + x_centre, -x + y_centre))

    return points


class birds_eye:

    def __init__(self, image, cordinates):
        self.original = image.copy()
        self.image = image
        self.c, self.r = image.shape[0:2]
        pst2 = np.float32(cordinates)
        pst1 = np.float32([[0, 0], [self.r, 0], [0, self.c], [self.r, self.c]])
        self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        self.transferB2I = cv2.getPerspectiveTransform(pst2, pst1)
        self.img2bird()

    def img2bird(self):
        self.bird = cv2.warpPerspective(self.image, self.transferI2B, (self.r, self.c))
        return self.bird

    def bird2img(self):
        self.image = cv2.warpPerspective(self.bird, self.transferB2I, (self.r, self.c))
        return self.image

    def setImage(self, img):
        self.image = img

    def setBird(self, bird):
        self.bird = bird

    def convrt2Bird(self, img):
        return cv2.warpPerspective(img, self.transferI2B, (self.r, self.c))

    def convrt2Image(self, bird):
        return cv2.warpPerspective(bird, self.transferB2I, (self.r, self.c))

    def projection_on_bird(self, p):
        M = self.transferI2B
        px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        return (int(px), int(py))

    def projection_on_image(self, p):
        M = self.transferB2I
        px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        return (int(px), int(py))

    def points_projection_on_image(self, center, radius):
        x, y = center
        points = midPointCircleDraw(x, y, radius)
        original = np.array([points], dtype=(np.float32))
        cvd = cv2.perspectiveTransform(original, self.transferB2I)
        return cvd[0]


def convertBack(x, y, w, h):
    xmin = int(round(x - w / 2))
    xmax = int(round(x + w / 2))
    ymin = int(round(y - h / 2))
    ymax = int(round(y + h / 2))
    return (xmin, ymin, xmax, ymax)


def Euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def Euclidean_distance_seprate(p1, p2):
    dX = np.sqrt((p1[0] - p2[0]) ** 2)
    dY = np.sqrt((p1[1] - p2[1]) ** 2)
    return (dX, dY)


def checkupArea(img, leftRange, downRange, point, color='g', Draw=False):
    hmax, wmax = img.shape[0:2]
    hmin = hmax - int(hmax * downRange)
    wmin = int(wmax * leftRange)
    if Draw:
        if color == 'r':
            color = (0, 0, 255)
        if color == 'g':
            color = (0, 255, 0)
        if color == 'b':
            color = (255, 0, 0)
        if color == 'k':
            color = (0, 0, 0)
        cv2.line(img, (0, hmin), (wmax, hmin), color, 1)
        cv2.line(img, (wmin, 0), (wmin, hmax), color, 1)
    x, y = point
    if x < wmin:
        if y > hmin:
            return True
    return False


def center_of_2box(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1[4:8]
    xmin2, ymin2, xmax2, ymax2 = box2[4:8]
    if xmin1 < xmin2:
        xmin = xmin1
    else:
        xmin = xmin2
    if ymin1 < ymin2:
        ymin = ymin1
    else:
        ymin = ymin2
    if xmax1 > xmax2:
        xmax = xmax1
    else:
        xmax = xmax2
    if ymax1 > ymax2:
        ymax = ymax1
    else:
        ymax = ymax2
    ymax -= 5
    box = (xmin, ymin, xmax, ymax)
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + w / 2
    y = ymax - h / 2
    return (
     int(x), int(y), xmin, ymin, xmax, ymax)