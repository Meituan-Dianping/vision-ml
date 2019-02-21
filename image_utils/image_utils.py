import time
import cv2
import numpy


def get_proposal(rects, img_shape):
    """
    get proposals with calculated shape
    :param rects: input rectangles
    :param img_shape: image shape
    :return: proposals of calculated shape
    """
    rectangles = []
    contour_area_thresh = img_shape[0] * img_shape[1] / 30
    for rect in rects:
        rect = rect["rect"]
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        rectangle_contour = numpy.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
        if 0 < cv2.contourArea(rectangle_contour) < contour_area_thresh and (max(w, h) / min(w, h)) < 1.5:
            rectangles.append(rectangle_contour)
    return rectangles


def merge_rectangle_contours(rectangle_contours):
    """
    merge input contours
    :param rectangle_contours: input rect_contours
    :return: merged contours
    """
    merged_contours = [rectangle_contours[0]]
    for rec in rectangle_contours[1:]:
        for i in range(len(merged_contours)):
            x_min = rec[0][0]
            y_min = rec[0][1]
            x_max = rec[2][0]
            y_max = rec[2][1]
            merged_x_min = merged_contours[i][0][0]
            merged_y_min = merged_contours[i][0][1]
            merged_x_max = merged_contours[i][2][0]
            merged_y_max = merged_contours[i][2][1]
            if x_min >= merged_x_min and y_min >= merged_y_min and x_max <= merged_x_max and y_max <= merged_y_max:
                break
            else:
                if i == len(merged_contours)-1:
                    merged_contours.append(rec)
    # print(len(rectangle_contours), len(merged_contours))
    return merged_contours


def get_roi_image(img, rectangle_contour):
    """
    get image of contour area
    :param img: input image
    :param rectangle_contour: input contour
    :return: image of contour area with same type
    """
    roi_image = img[rectangle_contour[0][1]:rectangle_contour[2][1],
                    rectangle_contour[0][0]:rectangle_contour[1][0]]
    return roi_image


def get_rectangle_proposal(binary, img_shape, min_size):
    """
    get proposals by edge feature calculation
    :param binary: input image of binary type
    :param img_shape: input image shape
    :param min_size: min_size of proposal
    :return: proposals
    """
    if len(binary.shape) > 2:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_contours = []
    contour_area_thresh = img_shape[0] * img_shape[1] / 30
    for counter in contours:
        x, y, w, h = cv2.boundingRect(counter)
        cnt = numpy.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        if min_size < cv2.contourArea(cnt) < contour_area_thresh and (max(w, h) / min(w, h)) < 1.5:
            rectangle_contours.append(cnt)
    rectangle_contours = sorted(rectangle_contours, key=cv2.contourArea, reverse=True)[:100]
    rectangle_contours = merge_rectangle_contours(rectangle_contours)
    return rectangle_contours


def get_pos(rec, scale):
    """
    get center position (x,y) reshaped by scale
    :param rec: rectangle
    :param scale: image resize scale
    :return: reshaped position(x,y)
    """
    x = int((rec[0][0]+rec[1][0])/2/scale)
    y = int((rec[1][1]+rec[2][1])/2/scale)
    return x, y


def get_label_pos(contour):
    """
    get label position (x,y) of input contour
    :param contour: input contour
    :return: label position(x,y)
    """
    center = get_pos(contour)
    x = int((int((center[0]+contour[2][0])/2)+contour[2][0])/2)
    y = int((int((center[1]+contour[2][1])/2)+contour[2][1])/2)
    return x, y


def get_gray_score(binary):
    """
    calculate binary image mean value of normalized array
    :param binary: input binary image
    :return: score between (0-1.0)
    """
    if len(binary.shape) > 2:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    binary = numpy.asarray(binary, dtype=numpy.float32)
    binary_copy = numpy.array(binary, dtype=numpy.float32)
    cv2.normalize(binary, binary_copy, 0, 1, cv2.NORM_MINMAX)
    gray_score = numpy.mean(binary_copy)
    return True if gray_score > 0.5 else False


def get_binary_image(image):
    """
    convert 3d image to 1d image
    :param image: input image
    :return: 1d image of gray type and binary type
    """
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    img_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return img_gray, img_binary


def calculate_time(f):
    """
    wrapper for function performance calculation
    """
    def wrapper(*args, **kwargs):
        t_start = time.time()
        ret = f(*args, **kwargs)
        t_end = time.time()
        print(f.__name__ + ":" + str(round((t_end - t_start) * 1000, 2)) + "ms")
        return ret
    return wrapper
