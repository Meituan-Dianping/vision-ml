import selectivesearch.selectivesearch
from all_config import *
from image_utils.image_utils import *
from keras.models import load_model
import os


img_rows, img_cols = IMG_ROW, IMG_COL
model = load_model(model_name)
model._make_predict_function()


def get_prediction(img_binary, rects):
    """
    predict roi of images, gives two scores for both class
    :param img_binary: image of binary type
    :param rects: rects of proposals
    :return: rects,score list of rects
    """
    rectangles = []
    score_list = []
    img_binary = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
    for rect in rects:
        roi = get_roi_image(img_binary, rect)
        roi = cv2.resize(roi, (img_rows, img_cols))
        score = model.predict(numpy.asarray([roi], numpy.float32).reshape(1, img_rows, img_cols, 1) / 255)
        if score[0][1] > 0.8:
            # print(score)
            rectangles.append(rect)
            score_list.append(score[0][1])
    return rectangles, score_list


def image_view(image):
    """
    view predicted image on window of cv2.show()
    :param image: image to show
    :return:
    """
    show_width = 320
    img = image
    if image.shape[1] > show_width:
        scale = show_width/image.shape[1]
        img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def get_proposals(img):
    """
    show generated proposals on input image
    :param img: input image
    :return:
    """
    img = cv2.imread(img)
    scale = 500 / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    gray, binary = get_binary_image(img)
    mg_lbl, regions = selectivesearch.selective_search(binary, SCALE, SIGMA, MIN_SIZE)
    regions = get_proposal(regions, img.shape)
    print("proposals:", len(regions))
    cv2.drawContours(binary, regions, -1, (255, 145, 30), 2)
    image_view(binary)


def model_predict(img_file, view):
    """
    :param img_file: image file path
    :param view: true if you want show the result
    :return:
    """
    res_obj = {
        "score": 0,
        "position": ""
    }
    prediction = "predict_image.png"
    img = cv2.imread(img_file) if view else cv2.imdecode(numpy.fromstring(img_file, numpy.uint8), 1)
    scale = 500 / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    gray, binary = get_binary_image(img)
    if get_gray_score(binary):
        mg_lbl, regions = selectivesearch.selective_search(binary, SCALE, SIGMA, MIN_SIZE)
        regions = get_proposal(regions, img.shape)
        rectangles, score_list = get_prediction(binary, regions)
        if len(score_list) > 0:
            score = round(max(score_list), 2)
            rect = rectangles[score_list.index(max(score_list))]
            position = get_pos(rect, scale)
            res_obj["score"] = float(score)
            res_obj["position"] = position
            cv2.drawContours(img, [rect], -1, (255, 145, 30), 2)
            if not os.path.exists(PREDICT_PATH):
                os.mkdir(PREDICT_PATH)
            cv2.imwrite(PREDICT_PATH+prediction, img)
    if view:
        print(res_obj)
        image_view(img)
    return res_obj


"""
:parameter image path
"""
if __name__ == "__main__":
    model_predict("image.png", view=True)
    # get_proposals("image.png")
