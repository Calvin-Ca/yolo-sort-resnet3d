import cv2

def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    '''
    @param img: numpy.ndarray from cv2.imread, shape (h, w, 3), bgr
    @param boxes: list of box, box is a list of [x1, y1, x2, y2, score, label]
            if score and label is not provided, it will not be drawn
    @param color: bgr
    @param thickness: int
    '''
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if len(box) > 4:
            if len(box) == 5:
                label = box[4]
                img = cv2.putText(img, f"{label}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
            else:
                score = box[4]
                label = box[5]
                img = cv2.putText(img, f"{label}:{score:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    return img

