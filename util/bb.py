import cv2


def draw_bb(img, bb, color):
    int_bb = tuple(map(int, bb))
    pt1 = int_bb[:2]
    pt2 = int_bb[2:]
    bb_img = cv2.rectangle(img=img,
                           pt1=pt1,
                           pt2=pt2,
                           color=color,
                           thickness=1)

    return bb_img

