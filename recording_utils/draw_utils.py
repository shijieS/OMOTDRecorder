#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the MIT license.
#   For a copy, see <https://opensource.org/licenses/MIT>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
import numpy as np
import cv2

def cut(d, small, great):
    if d < small:
        return small
    if d > great:
        return great
    return d

def getTextImg( text,
                textColor = (0, 0, 0),
                bgColor = (255, 0, 0)):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=4, thickness=4)[0]
    offsetX = text_width // 15
    offsetY = text_height // 5
    bg = np.zeros((text_height + 2*offsetY, text_width + 2*offsetX, 3), dtype=np.uint8)
    bg[:, :, :] = bgColor
    cv2.putText(bg, text, (offsetX, text_height+offsetY), cv2.FONT_HERSHEY_PLAIN, fontScale=4, thickness=4, color=textColor)
    return bg

def putPrettyText(frame, text, font_color, bg_color, box):
    text_img = getTextImg(text, font_color, bg_color)
    th, tw, _ = text_img.shape
    l, t, r, b = box
    w, h = r - l, b - t
    ih, iw, _ = frame.shape
    f = w / 1.5 / tw
    tw, th = int(f*tw+0.5), int(f*th+0.5)
    resized_text_img = cv2.resize(text_img, (tw, th))
    tt, tb, tl, tr = t-th, t, l, l+tw

    ttd, tt = cut(tt, 0, ih) - tt, cut(tt, 0, ih)
    tld, tl = cut(tl, 0, iw) - tl, cut(tl, 0, iw)
    tb = cut(tb, 0, ih) + ttd
    tr = cut(tr, 0, iw) + tld
    frame[tt:tb, tl:tr] = resized_text_img
    return frame

def putPrettyTextPos(frame, text, font_color, bg_color, pos, f):
    text_img = getTextImg(text, font_color, bg_color)
    th, tw, _ = text_img.shape
    ih, iw, _ = frame.shape
    f = iw*f/tw
    tw, th = int(f * tw + 0.5), int(f * th + 0.5)
    resized_text_img = cv2.resize(text_img, (tw, th))
    tt, tb, tl, tr = pos[1], pos[1]+th, pos[0], pos[0] + tw
    ttd, tt = cut(tt, 0, ih) - tt, cut(tt, 0, ih)
    tld, tl = cut(tl, 0, iw) - tl, cut(tl, 0, iw)
    tb = cut(tb, 0, ih) + ttd
    tr = cut(tr, 0, iw) + tld
    frame[tt:tb, tl:tr] = resized_text_img
    return frame


def cv_draw_one_box(frame,
                    box,
                    color,
                    content_color=None,
                    alpha=0.5,
                    text="",
                    font_color=None,
                    with_border=True,
                    border_color=None):
    """
    Draw a box on a frame
    """
    h, w, _ = frame.shape
    # draw box content
    if content_color is None:
        content_color = color

    (l, t, r, b) = tuple([int(cut(b, 0, m)) for b, m in zip(box, [w, h, w, h])])
    roi = frame[t:b, l:r]
    black_box = np.zeros_like(roi)
    black_box[:, :, :] = content_color
    cv2.addWeighted(roi, alpha, black_box, 1-alpha, 0, roi)
    # draw border
    if with_border:
        if border_color is None:
            border_color = color
        cv2.rectangle(frame, (l, t), (r, b), border_color, 1)
    # put text
    if font_color is None:
        font_color = color
    bg_color = [abs(255 - c) for c in font_color]
    if text is not None and text != "":
        frame = putPrettyText(frame, text, font_color, bg_color, (l, t, r, b))
    # textImg = getTextImg(text, font_color, [255 - for c in font_color])
    # cv2.putText(frame, text, (l, t), cv2.FONT_HERSHEY_PLAIN, 1, font_color)
    return frame

def cv_draw_mult_boxes(frame, boxes, colors=None):
    """
    Draw multiple boxes on one frame
    :param frame: the frame to be drawn
    :param boxes: all the boxes, whoes shape is [n, 4]
    :param color: current boxes' color
    :return:
    """
    boxes_len = len(boxes)
    if colors is None:
        colors = [get_random_color(i) for i in range(boxes_len)]
    for box, color in zip(boxes, colors):
        frame = cv_draw_one_box(frame, box, color)
    return frame


# def cv_draw_one_circle(frame, center, radius, color, alpha, border_color=None, with_border=True):
#     h, w, _ = frame.shape
#     box = int(center[0]-radius), int(center[1]-radius), int(center[0]+radius), int(center[1]+radius)
#     l, t, r, b = tuple([int(cut(b, 0, m)) for b, m in zip(box, [w, h, w, h])])
#     roi = frame[t:b, l:r]
#     black_box = np.zeros_like(roi)
#     cv2.circle(black_box, (int((l+r)/2), int((t+b)/2)),
#                int(min((b-t, r-l))), color, )

def cv_draw_8_points(frame, datas):
    points = [(int(datas[i*2]), int(datas[i*2+1])) for i in range(8)]
    pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    color = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
             (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
             (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
    for i in range(len(pairs)):
        cv2.line(frame, points[pairs[i][0]], points[pairs[i][1]], color[i], 2)
    return frame

def get_random_color(seed=None):
    """
    Get the random color.
    :param seed: if seed is not None, then seed the random value
    :return:
    """
    if seed is not None:
        np.random.seed(int(seed))
    return tuple([np.random.randint(0, 255) for i in range(3)])

def get_random_colors(num, is_seed=True):
    """
    Get a set of random color
    :param num: the number of color
    :param is_seed: is the random seeded
    :return: a list of colors, i.e. [(255, 0, 0), (0, 255, 0)]
    """
    if is_seed:
        colors = [get_random_color(i) for i in range(num)]
    else:
        colors = [get_random_color() for _ in range(num)]
    return colors

