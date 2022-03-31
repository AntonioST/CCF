import cv2
import numpy as np


def feature_detect_hcd(frame: np.ndarray):
    """cornerHarris

    https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345

    :param frame:
    :return:
    """
    dst = cv2.cornerHarris(frame.astype(np.uint8), 10, 3, 0.04)
    dst = cv2.dilate(dst, None)
    y, x = np.nonzero(dst > 0.01 * np.max(dst))
    return dict(x=x, y=y)


def feature_detect_gft(frame: np.ndarray):
    """goodFeaturesToTrack

    https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541

    :param frame:
    :return:
    """
    maxCorners = 100
    corners = cv2.goodFeaturesToTrack(frame.astype(np.uint8), maxCorners, 0.01, 15)

    x = []
    y = []
    for i in corners:
        _x, _y = i.ravel()
        x.append(_x)
        y.append(_y)

    return dict(x=x, y=y)


def feature_detect_sift(frame: np.ndarray):
    """SIFT

    https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html

    :param frame:
    :return:
    """
    n_features = 100
    sift = cv2.SIFT_create(n_features)
    kp, desp = sift.detectAndCompute(frame.astype(np.uint8), None)

    x = []
    y = []
    for p in kp:
        _x, _y = p.pt
        x.append(_x)
        y.append(_y)

    return dict(x=x, y=y)


def feature_detect_blob(frame: np.ndarray):
    """SimpleBlobDetector

    https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html

    :param frame:
    :return:
    """
    detector = cv2.SimpleBlobDetector_create()
    kp = detector.detect(frame.astype(np.uint8))

    x = []
    y = []
    for p in kp:
        _x, _y = p.pt
        x.append(_x)
        y.append(_y)

    return dict(x=x, y=y)


def feature_detect_orb(frame: np.ndarray):
    """ORB (Oriented FAST and Rotated BRIEF)

    https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
    https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

    :param frame:
    :return:
    """
    n_features = 100
    frame = frame.astype(np.uint8)
    detector = cv2.ORB_create(n_features)
    kp = detector.detect(frame, None)
    kp, desp = detector.compute(frame, kp)

    x = []
    y = []
    for p in kp:
        _x, _y = p.pt
        x.append(_x)
        y.append(_y)

    return dict(x=x, y=y)
