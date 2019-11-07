# -*- encoding: utf-8 -*-

"""
@File    : _tools.py.py
@Time    : 2019/10/30 9:24
"""
import builtins
import io
import time
import traceback
from io import StringIO, IOBase
from typing import Tuple, Callable

import numpy as np
import cv2 as cv


def print(*args, end="\n"):
    if end == "\n":
        list(map(lambda ele: builtins.print(ele, end=" "), args))
        builtins.print("[%d]" % traceback.extract_stack()[-2].lineno)
    else:
        list(map(lambda ele: builtins.print(ele, end=" "), args[:-1]))
        builtins.print(args[-1], end=end)


def timer(comment=None, output=print) -> Callable:
    """
    对修饰的函数进行计时
    :param comment: 输出计时时间前的文字，默认是函数名
    :param output: 指定输出的方式，默认使用 print
    :return:
    """

    def _timer(func):
        def wrapper(*args, **kw):
            begin_time = time.time()
            res = func(*args, **kw)
            comment_ = comment if comment and not hasattr(comment, "__call__") else func.__name__
            output("%s: %.6f s" % (comment_, time.time() - begin_time))
            return res

        wrapper.__name__ = func.__name__
        return wrapper

    if hasattr(comment, "__call__"):
        return _timer(comment)
    return _timer


class ShowMetaclass(type):
    """
    使类的 __str__ 方法为返回自己类名加所有不以 _ 开头的成员(不包括函数)的值
    """
    @staticmethod
    def _str(self: object):
        with StringIO() as buffer:
            buffer.write("%s(" % self.__class__.__name__)
            has_attr = False
            for attr in filter(lambda x: not x.startswith("_") and not hasattr(getattr(self, x), '__call__'), dir(self)):
                buffer.write("%s=%s, " % (attr, getattr(self, attr)))
                has_attr = True
            if has_attr:
                buffer.seek(buffer.tell() - 2)
            buffer.write(")")
            buffer.seek(0)
            return buffer.read()

    def __new__(mcs, name, bases, attrs, **kwargs):
        attrs['__str__'] = ShowMetaclass._str
        args = (name, bases, attrs)
        return type.__new__(mcs, *args, **kwargs)


class Rect:
    """
    描述一个矩阵的几何信息
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        self._x, self._y, self._w, self._h = x, y, w, h

    @property
    def area(self) -> int:
        return self._w * self._h

    @property
    def size(self) -> Tuple[int, int]:
        return self._w, self._h

    @property
    def top_left(self) -> Tuple[int, int]:
        return self._x, self._y

    @property
    def top_right(self) -> Tuple[int, int]:
        return self._x + self._w, self._y

    @property
    def bottom_left(self) -> Tuple[int, int]:
        return self._x, self._y + self._h

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return self._x + self._w, self._y + self._h

    @property
    def top_middle(self) -> Tuple[int, int]:
        return self._x + self._w // 2, self._y

    def left_middle(self) -> Tuple[int, int]:
        return self._x, self._y + self._h // 2

    @property
    def right_middle(self) -> Tuple[int, int]:
        return self._x + self._w, self._y + self._h // 2

    @property
    def bottom_middle(self) -> Tuple[int, int]:
        return self._x + self._w // 2, self._y + self._h

    @property
    def center(self) -> Tuple[int, int]:
        return self._x + self._w // 2, self._y + self._h // 2

    def __str__(self):
        return "Rect(%d,%d %dx%d)" % (self._x, self._y, self._w, self._h)


def image_regularization(image: np.ndarray) -> np.ndarray:
    """
    将图片裁剪至合适的大小
    :param image: 需要裁剪的图像
    :return: 如果需要裁剪，返回裁剪后的图像，否则返回原图像
    """
    width, height = image.shape[1] & (~3), image.shape[0]
    if width != image.shape[1]:
        image = cv.resize(image, (width, height))
    return image

    # width = image.shape[1] // 4 * 4
    # if width != image.shape[1]:
    #     image = image[:, 0:width]
    # return image


def capture_image(image: np.ndarray, rect: Rect) -> np.ndarray:
    """
    从图像中截取一部分
    :param image: 原图
    :param rect: 需要截取的图片区域
    :return: 截取到的图片
    """
    (x1, y1), (x2, y2) = rect.top_left, rect.bottom_right
    x1 = max(0, (x1 & ~3))  # 小于等于当前值的最大的 4 的倍数
    x2 = (x2 + 3) & (~3)  # 大于等于当前值的最小的 4 的倍数
    y1 = max(0, y1)
    return image[y1:y2, x1:x2].copy()
