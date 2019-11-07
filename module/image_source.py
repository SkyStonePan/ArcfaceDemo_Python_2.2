# -*- encoding: utf-8 -*-

"""
@File    : image_source.py
@Time    : 2019/10/15 16:00
"""
import logging
import os
from abc import ABCMeta
from abc import abstractmethod
from typing import Generator

import cv2 as cv
import time

import numpy as np

_logger = logging.getLogger(__name__)


def get_regular_file(path: str) -> Generator[str, None, None]:
    """
    得到指定路径下所有的普通文件（不包括 link 文件，不包括 . 开头的文件）
    不支持符号链接
    :param path: 如果是文件，则结果只包含文件本身。如果是目录，则结果包含目录下（包含子目录）所有的文件
    :return: 文件名的迭代器
    """
    if not os.path.exists(path):
        raise ValueError("不存在的路径 \"%s\"" % path)
    if os.path.islink(path):
        raise ValueError("不能是符号链接 \"%s\"" % path)
    if os.path.isfile(path):
        yield path
        return
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            # 排除 . 开头和符号链接的文件夹
            new_dirs = list(filter(lambda x: not x.startswith(".") and not os.path.islink(x), dirs))
            dirs.clear()
            dirs.extend(new_dirs)
            for file in filter(lambda x: not x.startswith(".") and not os.path.islink(x), files):
                yield os.path.join(root, file)
        return
    raise ValueError("既不是文件也不是目录 \"%s\"" % path)


def read_image(filename) -> np.ndarray:
    """
    通过 OpenCV 读取图片，与 OpenCV 不同的时支持中文路径
    :param filename: 图片文件名
    :return: 成功返回图片，失败返回空矩阵
    """
    if not os.path.exists(filename) or 5 * 1024 * 1024 < os.path.getsize(filename):
        return np.array([])
    image = cv.imdecode(np.fromfile(filename, dtype=np.uint8), cv.IMREAD_COLOR)
    return image if image is not None else np.array([])


class ImageSource(metaclass=ABCMeta):
    """
    用来向外界提供稳定输出的图片
    """

    @abstractmethod
    def read(self) -> np.ndarray:
        """
        读取一张图片数据
        :return:
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        释放资源
        :return:
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class LocalImage(ImageSource):
    """
    从本地图片（一张或者多张）或者视频中提取图片
    以最快 张/INTERVAL ms 提供一张图片
    """

    INTERVAL = 40.0 / 1000  # 40 ms

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise ValueError("路径不存在 \"%s\"" % path)
        if os.path.islink(path):
            raise ValueError("不能是链接\"%s\"" % path)
        if os.path.isdir(path):
            # 如果参数是目录，则只能是包含图片的目录
            self._image = LocalImage._image_gen(path)
        else:
            image = read_image(path)
            if image.size != 0:
                self._image = LocalImage._image_gen(path)
            else:
                self._image = LocalImage._image_gen_from_video(path)

        self._last_read_time = time.time() - LocalImage.INTERVAL - 1

    @staticmethod
    def _image_gen_from_video(path: str) -> Generator[np.ndarray, bool, None]:
        """
        读取本地视频提供稳定的图片输出，最后一张图片将重复输出
        :param path: 视频路径
        :return: 值为图片的迭代器
        """
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件 \"%s\"" % path)
        break_ = False

        _, last_image = cap.read()
        if last_image is None:
            raise ValueError("无效的视频文件 \"%s\"" % path)
        _, image = cap.read()
        while image is not None and not break_:
            # 这里返回上一帧是为了提前预知读取到视频尾，将最后一帧以拷贝的形式送出去
            break_ = yield last_image
            last_image = image
            _, image = cap.read()
        image = last_image
        cap.release()

        while not break_:
            break_ = yield image.copy()

    @staticmethod
    def _image_gen(path: str) -> Generator[np.ndarray, bool, None]:
        """
        为一张或者多张图片提供无限的图片输出
        :param path: 图片文件或者包含图片文件的目录
        :return: 值为图片的迭代器
        """
        last_image = None
        break_ = False
        for filename in get_regular_file(path):
            image = read_image(filename)
            if image.size != 0:
                break_ = yield image.copy()
                last_image = image
                if break_:
                    break
        image: np.ndarray = last_image
        if image is None:
            raise ValueError("没有找到图片文件 \"%s\"" % path)
        while not break_:
            break_ = yield image.copy()

    def read(self) -> np.ndarray:
        image = next(self._image)
        cur_time = time.time()
        passed_time = cur_time - self._last_read_time
        # print(passed_time, LocalImage.INTERVAL, passed_time < LocalImage.INTERVAL)
        if passed_time < LocalImage.INTERVAL:
            time.sleep(LocalImage.INTERVAL - passed_time)
            cur_time = time.time()
        self._last_read_time = cur_time
        return image

    def release(self) -> None:
        try:
            while self._image:
                self._image.send(True)
        except (StopIteration, TypeError):
            # TypeError 是因为有可能一次都没有 next
            pass


class LocalCamera(ImageSource):
    """
    打开本地默认摄像头
    """

    def __init__(self):
        _logger.info("正在打开默认摄像头...")
        self._cap = cv.VideoCapture(0)
        assert self._cap.isOpened(), "没有发现本地的摄像头"

    def read(self) -> np.ndarray:
        succeed, image = self._cap.read()
        if not succeed:
            _logger.warning("读取摄像头失败")
        return image

    def release(self) -> None:
        self._cap.release()
