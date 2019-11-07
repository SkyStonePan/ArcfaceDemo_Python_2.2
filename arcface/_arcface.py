# -*- encoding: utf-8 -*-

"""
@File    : _arcface.py
@Time    : 2019/10/14 16:30
ArcSoft Face SDK 的封装
"""
import ctypes
import logging
import time
from enum import Enum
from functools import reduce
from typing import List, Optional

import cv2 as cv
import numpy as np

from . import _arcsoft_face_func as arcface_sdk
from . import _arcsoft_face_struct as arcface_class
from ._tools import Rect, ShowMetaclass

_logger = logging.getLogger(__name__)


class Gender(Enum):
    """
    人脸的性别
    """
    Female = 0
    Male = 1


class Angle3D(metaclass=ShowMetaclass):
    """
    人脸的角度信息，具体可见[接入指南](https://ai.arcsoft.com.cn/manual/v22/arcface_windows_guideV22.html)的“人脸三维角度检测”。
    """

    def __init__(self, roll: int, yaw: int, pitch: int):
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch

    def __str__(self):
        return "(%d,%d,%d)" % (self.roll, self.yaw, self.pitch)


class FaceInfo(metaclass=ShowMetaclass):
    """
    保存单张人脸信息
    """

    def __init__(self, rect: Rect, orient: int, face_id: int = None):
        self.rect = rect
        self.orient = orient
        if face_id is not None:
            self.face_id = face_id

    def __str__(self) -> str:
        values = self.rect, self.orient
        str_format = "%s-%d"
        if hasattr(self, 'face_id'):
            values = *values, self.face_id
            str_format += "-%d"
        return str_format % values

    def to_sdk_face_info(self):
        face_info = arcface_class.SingleFaceInfo()
        rect = self.rect
        face_info.rect.left, face_info.rect.top = rect.top_left
        face_info.rect.right, face_info.rect.bottom = rect.bottom_right
        face_info.orient = self.orient
        return face_info


class ArcFace:
    """
    对 Python 版本的 ArcSoft Face 的再封装
    """
    VIDEO_MODE: int = 0x00000000
    IMAGE_MODE: int = 0xFFFFFFFF

    # 功能组合
    NONE: int = 0x00000000
    FACE_DETECT: int = 0x00000001
    FACE_RECOGNITION: int = 0x00000004
    AGE: int = 0x00000008
    GENDER: int = 0x00000010
    ANGLE: int = 0x00000020
    LIVENESS: int = 0x00000080
    IR_LIVENESS: int = 0x00000400

    # 图片格式
    PAF_RGB24_B8G8R8: int = 0x201

    # 检测的角度
    OP_0_ONLY: int = 0x1
    OP_90_ONLY: int = 0x2
    OP_270_ONLY: int = 0x3
    OP_180_ONLY: int = 0x4
    OP_0_HIGHER_EXT: int = 0x5

    # 部分需要用的错误码
    ALREADY_ACTIVATED: int = 90114  # SDK 已激活
    FACE_FEATURE_LOW_CONFIDENCE_LEVEL: int = 81925  # 人脸特征检测结果置信度低

    APP_ID: bytes = b""
    SDK_KEY: bytes = b""

    def __init__(self, mode: int):

        self._engine = None
        ArcFace._activate()
        self._init_engine(mode)

    @staticmethod
    def _activate() -> None:
        """
        激活 SDK
        :return: None。失败抛出异常
        """
        assert ArcFace.APP_ID, "没有配置 APP ID"
        assert ArcFace.SDK_KEY, "没有配置 SDK KEY"
        ret = arcface_sdk.online_activation(ArcFace.APP_ID, ArcFace.SDK_KEY)
        if ret != 0 and ret != ArcFace.ALREADY_ACTIVATED:
            raise ArcFace._get_exception("Activate", "激活失败", ret)

    def _init_engine(self, mode: int) -> None:
        """
        初始化引擎
        :param mode: image 或者 video 模式
        :return: None。直接将引擎赋值给成员变量，失败抛出异常
        """
        assert (mode == ArcFace.VIDEO_MODE or mode == ArcFace.IMAGE_MODE)

        mask: int = reduce(lambda a, b: a | b, [
            ArcFace.FACE_DETECT,
            ArcFace.FACE_RECOGNITION,
            ArcFace.AGE,
            ArcFace.GENDER,
            ArcFace.ANGLE,
            ArcFace.LIVENESS,
        ])
        self._engine = arcface_sdk.c_void_p()
        ret = arcface_sdk.init_engine(
            mode,  # VIDEO 模式 / IMAGE 模式
            ArcFace.OP_0_ONLY,  # 人脸检测角度
            30,  # 识别的最小人脸比例
            10,  # 最大需要检测的人脸个数
            mask,  # 需要启用的功能组合
            ctypes.byref(self._engine)  # 引擎句柄
        )
        if ret != 0:
            raise ArcFace._get_exception("InitEngine", "初始化失败", ret)

        self.mode = mode

    def detect_faces(self, image: np.ndarray) -> List[FaceInfo]:
        """
        检测人脸，不会对传进来的图片数据进行复制
        :param image:
        :return:
        """
        assert image.flags.c_contiguous, "图片数据的所占的内存必需连续"

        # 检测人脸
        faces = arcface_class.MultiFaceInfo()

        height, width = image.shape[:2]
        ret = arcface_sdk.detect_faces(
            self._engine,
            width,
            height,
            ArcFace.PAF_RGB24_B8G8R8,
            ArcFace._image_to_uint8_pointer(image),
            ctypes.byref(faces)
        )
        if ret != 0:
            raise ArcFace._get_exception("DetectFace", "检测人脸失败", ret)

        # 将格式转为 Python 风格
        def convert(rect, orient, face_id):
            x, y = rect.left, rect.top
            w, h = rect.right - x + 1, rect.bottom - y + 1
            return FaceInfo(Rect(x, y, w, h), orient, face_id)

        if self.mode == ArcFace.VIDEO_MODE:
            def faces_id(index: int):
                return faces.id[index]
        else:
            def faces_id(_: int):
                return None

        faces_python = []
        for i in range(faces.size):
            faces_python.append(convert(faces.rects[i], faces.orients[i], faces_id(i)))
        return faces_python

    def extract_feature(self, image: np.ndarray, face_info: FaceInfo) -> bytes:
        """
        人脸特征提取，不会对传进来的图片数据进行复制
        :param image: 图片数据
        :param face_info: 人脸信息
        :return: 特征值，失败返回空的特征值
        """
        assert image.flags.c_contiguous, "图片数据的所占的内存必需连续"
        # 提取特征
        single_face_info = face_info.to_sdk_face_info()
        feature = arcface_class.FaceFeature()
        height, width = image.shape[:2]
        ret = arcface_sdk.extract_feature(
            self._engine,
            width,
            height,
            ArcFace.PAF_RGB24_B8G8R8,
            ArcFace._image_to_uint8_pointer(image),
            single_face_info,
            ctypes.byref(feature)
        )

        if ret == 0:
            # string_at 会拷贝底层的数据
            return ctypes.string_at(feature.feature, feature.size)

        if ret == ArcFace.FACE_FEATURE_LOW_CONFIDENCE_LEVEL:
            return b""

        raise ArcFace._get_exception("ExtractFeature", "提取人脸特征失败", ret)

    def compare_feature(self, fea1: bytes, fea2: bytes) -> float:
        """
        人脸特征对比
        :param fea1: 人脸 1 的特征值
        :param fea2: 人脸 2 的特征值
        :return: 相似度 [0.0, 1.0]
        """
        compare_threshold = ctypes.c_float()
        ret = arcface_sdk.compare_feature(
            self._engine,
            ArcFace._bytes_to_sdk_feature(fea1),
            ArcFace._bytes_to_sdk_feature(fea2),
            compare_threshold
        )
        if ret != 0:
            raise ArcFace._get_exception("CompareFeature", "对比人脸特征失败", ret)

        return compare_threshold.value

    def set_liveness_threshold(self, threshold_rgb: float, threshold_ir: float) -> None:
        """
        设置活体检测的阈值
        :param threshold_rgb: RGB 活体置信度
        :param threshold_ir: IR 活体置信度
        :return: None
        """
        threshold = arcface_class.LivenessThreshold()
        threshold.thresholdmodel_BGR = threshold_rgb
        threshold.thresholdmodel_IR = threshold_ir
        ret = arcface_sdk.set_liveness_param(
            self._engine,
            ctypes.byref(threshold)
        )

        if ret != 0:
            raise ArcFace._get_exception("SetLivenessThreshold", "设置活体对比阈值失败", ret)

    def process_face(
            self,
            image: np.ndarray,
            face_info: FaceInfo,
            mask: int
    ) -> bool:
        """
        对人脸数据进行处理，为后面的年龄，角度，活体作准备
        :param image: 包含人脸的图片
        :param face_info: 人脸在图片中的位置
        :param mask: 需要处理的功能集合
        :return: 成功返回 True，失败返回 False
        """
        assert image.flags.c_contiguous, "图片数据的所占的内存必需连续"
        rect = arcface_class.Rect()
        rect.left, rect.top = face_info.rect.top_left
        rect.right, rect.bottom = face_info.rect.bottom_right

        multi_face_info = arcface_class.MultiFaceInfo()
        multi_face_info.size = 1
        multi_face_info.rects = ctypes.pointer(rect)
        orient = ctypes.c_int(face_info.orient)
        multi_face_info.orients = ctypes.pointer(orient)

        height, width = image.shape[:2]
        ret = arcface_sdk.process(
            self._engine,
            width,
            height,
            ArcFace.PAF_RGB24_B8G8R8,
            ArcFace._image_to_uint8_pointer(image),
            ctypes.byref(multi_face_info),
            mask
        )

        if ret != 0 and ret != ArcFace.FACE_FEATURE_LOW_CONFIDENCE_LEVEL:
            raise ArcFace._get_exception("ProcessFace", "处理人脸信息失败", ret)

        return ret != ArcFace.FACE_FEATURE_LOW_CONFIDENCE_LEVEL

    def is_liveness(self) -> Optional[bool]:
        """
        判断人脸是否是活的。需要在调用 process_face 后调用。
        :return: 是活的返回 True，否则返回 False，失败返回 None
        """
        liveness_info = arcface_class.LivenessInfo()
        ret = arcface_sdk.get_liveness_score(
            self._engine,
            ctypes.byref(liveness_info)
        )
        if ret != 0:
            raise ArcFace._get_exception("IsLiveness", "检测活体失败", ret)

        assert liveness_info.size == 1, "人脸数(%d)不为 1" % liveness_info.size

        is_live = liveness_info.is_live[0]

        if is_live == -1:
            return None

        return is_live == 1

    def get_age(self) -> Optional[int]:
        """
        获得年龄。需要在调用 process_face 后调用。
        :return: 年龄，未知返回 None
        """
        age_info = arcface_class.AgeInfo()
        ret = arcface_sdk.get_age(
            self._engine,
            ctypes.byref(age_info)
        )

        if ret != 0:
            raise ArcFace._get_exception("GetAge", "获取年龄失败", ret)

        assert age_info.size == 1, "人脸数(%d)不为 1" % age_info.size

        age = age_info.ages[0]

        if age == 0:
            return None
        return age

    def get_gender(self) -> Optional[Gender]:
        """
        获得性别。需要在调用 process_face 后调用。
        :return: 性别，失败返回 None
        """
        gender_info = arcface_class.GenderInfo()
        ret = arcface_sdk.get_gender(
            self._engine,
            ctypes.byref(gender_info)
        )

        if ret != 0:
            raise ArcFace._get_exception("GetGender", "获取性别失败", ret)

        assert gender_info.size == 1, "人脸数(%d)不为 1" % gender_info.size

        gender = gender_info.genders[0]

        if gender_info.genders == -1:
            return None

        return Gender.Male if gender == 0 else Gender.Female

    def get_angle(self) -> Angle3D:
        """
        人脸 3D 角度。需要在调用 process_face 后调用。
        :return: 3D 角度
        """
        angle_info = arcface_class.Angle3D()
        ret = arcface_sdk.get_angle3d(
            self._engine,
            ctypes.byref(angle_info)
        )

        if ret != 0:
            raise ArcFace._get_exception("GetAngle", "获取人脸角度失败", ret)

        assert angle_info.size == 1, "人脸数(%d)不为 1" % angle_info.size

        if angle_info.status[0]:
            raise ArcFace._get_exception("GetAngle", "获取人脸角度失败", angle_info.status[0])

        roll = round(angle_info.roll[0])
        yaw = round(angle_info.yaw[0])
        pitch = round(angle_info.pitch[0])
        return Angle3D(roll, yaw, pitch)

    def release(self) -> None:
        """
        释放 SDK 资源
        :return: None
        """
        ret = arcface_sdk.uninit_engine(self._engine)
        if ret != 0:
            raise ArcFace._get_exception("ReleaseSDK", "释放 SDK 资源失败", ret)

        self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @staticmethod
    def _to_uint8_pointer(pointer):
        return ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8))

    @staticmethod
    def _to_void_pointer(pointer):
        return ctypes.cast(pointer, ctypes.c_void_p)

    @staticmethod
    def _image_to_uint8_pointer(image: np.ndarray):
        return ArcFace._to_uint8_pointer(image.ctypes.data)

    @staticmethod
    def _bytes_to_sdk_feature(bs: bytes):
        """
        将 bytes 包装成底层 SDK 的人脸特征值结构体，共用内存
        :param bs: bytes 类型的特征值
        :return: 底层 SDK 的人脸特征值结构体
        """
        feature = arcface_class.FaceFeature()
        feature.feature = ArcFace._to_uint8_pointer(bs)
        feature.featureSize = len(bs)
        return feature

    @staticmethod
    def _get_exception(name: str, describe: str, errno: int = 0):
        """
        自定义的异常
        :param name: 异常名
        :param describe: 异常细节描述
        :param errno: 错误码
        :return: 异常对象
        """
        exception_class = type(name + "Exception", (Exception,), {'describe': describe, 'errno': errno})
        if errno == 0:
            exception_class.__str__ = lambda self: describe
        else:
            exception_class.errno = errno
            exception_class.__str__ = lambda self: "%s: %d" % (self.describe, self.errno)
        return exception_class()


def main():
    ArcFace.APP_ID = b"Bbvyu5GeUE8eaBhyLsNcp49HW6tuPx3sqWog8i9S41Q7"
    ArcFace.SDK_KEY = b"EvvwdWPFj7XjL1siCQiSD2qGjG6kmH5uZpAkggSw4kMF"

    arcface = ArcFace(ArcFace.IMAGE_MODE)
    image = cv.imread(r"20190905160950.jpg")
    # cv.waitKey(1000)

    begin_time = time.time()
    faces = arcface.detect_faces(image)
    print("%6s" % (time.time() - begin_time))
    print("face number:", len(faces))

    for face in faces:
        cv.rectangle(image, face.rect.top_left, face.rect.bottom_right, (255, 0, 0,), 2)

    # cv.imshow("", image)
    # cv.waitKey(0)
    face = faces[0]
    (x1, y1), (x2, y2) = face.rect.top_left, face.rect.bottom_right
    cv.imshow("1", image[y1:y2, x1:x2])
    cv.waitKey(1)

    # 提取特征
    print("提取特征")
    arcface.extract_feature(image, faces[0])

    # 处理
    print("处理人脸")
    mask = reduce(lambda a, b: a | b, [
        ArcFace.LIVENESS,
        ArcFace.AGE,
        ArcFace.GENDER,
        ArcFace.ANGLE,
    ])
    arcface.process_face(image, faces[0], mask)
    print("is_liveness: %s" % arcface.is_liveness())
    print("age        : %s" % arcface.get_age())
    print("gender     : %s" % arcface.get_gender())
    print("angle      : %s" % (arcface.get_angle(),))

    # print(arcface.compare_feature(fea1, fea2))

    arcface.release()


if __name__ == "__main__":
    main()
