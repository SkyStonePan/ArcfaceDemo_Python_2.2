"""
对 C++ SDK 的结构体的映射
虽然是直接映射，但是调整了命名风格，使其符合 PEP 8 命名规范
[ArcSoft ArcFace SDK API 在线文档]
(https://ai.arcsoft.com.cn/manual/arcface_windows_apiV2.html#1%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)
"""

from ctypes import *

_uint8_pointer = POINTER(c_uint8)
_int32_pointer = POINTER(c_int32)


class Rect(Structure):
    """
    人脸矩形框
    """
    left: c_int32
    top: c_int32
    right: c_int32
    bottom: c_int32

    _fields_ = [
        ('left', c_int32),
        ('top', c_int32),
        ('right', c_int32),
        ('bottom', c_int32),
    ]


class Version(Structure):
    """
    SDK 版本信息
    """
    version: c_char_p  # 版本号
    build_date: c_char_p  # 构建日期
    copy_right: c_char_p  # 版权信息

    _fields_ = [
        ('version', c_char_p),
        ('build_date', c_char_p),
        ('copy_right', c_char_p),
    ]


class ActiveFileInfo(Structure):
    """
    激活文件信息
    """
    start_time: c_char_p  # SDK 开始时间
    end_time: c_char_p  # SDK 截止时间
    platform: c_char_p  # 平台版本
    sdk_type: c_char_p  # SDK类型
    app_id: c_char_p  # APP ID
    sdk_key: c_char_p  # SDK KEY
    sdk_version: c_char_p  # SDK 版本号
    file_version: c_char_p  # 激活文件版本号

    _fields_ = [
        ('start_time', c_char_p),
        ('end_time', c_char_p),
        ('platform', c_char_p),
        ('sdk_type', c_char_p),
        ('app_id', c_char_p),
        ('sdk_key', c_char_p),
        ('sdk_version', c_char_p),
        ('file_version', c_char_p),
    ]


class SingleFaceInfo(Structure):
    """
    单人脸信息
    """
    rect: Rect  # 人脸框
    orient: c_int32  # 人脸角度

    _fields_ = [
        ('rect', Rect),
        ('orient', c_int32),
    ]


class MultiFaceInfo(Structure):
    """
    多人脸信息
    """
    rects: Rect  # 人脸框数组
    orients: c_int32  # 人脸角度(orientations)数组
    size: c_int32  # 检测到的人脸个数
    id: c_int32  # 在 VIDEO 模式下有效，IMAGE模式下为空

    _fields_ = [
        ('rects', POINTER(Rect)),
        ('orients', _int32_pointer),
        ('size', c_int32),
        ('id', _int32_pointer),
    ]


class FaceFeature(Structure):
    """
    人脸特征
    """
    feature: _uint8_pointer  # 人脸特征
    size: c_int32  # 人脸特征长度

    _fields_ = [
        ('feature', _uint8_pointer),
        ('size', c_int32),
    ]


class AgeInfo(Structure):
    """
    年龄信息
    """
    ages: _int32_pointer  # 0:未知; >0:年龄
    size: c_int32  # 检测的人脸个数

    _fields_ = [
        ('ages', _int32_pointer),
        ('size', c_int32),
    ]


class GenderInfo(Structure):
    """
    性别信息
    """
    genders: _int32_pointer  # 0:男性; 1:女性; -1:未知
    size: c_int32  # 检测的人脸个数

    _fields_ = [
        ('genders', _int32_pointer),
        ('size', c_int32),
    ]


class Angle3D(Structure):
    """
    3D 角度信息
    """
    roll: c_float  # 横滚角
    yaw: c_float  # 偏航角
    pitch: c_float  # 俯仰角
    state: c_int32  # 0:正常; 非0:异常
    size: c_int32  # 检测的人脸个数

    _fields_ = [
        ('roll', c_float),
        ('yaw', c_float),
        ('pitch', c_float),
        ('state', c_int32),
        ('size', c_int32),
    ]


class LivenessThreshold(Structure):
    """
    活体置信度
    """
    bgr_threshold: c_float  # RGB活体置信度
    ir_threshold: c_float  # IR活体置信度

    _fields_ = [
        ('bgr_threshold', c_float),
        ('ir_threshold', c_float),
    ]


class LivenessInfo(Structure):
    """
    活体信息 
    """
    is_live: _int32_pointer  # 0:非真人； 1:真人； -1：不确定； -2:传入人脸数>1
    size: c_int32  # 检测的人脸个数

    _fields_ = [
        ('is_live', _int32_pointer),
        ('size', c_int32),
    ]
