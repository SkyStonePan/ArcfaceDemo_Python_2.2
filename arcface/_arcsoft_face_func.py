"""
对 C++ SDK 中函数的映射
虽然是直接映射，但是调整了命名风格，使其符合 PEP 8 命名规范
[ArcSoft ArcFace SDK API 在线文档](https://ai.arcsoft.com.cn/manual/arcface_windows_apiV2.html#2%E6%8E%A5%E5%8F%A3)
"""

import platform
from pathlib import Path
from typing import Tuple

from ._arcsoft_face_struct import *


def load_dll() -> Tuple[CDLL, CDLL]:
    """
    加载 ArcFace SDK 动态库
    :return: 加载的两个动态库的元组
    """
    lib_path = Path(__file__).parents[1] / r"lib"
    bits = "32" if platform.architecture()[0] == "32bit" else "64"
    if platform.system() == "Windows":
        lib_path /= "windows_" + bits
        arcface_dll = CDLL(str(lib_path / "libarcsoft_face.dll"))
        arcface_engine_dll = CDLL(str(lib_path / "libarcsoft_face_engine.dll"))
    elif platform.system() == "Linux":
        assert bits == "64", "不支持 32 位的Linux"
        lib_path /= "linux_" + bits
        arcface_dll = CDLL(str(lib_path / "libarcsoft_face.so"))
        arcface_engine_dll = CDLL(str(lib_path / "libarcsoft_face_engine.so"))
    else:
        assert False, "不支持的平台。仅支持 32/64 "
    return arcface_dll, arcface_engine_dll


_arcface_dll, _arcface_engine_dll = load_dll()


# 获取激活文件信息
get_active_file_info = _arcface_engine_dll.ASFGetActiveFileInfo
get_active_file_info.restype = c_int32  # 成功返回 0
get_active_file_info.argtypes = (
    POINTER(ActiveFileInfo),  # active_file_info: 激活文件信息
)

# 用于在线激活 SDK
# 注：
#   1) 初次使用 SDK 时需要对 SDK 先进行激活，激活后无需重复调用
#   2) 调用此接口时必须为联网状态，激活成功后即可离线使用
online_activation = _arcface_engine_dll.ASFOnlineActivation
online_activation.restype = c_int32  # 成功返回 0 或 MERR_ASF_ALREADY_ACTIVATED
online_activation.argtypes = (
    c_char_p,  # app_id: 官网获取的 APP ID
    c_char_p,  # sdk_key: 官网获取的 SDK KEY
)

# 初始化引擎
init_engine = _arcface_engine_dll.ASFInitEngine
init_engine.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
init_engine.argtypes = (
    c_uint32,  # detect_mode: VIDEO 模式/IMAGE 模式 VIDEO 模式:处理连续帧的图像数据 IMAGE 模式:处理单张的图像数据
    c_int,  # detect_face_orient_priority: 人脸检测角度，推荐单一角度检测；IMAGE 模式下不 支持全角度（ASF_OP_0_HIGHER_EXT）检测
    c_int32,  # detect_face_scale_val: 识别的最小人脸比例（图片长边与人脸框长边的比值） VIDEO 模式取值范围[2,32]，推荐值为 16 IMAGE 模式取值范围[2,32]，推荐值为 30
    c_int32,  # detect_face_max_num: 最大需要检测的人脸个数，取值范围[1,50]
    c_int32,  # combined_mask: 需要启用的功能组合，可多选
    POINTER(c_void_p),  # h_engine: 引擎句柄
)

# 人脸检测
detect_faces = _arcface_engine_dll.ASFDetectFaces
detect_faces.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
detect_faces.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    c_int32,  # width: 图片宽度，为 4 的倍数
    c_int32,  # height: 图片高度，YUYV/I420/NV21/NV12 格式为 2 的倍数； BGR24/GRAY/DEPTH_U16 格式无限制
    c_int32,  # format: 颜色空间格式
    POINTER(c_uint8),  # img_data: 图片数据
    POINTER(MultiFaceInfo),  # detected_faces: 检测到的人脸信息
)

# 单人脸特征提取
extract_feature = _arcface_engine_dll.ASFFaceFeatureExtract
extract_feature.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
extract_feature.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    c_int32,  # width: 图片宽度，为 4 的倍数
    c_int32,  # height: 图片高度，YUYV/I420/NV21/NV12 格式为 2 的倍数； BGR24/GRAY 格式无限制
    c_int32,  # format: 颜色空间格式
    POINTER(c_uint8),  # img_data: 图片数据
    POINTER(SingleFaceInfo),  # face_info: 单张人脸信息
    POINTER(FaceFeature),  # feature: 人脸特征
)

# 人脸特征比对
compare_feature = _arcface_engine_dll.ASFFaceFeatureCompare
compare_feature.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
compare_feature.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(FaceFeature),  # feature1: 人脸特征值
    POINTER(FaceFeature),  # feature2: 人脸特征值
    POINTER(c_float),  # confidence_level: 比对结果，相似度
)

# 修改 RGB/IR 活体阈值，SDK 默认 RGB：0.75,IR：0.7
set_liveness_param = _arcface_engine_dll.ASFSetLivenessParam
set_liveness_param.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
set_liveness_param.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(LivenessThreshold),  # threshold: 活体置信度，推荐阈值 RGB:0.75, IR:0.7
)

# 人脸信息检测（年龄/性别/人脸 3D 角度），最多支持 4 张人脸信息检测，超过部分返 回未知（活体仅支持单张人脸检测，超出返回未知）,接口不支持 IR 图像检测。
process = _arcface_engine_dll.ASFProcess
process.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
process.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    c_int32,  # width: 图片宽度，为 4 的倍数
    c_int32,  # height: 图片高度，YUYV/I420/NV21/NV12 格式为 2 的倍数， BGR24 格式无限制
    c_int32,  # format: 颜色空间格式
    POINTER(c_uint8),  # img_data: 图片数据
    POINTER(MultiFaceInfo),  # detected_faces: 检测到的人脸信息
    c_int32,  # combined_mask: 检测的属性（
    # ASF_AGE
    # ASF_GENDER、
    # ASF_FACE3DANGLE
    # ASF_LIVENESS），支持多选 注：检测的属性须在引擎初始化接口的 combinedMask 参 数中启用
)

# IR 活体单人脸检测
process_ir = _arcface_engine_dll.ASFProcess_IR
process_ir.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
process_ir.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    c_int32,  # width: 图片宽度，为 4 的倍数
    c_int32,  # height: 图片高度，I420/NV21/NV12 格式为 2 的倍数， DEPTH_U16/GRAY 格式无限制
    c_int32,  # format: 颜色空间格式， 支持（I420/NV21/NV12/DEPTH_U16/GRAY）的检测
    POINTER(c_uint8),  # img_data: 图片数据
    POINTER(MultiFaceInfo),  # detected_faces: 人脸信息，用户根据待检测的功能选择需要使用的人脸。
    c_int32,  # combined_mask: 检测的属性（ASF_IR_LIVENESS） 注：检测的属性须在引擎初始化接口的 combinedMask 参 数中启用
)

# 获取年龄信息
get_age = _arcface_engine_dll.ASFGetAge
get_age.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
get_age.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(AgeInfo),  # age_info: 检测到的年龄信息
)

# 获取性别信息
get_gender = _arcface_engine_dll.ASFGetGender
get_gender.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
get_gender.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(GenderInfo),  # gender_info: 检测到的性别信息
)

# 获取 3D 角度信息
get_angle3d = _arcface_engine_dll.ASFGetFace3DAngle
get_angle3d.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
get_angle3d.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(Angle3D),  # p3dangle_info: 检测到脸部 3D 角度信息
)

# 获取 RGB 活体信息
get_liveness_score = _arcface_engine_dll.ASFGetLivenessScore
get_liveness_score.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
get_liveness_score.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(LivenessInfo),  # liveness_info: RGB 活体信息，详见 2.2.10 ASF_LivenessInfo
)

# 获取 IR 活体信息
get_liveness_score_ir = _arcface_engine_dll.ASFGetLivenessScore_IR
get_liveness_score_ir.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
get_liveness_score_ir.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
    POINTER(LivenessInfo),  # liveness_info: IR 活体信息，详见 2.2.10 ASF_LivenessInfo
)

# 获取版本信息
get_version = _arcface_engine_dll.ASFGetVersion
get_version.restype = c_int32  # 成功返回版本信息，失败返回 MNull。
get_version.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
)

# 销毁引擎
uninit_engine = _arcface_engine_dll.ASFUninitEngine
uninit_engine.restype = c_int32  # 成功返回 MOK，失败详见 3.2 错误码列表
uninit_engine.argtypes = (
    c_void_p,  # h_engine: 引擎句柄
)
