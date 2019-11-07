# -*- encoding: utf-8 -*-

"""
@File    : demo.py
@Time    : 2019/10/15 15:46
ArcSoft Face SDK 示例程序
"""
import argparse
import logging
import sys
import time
from typing import Dict, Generator

import cv2 as cv
import numpy as np
import yaml

from arcface import ArcFace, timer
from arcface import FaceInfo as ArcFaceInfo
from module.face_process import FaceProcess, FaceInfo
from module.image_source import LocalImage, ImageSource, LocalCamera
from module.text_renderer import put_text

_logger = logging.getLogger(__name__)


@timer(output=_logger.info)
def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    :return: 解析后的结果
    """
    parser = argparse.ArgumentParser(
        # usage="python %s --faces <人脸数据>" % os.path.basename(sys.argv[0]),
        description='ArcSoft Face SDK Demo'
    )
    parser.add_argument("--faces", metavar="人脸数据文件或者所在的目录")
    parser.add_argument("--faces-data", metavar="人脸数据库文件")
    parser.add_argument(
        "--source",
        help="视频数据来源，默认打开默认的摄像头。"
             "如果是图片文件路径，则使用图片"
             "如果是文件夹路径，则连续使用文件夹下的所有图片"
             "如果是视频路径，则使用视频文件"
    )
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    # 读取配置文件
    with open("profile.yml", "r", encoding="utf-8") as file:
        profile: Dict[str, str] = yaml.load(file, yaml.Loader)
        args.app_id = profile["app-id"].encode()
        args.sdk_key = profile["sdk-key"].encode()
    # video/case01001.mp4, video/多人008.mp4
    return args


def _frame_rate_statistics_generator() -> Generator[float, bool, None]:
    """
    统计视频帧率
    :return:
    """
    count = 0
    begin_time = time.time()
    break_ = False
    while not break_:
        if count != 100:
            fps = 0.0
        else:
            end_time = time.time()
            fps = count / (end_time - begin_time)
            count = 0
            begin_time = time.time()
        count += 1
        break_ = yield fps


def _draw_face_info(image: np.ndarray, face_info: FaceInfo) -> None:
    """
    将人脸的信息绘制到屏幕上
    :param face_info: 人脸信息
    :return: None
    """
    # 绘制人脸位置
    rect = face_info.rect
    color = (255, 0, 0) if face_info.name else (0, 0, 255)
    cv.rectangle(image, rect.top_left, rect.bottom_right, color, 2)
    # 绘制人的其它信息
    x, y = rect.top_middle
    put_text(image, "%s" % face_info, bottom_middle=(x, y - 2))
    # 绘制人脸 ID
    info = "%d" % face_info.arc_face_info.face_id
    x, y = rect.top_left
    put_text(image, info, left_top=(x + 2, y + 2))


def _show_image(image: np.ndarray) -> bool:
    """
    将图像在界面上显示出来
    :param image: 需要显示的图像
    :return: 如果需要退出就返回 True
    """
    # 显示到界面上
    cv.imshow("ArcFace Demo", image)
    # cv.imshow("arcface demo", image[::1, ::1])
    key = cv.waitKey(1)

    return key == ord('q') or key == ord('Q') or key == 27


@timer(output=_logger.info)
def _run_1_n(image_source: ImageSource, face_process: FaceProcess) -> None:
    """
    1:n 的整个处理的逻辑
    :image_source: 识别图像的源头
    :face_process: 用来对人脸信息进行提取
    :return: None
    """
    with ArcFace(ArcFace.VIDEO_MODE) as arcface:

        cur_face_info = None  # 当前的人脸

        frame_rate_statistics = _frame_rate_statistics_generator()

        while True:
            # 获取视频帧
            image = image_source.read()

            # 检测人脸
            faces_pos = arcface.detect_faces(image)

            if len(faces_pos) == 0:
                # 图片中没有人脸
                cur_face_info = None
            else:
                # 使用曼哈顿距离作为依据找出最靠近中心的人脸
                center_y, center_x = image.shape[:2]
                center_y, center_x = center_y // 2, center_x // 2
                center_face_index = -1
                min_center_distance = center_x + center_y + 4
                cur_face_index = -1
                for i, pos in enumerate(faces_pos):
                    if cur_face_info is not None and pos.face_id == cur_face_info.arc_face_info.face_id:
                        cur_face_index = i
                        break
                    x, y = pos.rect.center
                    if x + y < min_center_distance:
                        center_face_index = i
                        min_center_distance = x + y
                if cur_face_index != -1:
                    # 上一轮的人脸依然在，更新位置信息
                    cur_face_info.arc_face_info = faces_pos[cur_face_index]
                else:
                    # 上一轮的人脸不在了，选择当前所有人脸的最大人脸
                    cur_face_info = FaceInfo(faces_pos[center_face_index])

            if cur_face_info is not None:
                # 异步更新人脸的信息
                if cur_face_info.need_update():
                    face_process.async_update_face_info(image, cur_face_info)

                # 绘制人脸信息
                _draw_face_info(image, cur_face_info)

                # 绘制中心点
                # put_text(image, "x", bottom_middle=(center_x, center_y))

            # 显示到界面上
            if _show_image(image):
                break

            # 统计帧率
            fps = next(frame_rate_statistics)
            if fps:
                _logger.info("FPS: %.2f" % fps)

            # if all(map(lambda x: x.complete(), faces_info.values())):
            #     break


@timer(output=_logger.info)
def _run_m_n(image_source: ImageSource, face_process: FaceProcess) -> None:
    """
    m:n 的整个处理的逻辑
    :image_source: 识别图像的源头
    :face_process: 用来对人脸信息进行提取
    :return: None
    """
    with ArcFace(ArcFace.VIDEO_MODE) as arcface:

        faces_info: Dict[int, FaceInfo] = {}

        frame_rate_statistics = _frame_rate_statistics_generator()

        while True:
            # 获取视频帧
            image = image_source.read()

            # 检测人脸
            faces_pos: Dict[int, ArcFaceInfo] = {}
            for face_pos in arcface.detect_faces(image):
                faces_pos[face_pos.face_id] = face_pos

            # 删除过期 id, 添加新的 id
            cur_faces_id = faces_pos.keys()
            last_faces_id = faces_info.keys()
            for face_id in last_faces_id - cur_faces_id:
                faces_info[face_id].cancel()  # 如果有操作在进行，这将取消操作
                faces_info.pop(face_id)
            for face_id in cur_faces_id:
                if face_id in faces_info:
                    # 人脸已经存在，只需更新位置就好了
                    faces_info[face_id].arc_face_info = faces_pos[face_id]
                else:
                    faces_info[face_id] = FaceInfo(faces_pos[face_id])

            # 更新人脸的信息
            # for face_info in faces_info:
            #     face_process.async_update_face_info(image, face_info)
            opt_face_info = None
            for face_info in filter(lambda x: x.need_update(), faces_info.values()):
                if opt_face_info is None or opt_face_info.rect.size < face_info.rect.size:
                    opt_face_info = face_info

            if opt_face_info is not None:
                face_process.async_update_face_info(image, opt_face_info)

                # cv.imshow("temp", opt_face_info.image)
                # print(opt_face_info.image.shape)

            # 绘制人脸信息
            for face_info in faces_info.values():
                _draw_face_info(image, face_info)

            if _show_image(image):
                break

            # if all(map(lambda x: x.complete(), faces_info.values())):
            #     break

            # 统计帧率
            fps = next(frame_rate_statistics)
            if fps:
                _logger.info("FPS: %.2f" % fps)


@timer(output=_logger.info)
def main() -> None:
    args = _parse_args()

    ArcFace.APP_ID = args.app_id
    ArcFace.SDK_KEY = args.sdk_key

    if not args.faces and not args.faces_data:
        print("需要通过 --faces 指定包含人脸图片的文件或者目录")
        print("或者通过 --faces-data 指定已经生成好的人脸数据库")
        sys.exit(-1)

    face_process = FaceProcess()
    if args.faces and args.faces_data:
        with face_process:
            face_process.add_features(args.faces)
            face_process.dump_features(args.faces_data)
        return

    class AutoCloseOpenCVWindows:
        """
        用来在出现异常的情况下自动关闭用 OpenCV 显示的窗口
        """

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            cv.destroyAllWindows()

    with face_process, AutoCloseOpenCVWindows():

        if args.faces:
            face_process.add_features(args.faces)
        else:
            face_process.load_features(args.faces_data)

        if args.source:
            image_source = LocalImage(args.source)
        else:
            image_source = LocalCamera()

        run = _run_1_n if args.single else _run_m_n

        with image_source:
            run(image_source, face_process)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(message)s [%(threadName)s:%(name)s:%(lineno)s]",
        level=logging.INFO
    )

    main()
