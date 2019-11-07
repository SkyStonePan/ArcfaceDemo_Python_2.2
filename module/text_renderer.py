# -*- encoding: utf-8 -*-

"""
@File    : text_renderer.py
@Time    : 2019/10/18 9:45
提供 put_text 方法用于将文字直接绘制到 numpy.ndarray 类型的图片上面
"""
import platform
from typing import Callable, Tuple

import cv2 as cv
import freetype as ft
import numpy as np


class TextRenderer:
    """
    将文本渲染成图片
    """

    def __init__(self, font_file, char_size: int = 48 * 64):
        self._face = ft.Face(font_file)
        self.set_char_size(char_size)

    def set_char_size(self, size) -> None:
        self._face.set_char_size(size)

    def render_text(self, text: str) -> np.ndarray:
        """
        渲染黑白的文字。代码来源：https://github.com/rougier/freetype-py/blob/master/examples/hello-world.py
        :param text: 需要渲染的文字
        :return: 渲染后得到图片
        """
        slot: ft.GlyphSlot = self._face.glyph

        # First pass to compute bbox
        width, height, baseline = 0, 0, 0
        previous = 0
        for i, c in enumerate(text):
            self._face.load_char(c)
            bitmap = slot.bitmap
            # 这里与原代码有出入，因为原代码有个 BUG
            # 在使用微软雅黑渲染 ",0", "f," 时会抛异常（官方给的字体 Vera.ttf 也会抛异常）
            height = max(height, bitmap.rows)
            baseline = max(baseline, max(0, -(slot.bitmap_top - bitmap.rows)))
            kerning = self._face.get_kerning(previous, c)
            width += (slot.advance.x >> 6) + (kerning.x >> 6)
            previous = c

        height += baseline
        image = np.zeros((height, width), dtype=np.ubyte)

        # Second pass for actual rendering
        x, y = 0, 0
        previous = 0
        for c in text:
            self._face.load_char(c)
            bitmap = slot.bitmap
            top = slot.bitmap_top
            # left = slot.bitmap_left
            w, h = bitmap.width, bitmap.rows
            y = height - baseline - top
            kerning = self._face.get_kerning(previous, c)
            x += (kerning.x >> 6)
            image[y:y + h, x:x + w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h, w)
            x += (slot.advance.x >> 6)
            previous = c

        return image


def _rect_intersection(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    计算两个矩阵的重叠部分
    使用左上角和右下角的 x, y 值来确定一个矩阵
    :param rect1:
    :param rect2:
    :return: 重叠部分的矩阵坐标
    """
    ax1, ay1, ax2, ay2 = rect1
    bx1, by1, bx2, by2 = rect2
    return max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)


def _get_put_text():
    font_path = r""
    if platform.system() == "Windows":
        font_path = r"C:\Windows\Fonts\msyh.ttc"
    elif platform.system() == "Linux":
        font_path = r""
    elif platform.system() == "Mac":
        font_path = r""

    assert font_path, "没有字体文件"
    font_size = 10
    text_renderer = TextRenderer(font_path, 3 * font_size * 4 * font_size)

    def _put_text(
            image: np.ndarray,
            text: str,
            left_top: Tuple[int, int] = None,
            bottom_middle: Tuple[int, int] = None
    ) -> None:
        """
        将文本绘制到图片上
        :param image: 绘制的目标
        :param text: 需要绘制的文本
        :param bottom_middle: 绘制后的文本所占空间的底边中部的坐标
        :return: None
        """
        # 渲染文本
        try:
            text_image = text_renderer.render_text(text)
        except ValueError:
            text_image = text_renderer.render_text("渲染出现异常")

        # 计算绘制的目标区域上的坐标(这里的坐标左上角包含，右下角超出)
        h, w = text_image.shape[:2]
        if left_top is not None:
            x1, y1 = left_top
        elif bottom_middle is not None:
            bmx, bmy = bottom_middle
            x1, y1 = bmx - w // 2, bmy - (h - 1)
        else:
            raise ValueError("必需要设置一个描点")
        x2, y2 = x1 + w, y1 + h

        # 对区域进行截取, 防止超出范围
        nx1, ny1, nx2, ny2 = _rect_intersection((0, 0, image.shape[1], image.shape[0]), (x1, y1, x2, y2))
        h, w = text_image.shape[:2]
        tx1, ty1, tx2, ty2 = nx1 - x1, ny1 - y1, w - (x2 - nx2), h - (y2 - ny2)

        # 绘制到目标上
        text_image = text_image[ty1:ty2, tx1:tx2]
        mask = text_image != 0
        image[ny1:ny2, nx1:nx2][mask] = np.stack([text_image, text_image, text_image], axis=-1)[mask]
        # image[ny1:ny2, nx1:nx2] += 10 * np.stack([text_image, text_image, text_image], axis=-1)

    return _put_text


put_text: Callable[[np.ndarray, str], None] = _get_put_text()


def main():
    image = np.zeros([400, 400, 3], dtype=np.uint8)

    # put_text(image, "你好 word!", (200, 200))
    # 下面的示例曾经在绘制过程出现过异常
    put_text(image, ",0.00", (200, 100))
    put_text(image, "f,0.00", (200, 200))

    image = cv.resize(image, (400, 400))

    cv.imshow("image", image)
    cv.moveWindow("image", 0, 0)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
