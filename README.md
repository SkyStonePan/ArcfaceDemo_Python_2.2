# ArcSoft Face SDK Python 版 Demo

[ArcFace 2.2](https://ai.arcsoft.com.cn/product/arcface.html) 人脸识别 SDK Python 版本的 Demo。

## 目录

- [项目结构](#项目结构)
- [快速上手](#快速上手)
  - [运行效果说明](#运行效果说明)
  - [所有的参数介绍](#所有的参数介绍)
  - [推荐的运行方式](#推荐的运行方式)
- [相关问题](#相关问题)

## 项目结构

```text
lib/          # 存放 ArcFace SDK 库
  linux_64/
  windows_32/
  windows_64/
arcface/      # 用 Python 对 ArcFace SDK 封装的模块
  arcface.py                # ArcFace SDK 的 Python 接口
  arcsoft_face_func.py      # ArcFace SDK 里提供的函数
  arcsoft_face_struct.py    # ArcFace SDK 里有的结构体
  tools.py                  # 内包含一些辅助工具
module/       # 整个 Demo 依赖的关键模块
  face_process.py           # 调用 ArcFace，集成人脸识别、活体检测等异步操作
  image_source.py           # 提供视频帧
  text_renderer.py          # 渲染中文
profile.yml   # Demo 的配置文件
demo.py       # Demo 的主程序入口
```

## 快速上手

1. 安装 [Python 3.7](https://www.python.org/downloads)。
2. 安装模块 [opencv-python](https://pypi.org/project/opencv-python)、[freetype](https://pypi.org/project/freetype-py)、[PyYAML](https://pypi.org/project/PyYAML)。可通过 `python -m pip install opencv-python freetype-py PyYAML` 安装需要的全部依赖。
3. 从 [ArcSoft 官网](https://ai.arcsoft.com.cn/product/arcface.html)授权申请 SDK，下载相应版本的 SDK 并解压。
4. 将 `lib` 中的所有动态库拷贝到 `lib` 目录中对应的平台目录下。
5. 准备好人脸数据库，即一张或者多张包含人脸的图片（最好一张图片仅包含一张清晰的人脸）。将这些图片一起放在同一个目录下，目录的路径最好没有中文也没有空格。
6. 准备一个摄像头或者一个视频，图片也行。
7. 在 [profile.yml](profile.yml) 中配置你的 `APP ID` 和 `SDK KEY`。
8. 通过 `python demo.py --faces <人脸数据库路径> --source <图片或者视频路径>` 运行 Demo。
9. 按下 `Esc` 或 `q` 或 `Q` 来退出程序。程序是不会主动退出的，即使视频播放完成后也不会退出。

> 需要将上面的 <> 中的内容及 <> 自身替换成满足 <> 中说明的实际内容。  
> 图片（不包括 GIF）和视频的格式需要能被 OpenCV 支持的格式。

### 运行效果说明

当检测到人脸时会用一个红色的矩形框圈出人脸，如果后续识别到这个人脸在人脸数据库中，人脸框会由红变蓝。  
人脸框上面有五个信息，分别用 `,` 隔开。这五个信息分别是 `姓名`, `相似度`, `活体结果`, `性别`, `年龄`。一开始这五个信息都没有，所有会有四个 ','，在识别出对应的信息后，会陆续更新出来。

### 所有的参数介绍

| 参数 | 说明 |
| :--- | :--- |
| --faces | 指定包含人脸数据库人脸的文件或者文件夹路径。 |
|  | 如果是文件路径，则假设文件是图片，将采用图片中所有人脸作为人脸数据库。 |
|  | 如果是文件夹路径，将递归遍历文件夹下所有图片文件进行处理，遍历过程中忽略 '.' 开头的文件夹和文件。 |
|  | 对于图片中的人脸，如果只有一张人脸将以`文件名`命名，如果有多张人脸将以`文件名-数字编号` 命名。编号来自识别人脸的顺序，可以认为是没有规律的。 |
| --faces-data | 指定已经缓存好的人脸数据库文件路径。 |
| --source | 指定视频路径或者图片路径，如果没有这个参数将打开默认的摄像头。 |
| --single | 以 1:n 模式运行，默认 m:n。 |

### 推荐的运行方式

- 如果是第一次运行，先运行 `python demo.py --faces <人脸数据库文件路径> --faces-data <缓存人脸数据库文件的路径>` 来生成人脸数据库的缓存，加快下次启动的速度。  
- 生成缓存后可以使用 `python demo.py --faces-data <已缓存人脸数据库文件的路径> --source <图片或者视频路径>` 来运行。

> 直接运行 `python demo.py --faces <人脸数据库路径> --source <图片或者视频路径>` ，这会在提取所有人脸特征组成人脸数据库后直接启动。但因为提取人脸特征比较慢（约 130 秒每千张），如果人脸数据库较大，每次启动都需要漫长的等待。

## 相关问题

- 将图片传给 SDK 时，可以通过 `bytes(image)` 来将 np.ndarray 类型转为 bytes 类型，从而可以传给 SDK。这样做存在不足的地方是 `bytes(image)` 会对内存数据进行复制，而很多时候这个复制是不必要的。使用 `image.`[`ctypes.data`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes) 可以直接得到底层的数据地址，将这个地址传给 SDK 可以避免不必要的内存复制。这样做也存在不足，因为能这样做的前提是 image 底层数据所占的内存连续。如果底层数据所占的内存不连续，就会出现用 `cv.imshow("", image)` 显示正常，SDK 却不能像预期那样工作。一种常见的导致内存不连续的情况是在对图片使用 `image[:720, :1280]` 这种方式进行裁剪，通过这种方法裁剪得到的图片底层内存就是不连续的。可通过 `image.`[`flags.c_contiguous`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html) 来判断内存是否连续，通过 [`numpy.ascontiguousarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ascontiguousarray.html)`(image)` 来得到内部数据一定连续的图片。

## 相关链接

- [常见问题](https://ai.arcsoft.com.cn/manual/faqs.html)
- [错误码概览](https://ai.arcsoft.com.cn/manual/arcface_windows_apiV2.html#22%E9%94%99%E8%AF%AF%E7%A0%81%E6%A6%82%E8%A7%88)
- [SDK交流论坛](https://ai.arcsoft.com.cn/bbs)
- 虹助手微信(Hongzhushou88)
- QQ 交流群(659920696)
