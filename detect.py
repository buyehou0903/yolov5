# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""#运行 YOLOv5 目标检测命令行界面（CLI）文档
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources: #可以在不同的源上运行 YOLOv5 目标检测推断，例如图像、视频、目录、摄像头、流等
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam      #摄像头（--source 0）
                                                     img.jpg                         # image       #提供图像路径（img.jpg）
                                                     vid.mp4                         # video       #提供视频文件（vid.mp4）
                                                     screen                          # screenshot  #屏幕截图
                                                     path/                           # directory   #目录（path/）
                                                     list.txt                        # list of images  #图像列表文件（list.txt）
                                                     list.streams                    # list of streams #表示一个列表文件，其中包含视频流的路径
                                                     'path/*.jpg'                    # glob    #图像文件的路径，通配符 * 表示匹配指定目录下所有以 .jpg 结尾的图像文件
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube     #YouTube视频（https://youtu.be/LNwODJXcvt4）
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream #表示通过RTSP协议访问的视频流的URL

Usage - formats:       #指定要使用的 YOLOv5 模型的格式
    $ python detect.py --weights yolov5s.pt                 # PyTorch  #基于 PyTorch 的原生模型权重文件，可用于在 PyTorch 环境中加载和运行
                                 yolov5s.torchscript        # TorchScript #TorchScript是PyTorch 的脚本语言，这个文件包含了经过 TorchScript 转换的模型，使其能够在没有 PyTorch 环境的情况下运行
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn #ONNX 是一个开放的模型交换格式，该文件包含了经过 ONNX 转换的模型。它可以用于在 ONNX Runtime 或使用 OpenCV DNN 进行推断
                                 yolov5s_openvino_model     # OpenVINO #OpenVINO 是由 Intel 提供的工具和库，用于优化深度学习模型的推断。这个文件包含了经过 OpenVINO 转换的模型.
                                 yolov5s.engine             # TensorRT #TensorRT 是 NVIDIA 提供的深度学习推断加速库。这个文件是经过 TensorRT 优化和序列化的引擎文件。
                                 yolov5s.mlmodel            # CoreML (macOS-only) #CoreML 是苹果提供的用于在 iOS 和 macOS 上部署机器学习模型的框架。这个文件是经过 CoreML 转换的模型，适用于 macOS。
                                 yolov5s_saved_model        # TensorFlow SavedModel #TensorFlow SavedModel 是 TensorFlow 模型的标准序列化格式，适用于 TensorFlow 中加载和运行。
                                 yolov5s.pb                 # TensorFlow GraphDef #TensorFlow GraphDef 是 TensorFlow 1.x 版本中使用的模型序列化格式。
                                 yolov5s.tflite             # TensorFlow Lite #TensorFlow Lite 是用于移动和嵌入式设备的轻量级 TensorFlow 版本。这个文件是 TensorFlow Lite 推断所需的模型。
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU #TensorFlow Edge TPU 是为 Google Edge TPU 优化的 TensorFlow Lite 模型。
                                 yolov5s_paddle_model       # PaddlePaddle #PaddlePaddle 是百度提供的深度学习框架。这个文件包含经过 PaddlePaddle 转换的模型。
"""

import argparse # 解析命令行参数的库(argparse：它是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息)
import csv # Python 中用于处理 CSV 文件的标准库
import os # 与操作系统进行交互的文件库 包含文件路径操作与解析(os：它提供了多种操作系统的接口。通过os模块提供的操作系统接口，我们可以对操作系统里文件、终端、进程等进行操作)
import platform #提供了一个跨平台的 API，允许你获取关于当前运行 Python 代码的计算机平台信息。
import sys # sys模块包含了与python解释器和它的环境有关的函数(sys： 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分)
from pathlib import Path # Path能够更加方便得对字符串路径进行处理(pathlib：这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读)

import torch #pytorch 深度学习库(这是主要的Pytorch库。它提供了构建、训练和评估神经网络的工具)

#将当前项目添加到系统路径上，以使得项目中的模块可以调用。
#将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。
FILE = Path(__file__).resolve() # __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory #ROOT保存着当前项目的父目录
if str(ROOT) not in sys.path:   # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块，就执行下面语句add ROOT to PATH
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative #ROOT设置为相对路径

from ultralytics.utils.plotting import Annotator, colors, save_one_box  #ultralytics.utils.plotting：这个文件定义了Annotator类，可以在图像上绘制矩形框和标注信息。

from models.common import DetectMultiBackend  #models.common.py：这个文件定义了一些通用的函数和类，比如图像的处理、非极大值抑制等等。
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams       #utils.dataloaders.py：这个文件定义了两个类，LoadImages和LoadStreams，
                                                                                                                         它们可以加载图像或视频帧，并对它们进行一些预处理，以便进行物体检测或识别。
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh) #utils.general.py：  这个文件定义了一些常用的工具函数，
                                                                                                                    比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等。
from utils.torch_utils import select_device, smart_inference_mode  #utils.torch_utils.py：这个文件定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等等。


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL     #weights：训练的权重路径，可以使用自己训练的权重，也可以使用官网提供的权重。
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)  #source：测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头)，也可以是rtsp等视频流, 默认data/images
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path  #data：配置数据文件路径，包括image/label/classes等信息，训练自己的文件，需要作相应更改，可以不用管
        imgsz=(640, 640),  # inference size (height, width) #imgsz：预测时网络输入图片的尺寸，默认值为 [640]
        conf_thres=0.25,  # confidence threshold #conf-thres：置信度阈值，默认为 0.50
        iou_thres=0.45,  # NMS IOU threshold #iou-thres：非极大抑制时的 IoU 阈值，默认为 0.45
        max_det=1000,  # maximum detections per image #max-det：保留的最大检测框数量，每张图片中检测目标的个数最多为1000类
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu #device：使用的设备，可以是 cuda 设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
        view_img=False,  # show results #view-img：是否展示预测之后的图片/视频，默认False
        save_txt=False,  # save results to *.txt  #save-txt：是否将预测的框坐标以txt文件形式保存，默认False，使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
        save_csv=False,  # save results in CSV format #用于控制是否将检测结果保存为 CSV 文件，默认False
        save_conf=False,  # save confidences in --save-txt labels #save-conf：是否保存检测结果的置信度到 txt文件，默认为 False
        save_crop=False,  # save cropped prediction boxes #save-crop：是否保存裁剪预测框图片，默认为False，使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
        nosave=False,  # do not save images/videos #-nosave：不保存图片、视频，要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
        classes=None,  # filter by class: --class 0, or --class 0 2 3 #classes： 仅检测指定类别，默认为 None
        agnostic_nms=False,  # class-agnostic NMS #agnostic-nms： 是否使用类别不敏感的非极大抑制（即不考虑类别信息），默认为 False
        augment=False,  # augmented inference #augment：  是否使用数据增强进行推理，默认为 False
        visualize=False,  # visualize features #visualize： 是否可视化特征图，默认为 False
        update=False,  # update all models #update：  如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
        project=ROOT / 'runs/detect',  # save results to project/name #project：  结果保存的项目目录路径，默认为 'ROOT/runs/detect'
        name='exp',  # save results to project/name #name：  结果保存的子目录名称，默认为 'exp'
        exist_ok=False,  # existing project/name ok, do not increment #exist-ok：  是否覆盖已有结果，默认为 False
        line_thickness=3,  # bounding box thickness (pixels) #line-thickness：  画 bounding box 时的线条宽度，默认为 3
        hide_labels=False,  # hide labels #hide-labels：  是否隐藏标签信息，默认为 False
        hide_conf=False,  # hide confidences #hide-conf：  是否隐藏置信度信息，默认为 False
        half=False,  # use FP16 half-precision inference #half：  是否使用 FP16 半精度进行推理，默认为 False
        dnn=False,  # use OpenCV DNN for ONNX inference #dnn：  是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
        vid_stride=1,  # video frame-rate stride #vid_stride 参数指定了在处理视频帧时的步长（步长是指算法在处理视频帧时跳过的帧数）vid_stride=1 表示处理所有的视频帧，没有跳过任何帧。
                                                             如果将 vid_stride 设置为更大的值（例如，vid_stride=2），则算法将每隔两帧处理一次，以降低处理速度但减少计算量。
):  
    #初始化配置（用于处理输入来源，定义了一些布尔值区分输入是图片、视频、网络流还是摄像头。）
    source = str(source) #输入的路径变为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images  #是否保存图片和txt文件，如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #判断source是不是视频/图像文件路径 # Path()提取文件名。suffix：最后一个组件的文件扩展名。
                                                                     若source是"D://YOLOv5/data/1.jpg"， 则Path(source).suffix是".jpg"， Path(source).suffix[1:]是"jpg"
                                                                     而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # 判断source是否是链接# .lower()转化成小写 .upper()转化成大写 .title()首字符转化成大写，其余为小写,
                                                                                      .startswith('http://')返回True or Flase
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file) # 判断是source是否是摄像头# .isnumeric()是否是由数字组成，返回True or False
    screenshot = source.lower().startswith('screen') #用于确定 source 是否表示屏幕截图，
                                                    .startswith('screen'): 使用 startswith 方法检查字符串是否以 'screen' 开头。如果是，返回 True；否则返回 False。
    if is_url and is_file:
        source = check_file(source)  # download # 返回文件。如果source是一个指向图片/视频的链接,则下载输入数据source = check_file(source)

    # Directories 保存结果（用于创建保存输出结果的目录。创建一个新的文件夹exp（在runs文件夹下）来保存运行的结果）
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。
                                                                                        第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir # 根据前面生成的路径创建文件夹

    # Load model  加载模型（用于选择设备、初始化模型和检查图像大小）
                                        '''weights   指模型的权重路径
                                        device  指设备
                                        dnn  指是否使用OpenCV DNN
                                        data  指数据集配置文件的路径
                                        fp16  指是否使用半精度浮点数进行推理
                                        stride  指下采样率
                                        names   指模型预测的类别名称
                                        pt   是Pytorch模型对象'''
    device = select_device(device) # 获取设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件
    stride, names, pt = model.stride, model.names, model.pt #stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标，
                                                             names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...]
                                                             pt: 加载的是否是pytorch模型（也就是pt格式的文件）
    imgsz = check_img_size(imgsz, s=stride)  # check image size # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回

    # Dataloader 加载数据(根据输入的 source 参数来判断是否是通过 webcam 摄像头捕捉视频流)
    bs = 1  # batch_size #用于在模型推断阶段逐个处理样本，而不是批处理，bs = 1 表示每个批次只包含一个样本
    # 通过不同的输入源来设置不同的数据加载方式
    if webcam: ## 使用摄像头作为输入
        view_img = check_imshow(warn=True) # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载输入数据流'''source：输入数据源；image_size 图片识别前被放缩的大小；
                                                                                                    stride：识别时的步长， auto的作用可以看utils.augmentations.letterbox方法，
                                                                                                    它决定了是否需要将图片填充为正方形，如果auto=True则不需要'''
        bs = len(dataset)
    elif screenshot: # 直接从source文件下读取图片
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs  # 保存视频的路径# 前者是视频路径,后者是一个cv2.VideoWriter对象
#推理部分
    # Run inference 热身部分
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup # 使用空白图片（零矩阵）预先用GPU跑一遍预测流程，可以加速预测
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile()) #seen、windows和dt，分别表示已处理的图片数量、窗口列表和时间消耗列表
    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集中的图像路径、图像数组、原始图像、视频捕获对象、图像的尺寸
     with dt[0]:  # 记录数据加载的时间
        im = torch.from_numpy(im).to(model.device)  # 将图像数组转换为PyTorch张量，并移到指定设备上
        im = im.half() if model.fp16 else im.float()  # 如果模型使用FP16精度，将图像数据类型转换为半精度浮点数
        im /= 255  # 将像素值从0-255转换为0.0-1.0的范围
        if len(im.shape) == 3:
            im = im[None]  # 在需要的情况下扩展维度，添加批次维度

    # 推断
    with dt[1]:  # 记录推断的时间
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 根据条件创建可视化的路径
        pred = model(im, augment=augment, visualize=visualize)  # 使用模型进行推断，可以选择进行数据增强和可视化

    # NMS（非极大值抑制）
    with dt[2]:  # 记录NMS的时间
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 应用非极大值抑制

    # 第二阶段分类器（可选）
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # 定义CSV文件的路径
    csv_path = save_dir / 'predictions.csv'

    # 创建或追加到CSV文件
    def write_to_csv(image_name, prediction, confidence):
        data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}  # 构建CSV文件的数据行
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())  # 创建CSV写入对象
            if not csv_path.is_file():
                writer.writeheader()  # 如果文件不存在，写入CSV文件的头部
            writer.writerow(data)  # 写入数据行到CSV文件

      # 处理预测结果
for i, det in enumerate(pred):  # 遍历每张图像的预测结果
    seen += 1
    if webcam:  # 如果是处理摄像头输入（批处理大小 >= 1）
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
    else:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    p = Path(p)  # 转换为Path对象
    save_path = str(save_dir / p.name)  # 保存图像的路径，如 im.jpg
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 保存标签文件的路径，如 im.txt
    s += '%gx%g ' % im.shape[2:]  # 将图像尺寸添加到输出字符串中
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益（用于还原归一化坐标）
    imc = im0.copy() if save_crop else im0  # 用于保存裁剪预测框的图像
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 创建图像注释器对象
    if len(det):
        # 将预测框的坐标从图像尺寸转换为原始图像尺寸
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # 打印检测结果
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # 统计每个类别的检测数量
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 将类别和检测数量添加到输出字符串中

                # 写入检测结果
for *xyxy, conf, cls in reversed(det):  # 遍历每个检测框
    c = int(cls)  # 类别索引
    label = names[c] if hide_conf else f'{names[c]}'  # 获取类别名称，带有置信度信息
    confidence = float(conf)  # 获取检测置信度
    confidence_str = f'{confidence:.2f}'  # 将置信度格式化为字符串

    if save_csv:  # 保存到CSV文件
        write_to_csv(p.name, label, confidence_str)

    if save_txt:  # 保存到文本文件
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化坐标转换为相对于原始图像的xywh格式
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 根据是否保存置信度选择不同的格式
        with open(f'{txt_path}.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')  # 写入文本文件

    if save_img or save_crop or view_img:  # 添加边界框到图像
        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 根据选项确定是否显示标签
        annotator.box_label(xyxy, label, color=colors(c, True))  # 在图像上标注边界框及标签
    if save_crop:  # 保存裁剪的预测框
        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

# 显示检测结果
im0 = annotator.result()  # 获取带有标注的图像
if view_img:  # 如果需要显示图像
    if platform.system() == 'Linux' and p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 创建可调整大小的窗口（仅在Linux下）
        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小
    cv2.imshow(str(p), im0)  # 显示图像
    cv2.waitKey(1)  # 等待按键输入（1毫秒）

           # 保存结果（带有检测框的图像）
if save_img:  # 如果需要保存图像
    if dataset.mode == 'image':  # 单张图像模式
        cv2.imwrite(save_path, im0)  # 保存图像
    else:  # 视频或流模式
        if vid_path[i] != save_path:  # 新视频
            vid_path[i] = save_path
            if isinstance(vid_writer[i], cv2.VideoWriter):
                vid_writer[i].release()  # 释放之前的视频写入器
            if vid_cap:  # 视频模式
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # 流模式
                fps, w, h = 30, im0.shape[1], im0.shape[0]
            save_path = str(Path(save_path).with_suffix('.mp4'))  # 强制在结果视频上添加 *.mp4 后缀
            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入器
        vid_writer[i].write(im0)  # 写入图像

# 打印时间（仅推理）
LOGGER.info(f"{s}{'' if len(det) else '(无检测), '}{dt[1].dt * 1E3:.1f}ms")

# 打印结果
t = tuple(x.t / seen * 1E3 for x in dt)  # 计算每张图像的速度
LOGGER.info(f'速度: %.1fms 预处理, %.1fms 推理, %.1fms NMS 每张图像，图像形状 {(1, 3, *imgsz)}' % t)
if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"结果保存至 {colorstr('bold', save_dir)}{s}")
if update:
    strip_optimizer(weights[0])  # 更新模型（解决SourceChangeWarning问题）



# 解析命令行参数
def parse_opt():
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='模型路径或 Triton URL')
    #weights：模型权重文件的路径
    #nargs：参数值的个数。'+' 表示可以接受多个参数值，并以列表形式存储
    #type=str：参数值的类型，指定为字符串
    # 数据源参数
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='文件/目录/URL/glob/屏幕/0(摄像头)')
    #source：输入图像、视频或摄像头的路径或URL
    # 数据集参数
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='（可选）dataset.yaml 路径')
    #data：数据集的配置文件路径，用于加载类别标签等信息
    # 推理参数
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='推理尺寸 h,w')
    #imgsz：推理时输入图片的尺寸，默认值为 [640]
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    #conf-thres：置信度阈值，默认为 0.50（调整检测结果的严格度，在输出结果中，只有置信度大于该阈值的检测框才会被保留，小于该阈值的将被过滤掉）
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 阈值')
    #iou-thres：非极大抑制时的 IoU 阈值，默认为 0.45（IoU 阈值表示两个边界框之间的重叠度阈值，当两个边界框的 IoU 大于该阈值时，NMS 将保留置信度更高的那个边界框，而舍弃另一个，如果 IoU 小于该阈值，两个边界框都将被保留。
    较高的 IoU 阈值将导致更大的重叠度被允许，从而保留更多的边界框，但可能导致重复检测。较低的 IoU 阈值将限制边界框的重叠度，提高去重效果，但可能会导致一些物体漏检）
    parser.add_argument('--max-det', type=int, default=1000, help='每张图像的最大检测数')
    #max-det：每张图片最多检测出的物体数，默认为 1000
    # 设备参数
    parser.add_argument('--device', default='', help='cuda 设备，例如 0 或 0,1,2,3 或 cpu')
    #device：使用的设备，可以是 cuda 设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
    # 可视化参数
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    #view-img：是否在推理时显示结果，默认为 False
    #action='store_true' 表示当命令行中包含 --view-img 时，将该参数的值设置为 True。如果未指定 --view-img，则其值将保持默认的 False。
    parser.add_argument('--save-txt', action='store_true', help='保存结果为 *.txt 文件')
    #save-txt：是否保存检测结果到 TXT 文件，默认为 False
    parser.add_argument('--save-csv', action='store_true', help='以 CSV 格式保存结果')
    #save-conf：是否保存检测结果的置信度到 TXT 文件，默认为 False
    parser.add_argument('--save-conf', action='store_true', help='将置信度保存在 --save-txt 标签中')
    #save-conf：是否将置信度保存在 --save-txt 标签中
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪后的预测框')
    #save-crop：是否保存检测结果中的物体裁剪图像，默认为 False
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    #nosave：是否保存结果图像或视频，默认为 False

    # 过滤参数
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤：--classes 0，或 --classes 0 2 3')
    #classes：仅检测指定类别，默认为 None
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的 NMS')
    #agnostic-nms：是否使用类别不敏感的非极大抑制（即不考虑类别信息），默认为 False
    #NMS 的目标是去除具有重叠区域的检测框，以避免对同一个目标进行多次计数。agnostic-nms 这个参数控制了在执行 NMS 时是否考虑物体的类别信息
    #如果 agnostic-nms 为 True，则 NMS 不考虑物体的类别信息。它会把所有类别的检测框看作是一个整体，只保留最具代表性的框，而不管它们属于哪个类别。
    #如果 agnostic-nms 为 False，则 NMS 会按照物体的类别独立进行操作。对于每个类别，它会单独执行 NMS，确保在同一类别内执行抑制，而不同类别之间的框不受影响。
    # 推理增强参数
    parser.add_argument('--augment', action='store_true', help='增强推理')
    #augment：是否使用数据增强进行推理，默认为 False
    #augment 参数控制是否在推理过程中使用数据增强。数据增强是通过对输入图像进行一系列随机变换来生成多样性的训练样本，从而提高模型的泛化能力。
    在推理时使用数据增强可能会产生更多的变化，以更全面地测试模型在不同情境下的性能。
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    #visualize：是否可视化特征图，默认为 False

    # 模型更新参数
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    #update：是否更新所有模型，默认为 False（系统会检查模型文件是否存在更新，并在必要时下载或更新它们）

    # 保存结果参数
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='将结果保存到项目/名称')
    #project：结果保存的项目目录路径，默认为 'ROOT/runs/detect'
    parser.add_argument('--name', default='exp', help='将结果保存到项目/名称')
    #name：结果保存的子目录名称，默认为 'exp'
    parser.add_argument('--exist-ok', action='store_true', help='存在项目/名称时是否递增')
    #exist-ok：是否覆盖已有结果，默认为 False

    # 可视化参数
    parser.add_argument('--line-thickness', default=3, type=int, help='边界框厚度（像素）')
    #line-thickness：画 bounding box 时的线条宽度，默认为 3
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
    #hide-labels：是否隐藏标签信息，默认为 False
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    #hide-conf：是否隐藏置信度信息，默认为 False

    # 推理性能参数
    parser.add_argument('--half', action='store_true', help='使用 FP16 半精度推理')
    #half：是否使用 FP16 半精度进行推理，默认为 False（当 half 为 True 时，模型将使用 FP16 进行推理，从而减少计算和存储的精度，但可能会牺牲一些模型的准确性）
    parser.add_argument('--dnn', action='store_true', help='使用 OpenCV DNN 进行 ONNX 推理')
    #dnn：是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧率步长')
    #vid-stride：视频帧率步长，默认为1，定义了每隔多少帧进行一次检测

    # 参数解析
    opt = parser.parse_args() #Python 内置的 argparse 模块，该模块用于解析命令行参数。函数的返回值是一个包含所有解析参数的对象，可以通过调用对象的属性获取参数的值。

    # 推理尺寸扩展
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    # 打印参数
    print_args(vars(opt))

    return opt



def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop')) #检查程序所需的依赖项是否已安装
    run(**vars(opt)) # 将 opt 变量的属性和属性值作为关键字参数传递给 run() 函数


if __name__ == '__main__':
    opt = parse_opt() # 解析命令行参数并将其存储在 opt 变量中
    main(opt) #调用主函数，并将 opt 变量作为参数传递给它
