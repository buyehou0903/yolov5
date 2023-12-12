# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)#使用预训练模型进行训练
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch #从头开始训练（不使用预训练权重）

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3 #使用分布式训练（DDP）在多个 GPU 上进行训练

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

#导入的是用户安装的依赖包
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
 
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
 
# 2.路径解析
FILE = Path(__file__).resolve()
# 获取当前文件train.py的绝对路径，并赋值给FILE变量，__file__指train.py
ROOT = FILE.parents[0]  # YOLOv5 root directory
# 获取当前文件的父级目录并赋值给ROOT变量，也就是yolov5的绝对路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# 将当前文件的父级目录添加到Python解释器搜索模块的路径列表中，
# 这样可以通过import导入父级目录下的模块，
# sys.path存储了python模块的路径，是一个列表
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# 将当前文件的父目录转换为相对路径
 
# 3.导入模块
import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    # 创建一些变量存储解析到的参数
    callbacks.run('on_pretrain_routine_start') # 训练过程即将开始，用于执行一些训练预处理或初始化的操作
    # 回调函数：对象列表或对象，用于在不同的训练过程中进行一些数据的处理或展示
 
    # Directories 保存目录
    w = save_dir / 'weights'  # weights dir 权重保存的路径
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt' # 保存最后一次和最好的一次权重
 
    # Hyperparameters 超参数
    # hyp是字典形式或字符串形式
    # 若是字典形式，则保存了超参数的键值对，无需解析
    # 若是字符串，则保存了以yaml格式保存的文件路径
    if isinstance(hyp, str):  # 如果hyp是字符串，说明是路径格式，需要解析
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict 将yaml文件解析成字典形式，并加载到hpy变量中
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))  # 打印日志信息：超参数信息
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
 
    # Save run settings
    if not evolve: # 如果不使用进化超参数
        yaml_save(save_dir / 'hyp.yaml', hyp) # 将超参数信息以yaml形式保存
        yaml_save(save_dir / 'opt.yaml', vars(opt)) # 将参数信息转换为字典形式并以yaml形式保存
 
    # Loggers 日志记录 具体可以看一下Loggers这个类
    # 使用哪种形式进行记录 clearml形式或wandb形式
    # 记录结果就是result.png中的结果
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.clearml:
            data_dict = loggers.clearml.data_dict  # None if no ClearML dataset or filled in by ClearML
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size
 
        # Register actions
        # 将Loggers类中定义的非私有方法注册到回调函数中，以在训练过程中对每个回调函数进行调用。
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
 
    # Config  配置
    plots = not evolve and not opt.noplots  # create plots 训练过程中画线
    cuda = device.type != 'cpu'
    # 初始化随机种子，训练过程可复现
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 以下两行是进行分布式并行训练时，检查数据集的格式是否符合要求，将检查结果保存在data_dict变量中，data_dict是字典形式，内容是根据data文件夹下指定的yaml文件解析得到的
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    # 获得训练数据集路径和验证数据集路径
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes 获取类别数
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names 获取类别名
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check 检查类别数是否相等
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset 判断是否是coco数据集
 
    # Model
    check_suffix(weights, '.pt')  # check weights 检查权重文件是否以.pt结尾
    pretrained = weights.endswith('.pt') # 存储预训练权重
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally 检查有没有权重，没有就下载
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        # 加载权重，并将权重参数以字典形式存储到ckpt变量中
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create 创建新模型
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # exclude列表包含了需要排除的参数名称，这些参数需要重新训练，保证它们不受预训练模型的影响
        # 在训练过程中，有些参数是可以从预训练模型直接加载，而有些参数则需要重新训练
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 加载预训练模型的所有参数，并转换为float类型，以字典形式存储在csd变量中
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 对预训练模型和新模型中的权重参数取交集，并排除exclude中的内容
        # 交集中的参数在后续权重更新时会用到，exclude中的内容需要训练
        model.load_state_dict(csd, strict=False)  # load
        # 把所有相同的参数加载到新模型 预训练模型中某些参数对自己的模型是有帮助的
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report 记录从预训练模型中转移了多少参数，以及新模型中一共有多少参数
    else: # 如果没有预训练权重，则从头开始训练模型
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP 检查是否启用混合精度训练，启用则amp返回Ture
 
    # Freeze 控制冻结哪些层 创建了一个冻结参数列表freeze，用于决定需要冻结的参数层数
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 参数freeze是一个整数列表或单个整数，用于表示需要冻结的参数层数
    # 如果freeze是一个整数，则表示要冻结模型中的前freeze层参数
    # 如果freeze是一个整数列表，则表示要冻结模型中包含在这些层数中的所有参数
    for k, v in model.named_parameters(): # 遍历模型中所有参数，返回键和值
        v.requires_grad = True  # train all layers 更新梯度设为True，训练过程中更新参数
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze): # 如果x表示的参数所在的层在需要冻结的层中
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False # 则将对应的更新梯度设为False，训练过程中不更新参数
 
    # Image size 调整输入图片的尺寸
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 计算当前模型的最大stride，并取stride和32的最大值
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # 检查用户指定的输入图像的大小是否是步长gs的整数倍，调整输入模型的图片大小
 
    # Batch size bs=-1表示自动计算batch_size大小
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})
 
    # Optimizer 优化器
    nbs = 64  # nominal batch size 名义上bs 一批数据越多，梯度更新方向越准确
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 计算梯度累计的步数accumulate
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # 根据nbs、bs、accumulate调整权重衰减系数
    # 用于控制过拟合的权重衰减参数，使其在不同的批次大小和梯度累积步数下，具有相似的作用，避免影响模型的训练效果
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    # 创建一个优化器对象，以便在模型的训练阶段中通过梯度反向传播算法
    # 对模型参数进行更新，达到训练集优化的目的
    # smart_optimizer中定义了优化器类型以及要优化的参数变量
 
    # Scheduler 学习率调整策略
    if opt.cos_lr: # 余弦函数学习率调整
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:  # 线性函数学习率调整
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
 
    # EMA
    # 创建一个指数移动平均模型对象（ema），以用于在训练期间计算模型权重的指数滑动平均值，并在验证期间使用这个平均值对模型进行评估
    ema = ModelEMA(model) if RANK in {-1, 0} else None
 
    # Resume 恢复训练
    best_fitness, start_epoch = 0.0, 0
 # 这两个变量分别用来存储模型训练到目前为止在验证集上达到的最佳效果（best_fitness）以及模型开始训练的轮数（start_epoch）
    if pretrained: # 如果需要预训练
        if resume: # 检查是否需要从断点处恢复训练
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
    # 调用 smart_resume() 函数加载训练轮次信息，以及之前训练过程中存储的模型参数、优化器状态、指数滑动平均模型等状态信息，并将其恢复到当前状态中
        del ckpt, csd # 删除不需要的变量释放内存空间
 
    # DP mode  多gpu训练
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)
 
    # SyncBatchNorm 与多分布式训练相关
    # 若采用多分布式训练则将banchnorm层替换为SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
 
    # Trainloader 训练数据加载
    # 根据输入参数从指定路径加载训练数据，并生成一个数据加载器和一个数据集对象，用于模型训练
    train_loader, dataset = create_dataloader(train_path, #训练数据集路径
                                              imgsz, # 输入图像尺寸
                                              batch_size // WORLD_SIZE, #每批图像数
                                              gs, # global_size
                                              single_cls, # 是否单类别
                                              hyp=hyp, # 控制模型训练的超参数
                                              augment=True, # 是否图像增强
                                              cache=None if opt.cache == 'val' else opt.cache, # 数据是否需要缓存。当被设置为 'val' 时，表示训练和验证使用同一个数据集
                                              rect=opt.rect, # 是否用矩形训练方式
                                              rank=LOCAL_RANK, # 分布式训练，表示当前进程在节点中的排名
                                              workers=workers, #指定 DataLoader 使用的工作线程数
                                              image_weights=opt.image_weights, # 图像权重
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True #是否打乱数据集顺序)
    labels = np.concatenate(dataset.labels, 0)
    # 将数据集中所有标签数据（即类别信息）按照纵向顺序拼接成一个新的一维数组 labels
    # 用于后续在模型训练过程中进行分类准确率计算
    mlc = int(labels[:, 0].max())  # max label class 获得最大类别值
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    # 判断最大类别号是否小于类别数量，若大于等于则抛出异常
 
    # Process 0
    if RANK in {-1, 0}:
        # 验证数据集加载器
        val_loader = create_dataloader(val_path, # 路径
                                       imgsz,  # 图像尺寸
                                       batch_size // WORLD_SIZE * 2, # 每批图像数量
                                       gs,  # 全局步数
                                       single_cls, # 是否单类别
                                       hyp=hyp, # 超参数
                                       cache=None if noval else opt.cache, # 缓存文件的路径，用于加载之前处理过的图像
                                       rect=True, # 是否矩形验证
                                       rank=-1, # 分布式训练中进程的排名，不使用分布式训练
                                       workers=workers * 2, #数据加载器使用的进程数，用于并行加载数据
                                       pad=0.5, # 图像填充比例，用于处理图像大小不一致的情况
                                       prefix=colorstr('val: ')# 数据加载器的名称)[0] 
 
        if not resume:
            if plots:
                plot_labels(labels, names, save_dir) # 保存所有类别的标签数量，生成lables.png
 
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                # 在训练模型之前，自动计算并显示建议的锚框大小和比例
            model.half().float()  # pre-reduce anchor precision
            # 将模型的权重从浮点数格式转换为半精度浮点数格式
 
        callbacks.run('on_pretrain_routine_end')
        # 在每个预训练迭代结束后执行 on_pretrain_routine_end 回调函数
        # 用于对预训练模型进行必要的调整或者处理
 
    # DDP mode 多gpu训练
    if cuda and RANK != -1:
        model = smart_DDP(model)
 
    # Model attributes 模型属性
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # de_parallel(model) 函数用于将模型转换为可以在多个GPU上并行运行的形式
    # 函数会对模型的参数进行划分，使得不同的部分可以并行地计算
    # 代码通过取 model 的最后一个模块的 nl 属性，获得检测层的数量
    # 根据 nl 的值，代码更新了超参数 hyp 中的三个值：box，cls，obj，以及 label_smoothing
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    # 因为不同层的输出尺寸不同，为了保证超参数在不同层之间的一致性
    # 这里对 hyp 中的三个超参数进行了缩放，使得它们与层数和类别数量成正比
    
    # 以下4行代码的作用是为模型的训练做准备，为模型的属性赋值，并计算各个类别的权重
    model.nc = nc  # attach number of classes to model 数据集中模型数量赋值给模型nc属性
    model.hyp = hyp  # attach hyperparameters to model 加载超参数，训练使用
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 根据数据集的标签计算每个类别的权重，并将其赋值给模型的 class_weights 属性
    # 这个属性在训练过程中用于动态调整损失函数中各个类别的权重，从而更加关注重要的类别
    model.names = names 
    # 将数据集中的类别名称列表 names 赋值给模型的 names 属性，表示每个输出通道对应的类别名称
 
    # Start training  开始训练
    t0 = time.time() # 记录时间
    nb = len(train_loader)  # number of batches batch_size的长度，表示一共传入了几次batch
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # 计算预热期的迭代次数nw，训练过程开始时逐渐增加学习率，直到训练过程稳定
    # 预热期的目的是避免模型在初始的训练阶段过快地收敛到次优解，从而提高模型收敛到更优解的概率。
    # 预热期的迭代数量应该根据数据集的大小和超参数的具体设置进行调整，以取得最佳效果。
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 在训练过程中，last_opt_step 变量用于记录上一次进行梯度下降更新的迭代次数
    # 在进行优化器状态恢复时，可以使用该变量作为开始迭代次数，并继续训练。
    # 如果没有上一次的迭代记录，可以将该变量设置为 -1。
    # 这行代码的作用是初始化 last_opt_step 变量，并在训练过程中用于更新，以便在需要时进行优化器状态恢复
    maps = np.zeros(nc)  # mAP per class 创建了一个长度为 nc 的数组 maps，并将其所有元素初始化为 0
    # 初始化一个数组 maps，作为记录模型 mAP 值的容器。
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 初始化一个元组 results，作为记录模型性能指标的容器，方便在训练过程中进行记录和更新
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 初始化学习率调度器 scheduler 的 last_epoch 属性，以便在训练开始时自动计算出下一个 epoch 的学习率
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 创建一个 PyTorch 自动混合精度训练的梯度缩放器 scaler，并根据布尔变量 amp 是否启用自动混合精度进行配置
    # 在训练模型的过程中，我们可以使用scaler对象来缩放梯度，并执行正常的反向传播和优化更新操作
    # 由于采用了优化的计算方式，自动混合精度训练可以让模型在不影响性能的情况下，更快地完成训练
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 创建一个 EarlyStopping 对象 stopper，并将布尔变量 stop 的值初始化为 False
    # 在训练过程中，EarlyStopping 是一个常用的策略，用于避免模型训练过度拟合
    # 通过在每个 epoch 结束后计算验证集上的性能表现，如果模型在连续若干 epoch 上没有明显的改善，就可以终止训练，避免进一步过度拟合
    # stopper 对象是一个EarlyStopping类的实例，用于在模型训练过程中进行性能验证和终止训练操作
    # patience 参数指定了在连续多少个 epoch 上没有出现性能改善时，就会触发 EarlyStopping 策略
    # stop 变量表示是否需要终止训练
    compute_loss = ComputeLoss(model)  # init loss class
    # 创建一个 ComputeLoss 对象 compute_loss，用于计算模型在每个 epoch 上的损失函数值，
    # 并将模型 model 作为其输入参数进行初始化
    callbacks.run('on_train_start')
    # 打印日志信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # 开始一轮一轮训练epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train() # 启用了模型的训练模式
        # 在训练模式下，模型启用了一些特定的模块，比如 Dropout 和 Batch Normalization，用于防止模型的过拟合和稳定梯度更新
        # 在测试模式下，模型禁用了这些特定的模块，以便对新数据进行精确的预测
 
        # Update image weights (optional, single-GPU only)
        # 用于从数据集标签中计算并更新图像权重
        # 一批一批传数据时，难识别的目标希望它多传入几次，数据集的每张图片分配一个采样权重，难识别的权重大
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights 计算每个类别的权重，以便平衡类别不平衡的数据集
            # 数量多权重大，某一类的不精确度比较高，就会算出一个比较大的类别权重，增加它被采样的概率
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights 计算每个样本的图像权重
            # 算出来类别权重，但传给模型的是图片，而不是检测框，所以需要把类别权重转换为图片权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # 选择一些样本进行训练的过程，基于每个样本的图像权重，从数据集中随机选取相应数量的样本
            # 可以确保训练过程中每个类别都得到了足够的关注，从而减少类别不平衡的影响
 
        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
 
        mloss = torch.zeros(3, device=device)  # mean losses
        # 初始化mloss变量，用于缓存在训练过程中的损失值，并且将其存储在指定的设备上，以便更高效地计算和更新
        if RANK != -1: # 分布式训练
            train_loader.sampler.set_epoch(epoch) # 需要设置sampler的随机数种子，以保证每个epoch中样本的随机性
        pbar = enumerate(train_loader) # 遍历train_loader时获取进度条信息
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        # 日志记录
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        # 显示训练进度
        optimizer.zero_grad()
        # 清空梯度缓存，准备下一次新的梯度计算
        # 通常在每个batch的训练开始前，我们都会调用这个方法
        # 以清空上一个batch的梯度缓存，并开始当前batch的正向传播和相应的反向传播计算
        for i, (imgs, targets, paths, _) in pbar:  # 一批一批的取数据，每次取16 batch -------------------------------------------------------------
        # 遍历train_loader中的所有数据，并获得当前batch的输入、标签以及对应的路径信息，从而进行模型的训练
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 从第0轮开始，到目前为止一共训练了多少批数据，起到记录批次的功能
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            # 将imgs转换为PyTorch张量，并将其移动到GPU设备上进行加速计算
            # non_blocking=True表示数据转移过程是非阻塞的，这意味着转移操作将在后台异步进行，而不会影响后续的代码执行。这样可以提高数据转移和模型计算的效率
            # .float() / 255是将数据类型转换为float，并进行归一化操作，
            # 将像素值从[0, 255]范围缩放到[0, 1]范围内，以便更好地用于训练和优化模型
 
            # Warmup 热身训练，开始使用小学习率，慢慢升到设置的学习率
            if ni <= nw:  # 当前批次小于设置的wp所需批次时，不需要更新学习率
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                # 计算在梯度累积方式下，需要累积多少个batch的梯度
                for j, x in enumerate(optimizer.param_groups): # 循环更新优化器参数
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    # 计算当前参数组的学习率lr
                    # 需要计算当前学习率下降的幅度，可以使学习率在训练的初期快速增加，帮助模型更快地收敛；
                    # 然后，随着训练的进行，逐渐减小学习率，避免训练过程中震荡不收敛
                    # 如果当前为第一次学习率更新（即j为0），使用’warmup_bias_lr’这个超参数作为学习率的下降幅度，否则下降幅度为0
                    # 根据下降幅度，接着将其乘上当前参数组的初始学习率x[‘initial_lr’]
                    # 并使用一个类型为lf的函数对学习率进行动态调整。最终得到的结果就是当前参数组的学习率lr
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                    # 根据全局训练步数的值来动态调整当前参数组的动量
                    # 在全局训练步数ni小于阈值nw时，动量的值逐渐增加，以加速模型参数的更新过程；
                    # 当全局训练步数ni超过阈值nw时，动量的值逐渐减少，以减缓模型参数的更新速度，避免在minima处震荡。
 
            # Multi-scale 多尺度训练 数据增强过程中，对图像进行随机尺寸缩放
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
            # 训练过程随机化得到一个比例因子 用这个因子改变输入图片的尺度，起到多尺度训练的效果
                if sf != 1: #首先判断缩放比例sf是否等于1，如果不等于1则说明需要对图像进行缩放
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 计算出缩放后的新尺寸ns，将其对齐到gs的整数倍
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    # 使用nn.functional.interpolate函数对图像进行插值操作，缩放到新尺寸ns
                    # 最终得到的imgs就是缩放后的图像数据
 
            # Forward  前向传播
            with torch.cuda.amp.autocast(amp): # 开启混合精度训练
                pred = model(imgs)  # forward 将图片输入网络前向传播得到预测结果
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                # 利用模型预测信息和标注信息计算损失值和损失组件
                # 字典类型的loss_items，该字典记录了模型的损失值相对于不同组件的损失贡献度
                if RANK != -1:  # 分布式训练
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
                # loss将会乘以4倍以增大惩罚项的强度，进一步影响模型训练的收敛速度和结果
                # 某些任务通常要求模型在高频率信息的保留方面做得更好，同时往往要付出更大的计算代价
                # 模型的损失函数加入一些惩罚项，比如L2损失项，以此来约束预测结果与真实值之间的平滑程度。
            # Backward  反向传播
            scaler.scale(loss).backward()
            # 使用混合精度进行反向传播之前，我们需要把损失值通过scaler.scale()乘上比例因子，以确保数值的稳定性
 
            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
            # 当前训练步数（ni）和上一次优化步数（last_opt_step）之差大于等于指定的累积梯度步数（accumulate）时，执行优化器的step操作，进行参数更新
                scaler.unscale_(optimizer)  # unscale gradients
                # 执行了自动混合精度的反向传播操作之后，使梯度返回到原始的32位浮点型格式
                #（反向自动混合精度），以便进行进一步的梯度处理或优化器更新操作。
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                # 对模型的梯度进行裁剪，避免其过大而导致梯度爆炸的问题
                # 梯度爆炸指的是在反向传播过程中神经网络的梯度变得非常大，从而导致模型的训练变得不稳定
                # 这种情况可能会导致梯度消失或梯度耗散，从而影响模型的收敛速度和效果
                # 当梯度的范数（也称为“L2范数”）超过了指定的最大值max_norm时，
                # 裁剪操作将按比例缩小所有的梯度，以确保梯度的大小不超过max_norm
                scaler.step(optimizer)  # optimizer.step 更新优化器中的权重参数
                scaler.update()
                # 使用自动混合精度（AMP）功能更新scaler的状态
                # 使用scaler.step()函数更新参数后，我们需要通过scaler.update()函数更新缩放比例因子，以便在下一次batch中使用
                optimizer.zero_grad()
                # 将所有模型参数的梯度信息归零，以确保当前batch训练使用的是新的梯度信息
                if ema:
                    ema.update(model)
                # 在训练过程中，每次更新后，EMA算法会对模型参数进行平均值计算，
                # 并将平均值应用到训练好的模型参数中，以平滑模型参数的变化并防止过拟合
                last_opt_step = ni
                # 把last_opt_step更新为当前的训练步数，用于下一次的优化器step操作
 
            # Log 日志记录
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # 该公式通过计算之前所有的平均值和当前batch中的损失值来更新新的平均值
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots)
                if callbacks.stop_training:  # 训练结束
                    return
            # end batch ------------------------------------------------------------------------------------------------
 
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        # 获取优化器（optimizer）中每个参数组（param_groups）的学习率（lr）
        scheduler.step() # 一轮所有批次训练完后，根据之前的学习率更新策略更新学习率
 
        if RANK in {-1, 0}:
            # mAP  计算平均精度
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # 判断当前是否为最后一个epoch，以便在训练结束时进行相关的操作
            # 当前epoch数是否等于总的epoch数减1（epochs-1），或者调用了早停机制而停止了训练
            if not noval or final_epoch:  # Calculate mAP
            # 判断noval变量是否为False，如果是False则表示当前需要进行验证操作
            # 判断了final_epoch变量的值，如果final_epoch为True表示当前是最后一个epoch，则需要进行验证操作
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           half=amp,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)
 
            # Update best mAP 更新最优map
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # 根据给定的评价指标来计算当前训练的模型对于验证集的表现
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check 检查是否需要提前停止
            if fi > best_fitness: # 如果当前fi大于best_fitness
                best_fitness = fi #就将当前fi赋值给best_fitness
            log_vals = list(mloss) + list(results) + lr
            # 将当前epoch的各个指标（例如损失函数值、验证结果、学习率）记录下来，进行可视化
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
 
            # Save model 保存模型
            if (not nosave) or (final_epoch and not evolve):  # if save
            # 如果nosave为False，则表示需要保存模型参数
            # 如果final_epoch为True，且evolve为False，
            # 则表示当前是最后一个epoch，且不处于进化算法（evolution）中，此时也需要保存模型参数。
                                            ckpt = {
                                                'epoch': epoch,
                                                'best_fitness': best_fitness,
                                                'model': deepcopy(de_parallel(model)).half(),
                                                'ema': deepcopy(ema.ema).half(),
                                                'updates': ema.updates,
                                                'optimizer': optimizer.state_dict(),
                                                'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                                                'opt': vars(opt),
                                                'date': datetime.now().isoformat() #标记模型参数保存的时间}
                # 将当前训练的各项参数存储到一个字典ckpt中，以便在保存模型参数的时候使用
                # 将ckpt保存为一个文件，以便在训练结束后重新使用模型参数
                # Save last, best and delete 保存最后一次以及最好的一次
                torch.save(ckpt, last) # 使用torch.save函数将ckpt字典保存到文件中
                # last表示指定的路径
                if best_fitness == fi:
                    torch.save(ckpt, best) # 如果本轮拟合度最好，就保存best.pt
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                # 按照一定的周期自动保存训练过程中的模型参数，方便我们进行后续的模型调试和评估
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
 
        # EarlyStopping 是否提前停止训练
        if RANK != -1:  # if DDP training  多gpu训练
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop: # 如果满足停止训练条件，则跳出训练，防止过拟合
            break  # must break all DDP ranks
 
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                # 训练结束之后，将模型文件中的优化器信息删除，以便模型可以被更方便地加载、测试和部署
                if f is best: #用效果最好的权重在验证集上再跑一遍 并输出最终的指标以及每一类的各个指标
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(
                                        data_dict,
                                        batch_size=batch_size // WORLD_SIZE * 2,
                                        imgsz=imgsz,
                                        model=attempt_load(f, device).half(),
                                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        save_json=is_coco,
                                        verbose=True,
                                        plots=plots,
                                        callbacks=callbacks,
                                        compute_loss=compute_loss)  # val best model with plots
                    # 对模型进行验证，并获得模型在验证集上的性能表现，从而对模型进行调整和优化
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
 
        callbacks.run('on_train_end', last, best, plots, epoch, results)
 
    torch.cuda.empty_cache() # 清空PyTorch在当前显存中所占用的缓存空间
    return results # 返回结果

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')#weights: 模型预训练权重路径
    (1，若在命令行中使用"–weights" 参数，可指定预训练权重文件(路径)；
     2，若在命令行不使用"–weights" 参数，则预训练权重文件(路径)为自定义的default默认值；
     3，若既使用命令行"–weights" 参数，又自定义了default默认值，则模型使用的是命令行"–weights" 参数指定的预训练权重文件(路径)
     4，若不进行预训练，可使用"–weights" 参数指定一个空字符串：“”，或者将default默认值设置为空字符串：“”)
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')#cfg:模型结构文件路径，默认为空
    (1，在已经使用"–weights" 参数加载了预训练权重的情况下，可以不使用该参数，模型结构直接使用预训练权重中保存的模型结构；
     2，不使用"–weights" 参数使用"–cfg" 参数，表示模型从头开始训练，不进行预训练；
     3，“–weights” 参数和"–cfg" 参数必须要有一个，不然代码会报错)
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')#data:数据集配置文件路径
    (1，如果没有检查到数据集，代码会自动下载coco128数据集，也可以自己下载；
     2，把yolov5官方的数据集配置文件中的数据集下载部分内容给注释掉，代码则不会自动下载)
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')#hyp:训练超参数配置文件路径
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')#epochs:表示训练整个训练集的次数
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')#batch-size:训练批量大小,训练批量大小表示每个 mini-batch 中的样本数，
                                                                                                                     #batch-size设置为n表示一次性从训练集中获取n张图片送入模型进行训练；
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')#imgsz:模型训练和验证时输入图片的尺寸
    parser.add_argument('--rect', action='store_true', help='rectangular training')#rect:矩形训练，默认关闭(矩形训练过程中会对输入的矩形图片进行预处理，通过保持原图高宽比进行resize后，
                                                                                   对resize后的图片进行填充，填充到32的最小整数倍，然后进行矩形训练，减少训练时间。)
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')#resume:断点续训，默认关闭(
    1，断点续训就是从上一个训练任务中断的地方继续训练，直至训练完成；
    2，当模型按指定的epoch训练完成后，则无法进行断点续训；
    3，需要搭配"–weights" 参数使用，指定训练中断保存的最后一次模型权重文件
    nargs 参数表示接受的命令行参数个数。对于 '?'，它表示接受零个或一个参数。在这个特定的情况下，nargs='?' 的作用是指定 --resume 这个命令行参数可以有零个或一个值)
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')#nosave:只保留最后一次训练的权重，默认关闭
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')#noval:只对最后一次训练进行验证，默认关闭
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')#noautoanchor:关闭自动计算锚框功能，默认关闭(
                                                                                         yolov5采用的是kmeans聚类算法来计算anchor box的大小和比例，最终自动计算出一组最合适训练的锚框。)
    parser.add_argument('--noplots', action='store_true', help='save no plot files')#noplots:不保存可视化文件
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')#evolve:使用超参数优化算法进行自动调参，默认关闭(
    1，yolov5采用遗传算法对超参数进行优化，寻找一组最优的训练超参数；
    2，开启后传入参数n，训练每迭代n次进行一次超参数进化；
    3，开启后不传入参数，则默认为const=300)
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')#bucket:从谷歌云盘下载或上传数据(
    1，该参数用于指定 gsutil bucket 的名称，其中 gsutil 是 Google 提供的一个命令行工具，用于访问 Google Cloud Storage（GCS）服务；
    2，GCS 是 Google 提供的一种对象存储服务，在训练模型时，如果需要使用 GCS 中的数据集，就需要指定 bucket 的名称)
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')#cache:缓存数据集，默认关闭(
    1，缓存数据集图片到内存中，训练时模型直接从内存中读取，加快数据加载和训练速度
    2，若"–cache"参数指定值，可以指定的；值：ram/disk；
    3，若"–cache"参数不指定值，则默认为const='ram')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')#image-weights:对数据集图片进行加权训练，默认关闭(需要搭配"–rect"参数一起使用)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#device:选择训练使用的设备处理器，CPU或者GPU，默认为空(
    1，默认为空时，代码会进行自动选择，若检查到设备有GPU则优先使用GPU进行训练，若没有则使用CPU进行训练；
    2，使用GPU训练时，0,1,2,3分别表示第1，2，3，4张GPU)
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')#multi-scale:多尺度训练，默认关闭(开启多尺度训练，训练过程中每次输入图片会放大或缩小50%)
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')#single-cls:单类别训练，默认关闭
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')#optimizer:选择训练使用的优化器，默认使用SGD
                                                                                           (choices=[‘SGD’, ‘Adam’, ‘AdamW’]表示只能选择’SGD’, ‘Adam’, 'AdamW’这三种优化器，也可以添加自定义的优化器)
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')#sync-bn:使用SyncBatchNorm(同步批量归一化)，只有在使用DDP模式(分布式训练)时有效，默认关闭
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')#workers:设置Dataloader使用的最大numworkers，默认设置为8
    (1，Dataloader中numworkers表示加载处理数据使用的线程数，使用多线程加载数据时，每个线程会负责加载和处理一批数据，数据加载处理完成后，会送入相应的队列中，最后主线程会从队列中读取数据，并送入GPU中进行模型计算；
     2，numworkers为0表示不使用多线程，仅使用主线程进行数据加载和处理
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')#project:设置每次训练结果保存的主路径名称
                                                                                              (你每次训练会生成一个单独的子文件夹，主路径就是存放你这些单独子文件夹的地方，可以自己命名)
    parser.add_argument('--name', default='exp', help='save to project/name')#name:设置每次训练结果保存的子路径名称
                                                                (子路径是上面在’–project’中提到的每次训练生成的单独的子文件夹，可以自己命名,你每次训练生成的模型权重文件、可视化结果以及其它结果文件保存的地方)
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')#exist-ok:是否覆盖同名的训练结果保存路径，默认关闭，表示不覆盖
    (1，不使用’–exist-ok’参数时，如果’–name’指定的名称不变，比如’exp’，每次训练会按顺序新建文件夹，例如exp1、exp2、exp3、… 、expn；
    2，使用’–exist-ok’参数时，如果’–name’指定的名称不变，比如’exp’，每次训练则不会新建文件夹，训练结果会覆盖原先文件夹中保存的所有结果)
    parser.add_argument('--quad', action='store_true', help='quad dataloader')#quad:是否使用quad dataloader，默认关闭
                                                     (quad dataloader 是一种数据加载器，它可以并行地从磁盘读取和处理多个图像，并将它们打包成四张图像，从而减少了数据读取和预处理的时间，并提高了数据加载的效率)
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')#cos-lr:训练学习率衰减策略使用余弦退火策略，默认关闭
                                                                                    (余弦退火策略在训练初期加快学习速度，训练后期减小学习率，从而更好地学习数据的分布，避免模型陷入局部最优。)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')#label-smoothing:训练使用标签平滑策略，防止过拟合
    (1，默认为0.0，即标签平滑策略使用的epsilon为0.0；
     2，将标签平滑策略使用的epsilon设置为0.1：表示在每个标签的真实概率上添加一个 epsilon=0.1的噪声，从而使模型对标签的波动更加鲁棒；
     3，–label-smoothing 参数的值应该根据具体的数据集和模型来调整，以达到最优的训练效果)
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')#patience:训练使用EarlyStopping策略，防止过拟合
                                               (‘–patience’参数指定为整数n时，表示模型在训练时，若连续n个epoch验证精度都没有提升，则认为训练已经过拟合，停止训练。’–patience’可根据具体的模型和数据集进行调整)
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')#freeze:训练使用冻结训练策略，默认关闭
    (1，冻结训练是指在训练过程中冻结模型中的某些层，冻结的层不进行权重参数更新；
     2，指定’0’或’-1’，不冻结任何层，更新所有层的权重参数
     3，指定n，冻结前n(0<n<=10)层，即只更新前n层的权重参数)
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')#save-period:每训练n个epoch保存一次训练权重，默认关闭
    (1，n>0，每训练n个epoch保存一次训练权重；
     2，n<=0，关闭save-period，只保存best和last权重。)
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')#seed:设置训练使用的全局随机种子(随机种子可以保证每次生成的结果都一致，从而有利于代码的可复现性)
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')#local_rank:是否使用分布式训练，默认关闭

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')#entity：用于指定模型实体的参数
    (模型实体可以是一个实体名称或实体 ID，通常用于在实体存储库中管理模型的版本控制和记录。
    在使用实体存储库时，你需要创建一个实体来存储模型，并在训练时指定该实体，这样训练结果就可以与该实体相关联并保存到实体存储库中。
    该参数默认值为 None，如果未指定实体，则训练结果将不会与任何实体相关联)
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')#upload_dataset:用于上传数据集，默认关闭
    (1，如果命令行未使用’–upload_dataset’参数，则默认值为default=False，表示不上传数据集。
     2，如果命令行使用’–upload_dataset’参数，但没有传递参数，则默认值为const=True，表示上传数据集。
     3，如果命令行使用’–upload_dataset’参数，并且传递了参数’val’，则默认为True，表示要上传val数据集)
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')#bbox_interval:指定在训练过程中每隔多少个epoch记录一次带有边界框的图片，默认关闭
    (1，n>0，每隔n个epoch记录一次带有边界框的图片；
     2，n<=0，关闭–bbox_interval)
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')#artifact_alias:用于指定要使用的数据集工件的版本别名。
                                                   (在使用MLFlow等工具跟踪模型训练和数据集版本时，会给每个版本分配唯一的别名。通过指定此参数，可以使用特定版本的数据集工件。默认情况下，使用最新版本的数据集工件)

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()): # 主函数
    # Checks 1.检查工作
    if RANK in {-1, 0}: # RANK是与分布式训练相关的，默认不进行分布式训练，值为-1
        print_args(vars(opt)) # 打印传入的参数信息
        check_git_status()    # 检查github代码是否更新
        check_requirements()  # 检查项目所需的依赖包
 
    # Resume 2.从中断恢复（接着上一次继续训练）
    if opt.resume and not (check_wandb_resume(opt) or opt.evolve):  # resume from specified or most recent last.pt
    # 如果opt.resume为True表示需要恢复中断的任务，
    # check_wandb_resume(opt)返回False表示训练没有被wandb恢复，
    # opt.evolve返回False表示不是在执行遗传算法
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())  #获取最新的运行结果的文件路径，并赋值给last变量
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        # 构造一个路径，指向最近运行结果所在的路径的父级目录的父级目录下的opt.yaml文件。
        opt_data = opt.data  # original dataset 
        # 将程序所使用的数据集存储到变量opt_data中，以便后续使用。
        if opt_yaml.is_file():  # 检查opt.yaml是否存在
            with open(opt_yaml, errors='ignore') as f: # 存在则打开该文件
                d = yaml.safe_load(f) # 解析文件的内容并以字典的形式加载，存储在d变量中
        else: #若opt.yaml不存在
            d = torch.load(last, map_location='cpu')['opt']
            # 读取最近运行结果的文件并加载其中保存的PyTorch模型数据及其它信息
        opt = argparse.Namespace(**d)  # replace
        # 将之前从文件中读取到的训练选项信息转换成一个argparse.Namespace对象
        # 使用argparse.Namespace()构造一个命名空间对象opt
        # 并且将之前从文件中读取到的训练选项信息以字典的形式传给Namespace的构造函数
        # **是用来对一个字典进行解包的操作
        # # replace注释说明将opt对象更新为从文件中读取到的训练选项
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        # opt.cfg属性设置为空字符串('')，opt.weights属性设置为last文件名，opt.resume属性设置为True
        # 这些属性指定配置文件的路径、权重文件的路径以及是否恢复模型训练过程等选项。
    if is_url(opt_data): # 将文件路径保存在opt.data属性中
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
    # 检查数据、配置、超参数、权重文件、项目路径是否存在
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # 进行检查

    # 至少指定 --cfg 或 --weights 中的一个
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    # 如果使用进化算法，更新项目路径，并禁用恢复（resume）
    if opt.evolve:
        if opt.project == str(ROOT / 'runs/train'):  # 如果项目路径为默认路径，则改为 runs/evolve
            opt.project = str(ROOT / 'runs/evolve')
        opt.exist_ok, opt.resume = opt.resume, False  # 将 resume 参数传递给 exist_ok，并禁用 resume

    # 如果 opt.name 为 'cfg'，将其更改为模型配置文件的文件名（不带后缀）
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # 使用 model.yaml 文件的文件名作为名称

    # 保存目录路径，使用增量路径以防冲突
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

   # DDP 模式
# 选择设备
device = select_device(opt.device, batch_size=opt.batch_size)

# 如果 LOCAL_RANK 不为 -1，表示处于 DDP 模式
if LOCAL_RANK != -1:
    msg = '不兼容 YOLOv5 的多 GPU DDP 训练'
    
    # 确保不使用 --image-weights 选项
    assert not opt.image_weights, f'--image-weights {msg}'
    
    # 确保不使用 --evolve 选项
    assert not opt.evolve, f'--evolve {msg}'
    
    # 确保 --batch-size 不为 -1（AutoBatch 模式需要有效的 --batch-size）
    assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
    
    # 确保 --batch-size 是 WORLD_SIZE 的倍数
    assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
    
    # 确保有足够的 CUDA 设备
    assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
    
    # 设置当前 CUDA 设备
    torch.cuda.set_device(LOCAL_RANK)
    
    # 初始化分布式数据并行训练组
    dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo',
                            timeout=timedelta(seconds=10800))


  # 训练
if not opt.evolve:
    train(opt.hyp, opt, device, callbacks)

# 进行超参数演化
else:
    # 超参数演化元数据（变异尺度 0-1，下限，上限）
    meta = {
        'lr0': (1, 1e-5, 1e-1),  # 初始学习率 (SGD=1E-2, Adam=1E-3)
        'lrf': (1, 0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
        'momentum': (0.3, 0.6, 0.98),  # SGD 动量/Adam beta1
        'weight_decay': (1, 0.0, 0.001),  # 优化器权重衰减
        'warmup_epochs': (1, 0.0, 5.0),  # 预热周期（小数也可以）
        'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
        'warmup_bias_lr': (1, 0.0, 0.2),  # 预热初始偏置学习率
        'box': (1, 0.02, 0.2),  # 目标框损失权重
        'cls': (1, 0.2, 4.0),  # 类别损失权重
        'cls_pw': (1, 0.5, 2.0),  # 类别 BCELoss 正样本权重
        'obj': (1, 0.2, 4.0),  # 目标损失权重（与像素一起缩放）
        'obj_pw': (1, 0.5, 2.0),  # 目标 BCELoss 正样本权重
        'iou_t': (0, 0.1, 0.7),  # IoU 训练阈值
        'anchor_t': (1, 2.0, 8.0),  # 锚点倍数阈值
        'anchors': (2, 2.0, 10.0),  # 输出网格的每个锚点数（0 表示忽略）
        'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma（efficientDet 默认 gamma=1.5）
        'hsv_h': (1, 0.0, 0.1),  # 图像 HSV-Hue 增强（分数）
        'hsv_s': (1, 0.0, 0.9),  # 图像 HSV-Saturation 增强（分数）
        'hsv_v': (1, 0.0, 0.9),  # 图像 HSV-Value 增强（分数）
        'degrees': (1, 0.0, 45.0),  # 图像旋转（+/- 度）
        'translate': (1, 0.0, 0.9),  # 图像平移（+/- 分数）
        'scale': (1, 0.0, 0.9),  # 图像缩放（+/- 增益）
        'shear': (1, 0.0, 10.0),  # 图像剪切（+/- 度）
        'perspective': (0, 0.0, 0.001),  # 图像透视（+/- 分数），范围 0-0.001
        'flipud': (1, 0.0, 1.0),  # 图像上下翻转（概率）
        'fliplr': (0, 0.0, 1.0),  # 图像左右翻转（概率）
        'mosaic': (1, 0.0, 1.0),  # 图像混合（概率）
        'mixup': (1, 0.0, 1.0),  # 图像混合（概率）
        'copy_paste': (1, 0.0, 1.0)}  # 分割复制粘贴（

       with open(opt.hyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)  # 加载超参数字典
    if 'anchors' not in hyp:  # 在 hyp.yaml 中注释掉了 anchors
        hyp['anchors'] = 3
if opt.noautoanchor:
    del hyp['anchors'], meta['anchors']

opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 仅验证/保存最终时期
# ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 可演化的索引
evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'

if opt.bucket:
    # 如果存在，下载 evolve.csv
    subprocess.run([
        'gsutil',
        'cp',
        f'gs://{opt.bucket}/evolve.csv',
        str(evolve_csv), ])

for _ in range(opt.evolve):  # 进化的代数
    if evolve_csv.exists():  # 如果 evolve.csv 存在：选择最佳超参数并进行变异
        # 选择父代
        parent = 'single'  # 父代选择方法：'single' 或 'weighted'
        x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
        n = min(5, len(x))  # 要考虑的先前结果的数量
        x = x[np.argsort(-fitness(x))][:n]  # 最佳 n 个变异
        w = fitness(x) - fitness(x).min() + 1E-6  # 权重（总和 > 0）
        
        if parent == 'single' or len(x) == 1:
            # x = x[random.randint(0, n - 1)]  # 随机选择
            x = x[random.choices(range(n), weights=w)[0]]  # 权重选择
        elif parent == 'weighted':
            x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 权重组合

               # 变异
mp, s = 0.8, 0.2  # 变异概率，sigma
npr = np.random
npr.seed(int(time.time()))
g = np.array([meta[k][0] for k in hyp.keys()])  # 增益 0-1
ng = len(meta)
v = np.ones(ng)

while all(v == 1):  # 变异直到发生变化（防止重复）
    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)

for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
    hyp[k] = float(x[i + 7] * v[i])  # 变异

# 限制在范围内
for k, v in meta.items():
    hyp[k] = max(hyp[k], v[1])  # 下限
    hyp[k] = min(hyp[k], v[2])  # 上限
    hyp[k] = round(hyp[k], 5)  # 有效数字

# 训练变异
results = train(hyp.copy(), opt, device, callbacks)
callbacks = Callbacks()

# 写入变异结果
keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
        'val/obj_loss', 'val/cls_loss')
print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

# 绘制结果
plot_evolve(evolve_csv)
LOGGER.info(f'超参数进化结束 {opt.evolve}generations\n'
            f"结果保存到 {colorstr('bold', save_dir)}\n"
            f'使用示例: $ python train.py --hyp {evolve_yaml}')

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')  
    opt = parse_opt(True)  # 解析命令行参数，创建Opt实例
    for k, v in kwargs.items():
        setattr(opt, k, v)  # 设置Opt实例的属性，用关键字参数覆盖默认值
    main(opt)  # 调用主函数进行训练
    return opt  # 返回Opt实例



if __name__ == "__main__":
    opt = parse_opt() # 调用parse_opt函数，解析用户传入的参数，存储到opt变量中
    # 这些参数用于传入其它模块或函数
    main(opt)  # 调用主函数，传入opt参数
