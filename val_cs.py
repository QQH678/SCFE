from ultralytics import YOLO


model = YOLO("runs/detect/train33/weights/best.pt")
#66.9,47(96 epoch)0.587      0.404
#67.5,47.4(120 epoch)
#67.6,47.5(127 epoch)
#67.7,47.6(140 epoch)
#67.8,47.7(150 epoch)
#67.7,47.6(155 epoch)
#67.5,47.5(160 epoch)
#our machine
#67.6,47.6(160 epoch)
#67.5,47.5(156 epoch)
#67.4,47.5(153 epoch)

#our machine-1
#67.5,47.5(150 epoch)
#67.4,47.4(160 epoch)
#dysamole
#67.7,47.6(150 epoch)
#67.8,47.7(dy+sppec)
#dy+sppf+spp
#67,47.2(81 epoch)
#67.5,47.6(100 epoch)
#67.8,47.9(110 epoch)
#68,48.1(116 epoch)
#68.1,48.1(120 epoch)
#68.2,48.2(137 epoch)
#68.2,48.2(140 epoch)
#68.3,48.3(145 epoch)
#68.4,48.4(151 epoch)
#68.3,48.4(155 epoch)
#68.3,48.3(158 epoch)
#best-sppelam-cm
#68.2,48(125 epoch)
#68.3,48.1(146 epoch)
#68.6,48.2(156 epoch)
#68.6,48.3(160 epoch)
#68.6,48.4(164 epoch)
#68.7,48.4(178 epoch)

#baseline 50.2,34
#b+ASAHI 66.7,47.1
#b+

validation_results = model.val(
    data="ultralytics/cfg/datasets/VisDrone_slice_12slice.yaml",          # 数据集配置文件路径
    # data="/home/dell/HHB/ultralytics-main/ultralytics/cfg/datasets/tinyperson.yaml",
    imgsz=2016,                  # 输入图像尺寸
    batch=1,                   # 每批次的图像数量
    conf=0.25,                  # 最低检测置信度阈值
    iou=0.45,                    # NMS的IoU阈值
    max_det=300,                # 每张图片的最大检测数
    device="cuda:0",            # 指定使用的设备 (GPU)
    half=True,                  # 使用半精度 (FP16) 推理
    save_json=True,             # 保存结果为 JSON 文件
    save_hybrid=False,          # 是否保存混合标签（原始标签 + 预测）
    plots=True,                 # 生成预测与标签对比的可视化图
    rect=True,                  # 使用矩形推理，减少填充，提高效率
    split="val",                # 使用验证集进行验证
    project="runs/val",         # 输出项目目录
    name="yolo11n_validation"   # 本次验证的运行名称
)

#print(validation_results)