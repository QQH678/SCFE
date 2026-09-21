from ultralytics import RTDETR

# 加载 YOLOv11 模型
model = RTDETR("runs/compare-rtdetr/rt7/weights/best.pt")
#66.9,47(96 epoch)
#67.5,47.4(120 epoch)

#sppf-lska
#67.4,47.5(165epoch)
#67.5,47.5(160epoch)
#67.5,47.5(153epoch)
#67.3,47.3(131epoch)
#yolo11-origin
#51,34.4(150epoch-640)
#51.1,34.5(111epoch)

validation_results = model.val(
    # data="/home/dell/QQH/xView.yaml",  # 数据集配置文件路径
    data="ultralytics/cfg/datasets/NWPU.yaml",
    imgsz=640,                  # 输入图像尺寸
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
