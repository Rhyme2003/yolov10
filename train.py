from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r"/home/wang/Desktop/yolov10/yolov10-main-0911/ultralytics/cfg/models/v10/yolov10n.yaml"
#数据集配置文件
data_yaml_path = r'/home/wang/Desktop/yolov10/yolov10-main-0911/dataset_1/dataset.yaml'


# model = YOLOv10("yolov10n.pt")
# model = YOLOv10("yolov10s.pt")
# model = YOLOv10("yolov10m.pt")
# model = YOLOv10("yolov10b.pt")
# model = YOLOv10("yolov10l.pt")
# model = YOLOv10("yolov10x.pt")


#预训练模型
pre_model_name = '/home/wang/Desktop/yolov10/yolov10-main-0911/runs/V10train/exp6/weights/epoch150.pt'
if __name__ == '__main__':
    # #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    # 不加载预训练模型
    # model = YOLOv10(model_yaml_path)

    # 训练配置
    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=200,
        batch=4,  # 根据GPU显存调整(4/8/16/32)
        workers=4,  # 数据加载线程数(Linux可以设高些，Windows设为0)
        optimizer='SGD',  # 也可以尝试 'AdamW'
        amp=False,  # 自动混合精度(出现NaN时可关闭)

        # lr0=0.01,  # 初始学习率
        # lrf=0.01,  # 最终学习率 = lr0 * lrf
        # momentum=0.937,  # SGD动量
        # weight_decay=0.0005,  # 权重衰减
        
        # 可视化配置
        project='runs/V10train',
        name='exp',
        save_period=10,  # 每10个epoch保存一次权重
        visualize=True,  # 增强可视化
        plots=True,  # 训练完成后生成结果图表
        box=0.05,  # 框体损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # 分布焦点损失权重
        
        # 验证配置
        # val=True,  # 训练期间验证
        # fraction=0.2,  # 验证集比例
    )
    
    # # 导出模型为ONNX格式(可选)
    # model.export(format='onnx')
