from ultralytics import YOLO

# 加载 YOLO 模型
model_path = "/home/hjk/桌面/tesk_rknn/yolov10n.pt"
model = YOLO(model_path)

# 导出为 ONNX 格式，批次大小为 1
onnx_file_path = "/home/hjk/桌面/tesk_rknn/yolov10n.onnx"
model.export(format="onnx", opset=12, dynamic=False)  # 明确禁用动态批次，并使用固定批次大小 1
