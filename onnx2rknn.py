from rknn.api import RKNN  # 导入RKNN库，用于将ONNX模型转换为RKNN模型

# 定义输入的ONNX模型路径和输出的RKNN模型路径
ONNX_MODEL = '/home/hjk/桌面/tesk_rknn/yolov10n.onnx'  # 输入的ONNX格式的YOLO模型
RKNN_MODEL = '/home/hjk/桌面/tesk_rknn/yolov10n_int8.rknn'  # 输出的RKNN模型（量化后的INT8格式）

# 输入图片路径，用于后续推理验证
IMG_PATH = '/home/hjk/桌面/tesk_rknn/IMG/000000000009.jpg'  # 一张示例图片路径
DATASET = '/home/hjk/桌面/tesk_rknn/dataset.txt'  # 量化数据集路径，RKNN将使用该数据集来进行量化校准

# 设置目标设备为RK3588s，假设这是一个带有NPU的设备
DEVICE_NAME = 'rk3588'  # 目标设备名称，RK3588s是Rockchip的一款NPU设备

# 启用量化（INT8量化）
QUANTIZE_ON = True  # 启用INT8量化，转换过程中会进行量化操作，降低模型精度，但加速推理
IMG_SIZE = 640  # 输入图像的尺寸（640x640），需要与模型训练时的输入尺寸匹配

def convert():
    # 创建RKNN对象，进行模型转换操作
    rknn = RKNN(verbose=True, verbose_file=None)  # 初始化RKNN对象，verbose=True表示输出详细的日志信息

    # 设置预处理参数：均值、标准差等
    # 这些参数主要用于图像预处理，比如归一化、RGB到BGR的转换等
    rknn.config(
        mean_values=[[0, 0, 0]],  # 均值处理，通常是图像通道的均值（针对不同的预训练模型，均值会不同）
        std_values=[[1, 1, 1]],  # 标准差处理，图像的标准差（通常是1，表示不进行标准化）
                                 # 设置目标平台为RK3588s
        quant_img_RGB2BGR=True  # 设置为True时，输入图像会从RGB格式转换为BGR格式
    )

    # 加载ONNX模型，并传递输入图像的尺寸
    ret = rknn.load_onnx(model=ONNX_MODEL, input_size_list=[[3, IMG_SIZE, IMG_SIZE]])  # 加载ONNX模型，指定输入图像的尺寸
    if ret != 0:
        print('Load model failed!')  # 如果加载模型失败，则打印错误信息
        exit(ret)  # 退出程序并返回错误代码

    # 构建RKNN模型，并启用量化过程，传递数据集用于量化校准
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)  # 启用量化，并传入量化数据集路径
    if ret != 0:
        print('Build model failed!')  # 如果模型构建失败，则打印错误信息
        exit(ret)  # 退出程序并返回错误代码

    # 导出转换后的RKNN模型
    ret = rknn.export_rknn(RKNN_MODEL)  # 导出RKNN模型到指定路径
    if ret != 0:
        print('Export rknn model failed!')  # 如果导出模型失败，则打印错误信息
        exit(ret)  # 退出程序并返回错误代码

    # 释放RKNN对象的资源
    rknn.release()  # 释放资源，避免内存泄漏

# 如果脚本作为主程序运行，则调用convert()函数
if __name__ == '__main__':
    convert()  # 执行模型转换操作
