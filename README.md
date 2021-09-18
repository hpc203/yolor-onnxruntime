# yolor-onnxruntime
使用ONNXRuntime部署anchor-free系列的YOLOR，包含C++和Python两种版本的程序

YOLOR是一个anchor-free系列的YOLO目标检测，不需要anchor作为先验。本套程序参考了YOLOR的
官方程序(https://github.com/WongKinYiu/yolor)， 官方代码里是使用pytorch作为深度学习框架的。
根据官方提供的.pth文件，生成onnx文件后，我本想使用OpenCV作为部署的推理引擎的，但是在加载onnx
文件这一步始终出错，于是我决定使用ONNXRuntime作为推理引擎。在编写完Python版本的程序后，
在本机win10-cpu环境里，在visual stdio里新建一个c++空项目，按照csdn博客里的文章讲解来配置onnxruntime，
配置的步骤跟配置Opencv的步骤几乎一样。在编写完c++程序后，编译运行，感觉onnxruntime的推理速度要比
opencv的推理速度快，看来以后要多多使用onnxruntime作为推理引擎了，毕竟onnxruntime是微软推出的专门针对
onnx模型做推理的框架，对onnx文件有着最原生的支持。
本套程序里的onnx文件从百度云盘下载，
链接：https://pan.baidu.com/s/1Mja0LErNE4dwyj_oYsOs2g 
提取码：qx2j
