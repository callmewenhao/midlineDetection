# midlineDetection! 💥

> name：midlineDetection！ヾ(≧▽≦*)o
>
> date：2021.12

## 项目介绍

常见的智能车赛道

<div align="center">
    <img src="output\readme\001.png" height="120" width="188" >
    <img src="output\readme\012.png" height="120" width="188" >
    <img src="output\readme\032.png" height="120" width="188" >
</div>

一般的搜赛道边线算法无法很好的处理图片顶部边线，它们可以帮助车子提前做转弯准备，基于这样的原因，我打算使用 deep learning 方法获得远处赛道边线的变化趋势

预期结果是获得图像中五个关键点坐标，如下：

<div align="center">
    <img src="output\readme\a1.png" height="120" width="188" >
    <img src="output\readme\a2.png" height="120" width="188" >
    <img src="output\readme\a3.png" height="120" width="188" >
</div>

## 数据集

非常非常小的数据集，只有 40 张图片用于训练，😜过拟合警告！



## 模型

ResNet18 作为 backbone，输入为 60*90 的灰度图片，输出是维度为 [5, 2] 5个点的坐标值



## 训练

训练参数如下：

```yaml
model: resnet18
batch_size: 8
learning_rate: 5e-4
total_epochs: 30
```



## 预测结果

<div align="center">
    <img src="output\readme\pred_output_000.png" height="180" width="250" >
    <img src="output\readme\pred_output_006.png" height="180" width="250" >
    <img src="output\readme\pred_output_032.png" height="180" width="250" >
</div>



## 问题

- 输出的范围没有限制！如上面第一幅图

- 数据集太小！
- 模型可以改进！
