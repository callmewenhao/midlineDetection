# midlineDetection! 💥

> name：midlineDetection！ヾ(≧▽≦*)o
>
> date：2021.12

## 项目介绍

常见的智能车赛道

<div align="center">
    <img src="PyTorch\output\readme\001.png" height="120" width="188" >
    <img src="PyTorch\output\readme\012.png" height="120" width="188" >
    <img src="PyTorch\output\readme\032.png" height="120" width="188" >
</div>
一般的搜赛道边线算法无法很好的处理图片顶部边线，它们可以帮助车子提前做转弯准备，基于这样的原因，我打算使用 deep learning 方法获得远处赛道边线的变化趋势

预期结果是获得图像中五个关键点坐标，如下

<div align="center">
    <img src="PyTorch\output\readme\a1.png" height="120" width="188" >
    <img src="PyTorch\output\readme\a2.png" height="120" width="188" >
    <img src="PyTorch\output\readme\a3.png" height="120" width="188" >
</div>


## 数据集

非常非常小的数据集，只有 41 张图片用于训练，😜过拟合警告！

另外，对于数据集图片的处理还有待提升：

- labelme.json 文件还需另存到 label 下面 (●'◡'●)，麻烦
- 使用训练集做的测试，这一点很不好！😵‍💫

## 模型

ResNet18 作为 backbone，输入为 60*90 的灰度图片

- Pytorch 版本中模型的输出维度为 [5, 2] 5个点的坐标值
- TensorFlow 版本中模型的输出维度为 [10, ] 5个点的坐标值，在计算 loss 时，我进行了reshape([-1, 5, 2])

另外，模型存在一定的问题：

- 无法控制输出范围在 0~1 之间，会预测出负值，也会超过 1，这一点 very sad!
- 个人感觉使用 Pytorch 做框架的训练结果比 TensorFlow 好！后者的 eval 结果不堪入目！🤣

## 训练

训练参数如下：

```yaml
# pytorch
model: resnet18
batch_size: 8
learning_rate: 5e-4
total_epochs: 30
```

```yaml
# tensorflow
model: resnet18
batch_size: 8
learning_rate: 5e-4
total_epochs: 25
```



## 预测结果

### pytorch 版本的结果

<div align="center">
    <img src="PyTorch\output\readme\pred_output_000.png" height="180" width="250" >
    <img src="PyTorch\output\readme\pred_output_006.png" height="180" width="250" >
    <img src="PyTorch\output\readme\pred_output_032.png" height="180" width="250" >
</div>
## 更新

- 2021.12.22 完成 tensorflow 版本的代码

