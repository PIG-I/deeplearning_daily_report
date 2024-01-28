# deeplearning_daily_report
record the process of learning DL
# 1.1 HRSC2016数据集下载

[HRSC2016 | Kaggle](https://www.kaggle.com/datasets/guofeng/hrsc2016?resource=download)

# 1.2配置深度学习相关环境

## 1.2.1[【傻瓜式】手把手教你搭建深度学习环境以及跑通Github代码（以Pix2PixGAN为例）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV11Z4y1f7u3/?spm_id_from=333.337.search-card.all.click&vd_source=02778f0c82c409496f3d6dc2e022f9b0)

## 1.2.2[Pytorch 通过Colab平台训练深度学习网络-Demo-毕设可用（Bubbliiiing 深度学习 教程）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1GS4y1Y7Qb/?spm_id_from=333.337.search-card.all.click&vd_source=02778f0c82c409496f3d6dc2e022f9b0)

note：1为本地环境搭建教程，2为用谷歌免费的colab平台训练教程

# 1.3简单学习神经网络与深度学习

[神经网络15分钟入门！足够通俗易懂了吧 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/65472471)

[【官方双语】深度学习之反向传播算法 上/下 Part 3 ver 0.9 beta_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV16x411V7Qg/?spm_id_from=333.880.my_history.page.click&vd_source=02778f0c82c409496f3d6dc2e022f9b0)

[【深度学习-第1篇】深度学习是什么、能干什么、要怎样学？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/632712454)

[【深度学习-第2篇】CNN卷积神经网络30分钟入门！足够通俗易懂了吧（图解） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/635438713)

# 1.4下周

尝试教程1.2.2，在colab上试运行代码。

# 2.1 摸索rotated-rtmdet

    看了看它在HRSC2016数据集上的成绩

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-16-06-28-image.png?msec=1706429189269)

[HRSC2016 Benchmark (Object Detection In Aerial Images) | Papers With Code](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

源代码地址：

https://github.com/open-mmlab/mmrotate/blob/1.x/README_zh-CN.md

    按照之前的计划，我试着跟着前文找到的教程在colab上训练模型，结果发现因为没理解rotated-rtmdet的整个结构，不知道源代码训练集和测试集是怎么导入的，导致现在是跑不了代码了。

    不过colab上的准备工作是准备的差不多了。

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-16-17-48-image.png?msec=1706429868535)

    T4GPU训练一个小模型够用了。将test文件和train文件，以及训练用的模型文件上传至云盘，再将云盘挂载到colab，利用谷歌提供的免费的T4GPU，自己电脑不行也可以训练一个小模型了。//估计真正跑代码的时候还有不少问题。

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-16-17-07-image.png?msec=1706429827627)

请教学长后，发现rotated-rtmdet用了个大框架（mmrotate）。想跑代码还得了解一下这个框架。

# 2.2 MMRotate

https://github.com/open-mmlab/mmrotate/blob/1.x/README_zh-CN.md

总结以下要点：

    1. MMRotate 的模块设计-[学习基础知识 (待更新) &mdash; MMRotate 1.0.0rc1 文档](https://mmrotate.readthedocs.io/zh-cn/1.x/overview.html)

    2.在Google Colab上安装(https://mmrotate.readthedocs.io/zh-cn/1.x/get_started.html#google-colab)

        [Google Colab](https://research.google.com/) 通常已经完成了Pytorch的安装， 因此我们只需要按照步骤完成MMCV和MMDetection的安装即可。

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-16-50-14-image.png?msec=1706431814366)

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-16-50-29-image.png?msec=1706431829763)

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-16-50-48-image.png?msec=1706431848306)

最后一步时间很长

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-17-12-39-image.png?msec=1706433159140)

    3.[学习配置文件 (待更新) &mdash; MMRotate 1.0.0rc1 文档](https://mmrotate.readthedocs.io/zh-cn/1.x/user_guides/config.html)这里展示了配置文件里相关代码的用法，格式等。对后续看懂代码很有帮助。

# 2.3 难点

根据requirement.txt文件安装包，还是会出现各个包版本不匹配的情况。

# 2.4 下周

![](file://C:\Users\Torches\AppData\Roaming\marktext\images\2024-01-28-18-01-14-image.png?msec=1706436074160)

搞懂rotated_rtmdet_tiny-9x-hrsc.py文件的来龙去脉，争取理解各个python文件的调用关系，先不急着跑代码了。
