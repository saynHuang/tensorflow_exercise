# 基于LSTM的鸢尾花预测模型
使用lstm网络来预测iris数据集  
构建单层LSTM网络，隐藏神经元数目为3  
损失函数为交叉熵  
Adam优化，学习率为0.001  
迭代训练1000次


输入按特征展开为时间序列，shape为[num_samples, num_features, 1]
输出为类别的one-hot编码

    ************************性能评价************************
    训练集准确率： 0.95
    测试集准确率： 0.96666664
    
以上算是模型训练的最好效果了，偶尔训练出来的模型准确率会很低，大概只有0.6~0.7