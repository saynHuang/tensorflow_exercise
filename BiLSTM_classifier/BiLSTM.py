import tensorflow as tf
import pandas as pd
import numpy as np


def main():
    print('--------------正在加载数据--------------')
    # 读取训练集数据
    train = np.array(pd.read_csv('../data/iris_training.csv'))
    x_train = train[:, 0:4]
    y_train = train[:, 4]  # 此时一个样本标签为单个数字

    # 读取测试集数据
    test = np.array(pd.read_csv('../data/iris_test.csv'))
    x_test = test[:, 0:4]
    y_test = test[:, 4]
    print('--------------加载完成--------------\n')


    label_embed = [[1, 0, 0],  # 定义标签转换字典，将标签转为one-hot编码
                   [0, 1, 0],
                   [0, 0, 1]
                   ]


    # ================构建LSTM网络================
    # 设置超参数
    # num_classes = 3  # 分类数目
    num_units = 3  # LSTM隐藏状态的大小
    num_features = 4  # 特征维数
    x_train = np.reshape(x_train, [-1, num_features, 1])  # 数据shape转为LSTM要求的格式
    x_test = np.reshape(x_test, [-1, num_features, 1])  # [n_samples, time_steps, embedding_dim]


    print('--------------开始构建LSTM网络--------------')
    # 构建graph
    xs = tf.placeholder(tf.float32, [None, num_features, 1])
    ys = tf.placeholder(tf.int32, [None])

    ########################如果按照之前的方式处理数据集，这里的代码可以删去########################
    label_embed = tf.Variable(label_embed)  # 没有这行代码的话，embedding_lookup的结果会不同，具体原因我不清楚
    y_label = tf.nn.embedding_lookup(label_embed, ys)  # 将标签转换为one-hot编码
    ###############################同时ys的shape为[None, num_classes]###############################

    # LSTM
    lstm_fw_cell = tf.keras.layers.LSTMCell(num_units)  # 定义一个LSTM做为前向LSTM
    lstm_bw_cell = tf.keras.layers.LSTMCell(num_units)  # 定义一个LSTM做为后向LSTM
    [lstm_fw_out, lstm_bw_out], states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, xs, dtype=tf.float32)
    logit = tf.concat([lstm_fw_out[:, -1, :], lstm_bw_out[:, -1, :]], axis=1)  # 将fw，bw的最后一个time_steps输出拼到一起

    w = tf.Variable(tf.random_normal([2*num_units, num_units]))  # 由于之前的拼接，logit维度为2*num_units
    b = tf.Variable(tf.zeros([num_units], dtype=tf.float32))  # 所以添加一个线性层，把logit的维度降为一半
    out = tf.add(tf.matmul(logit, w), b)  # 最后输出的维度即为标签的维度

    # 定义损失函数以及准确度
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_label, out))  # softmax交叉熵
    access = tf.equal(tf.argmax(out, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(access, tf.float32))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)  # 用Adam优化损失
    print('--------------网络构建完成--------------\n')

    print('--------------开始训练--------------')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
        if i % 100 == 0:
            print('Loss（train set）:%.2f' % (sess.run(loss, feed_dict={xs: x_train, ys: y_train})))
    print('--------------训练完成--------------\n')


    print('************************性能评价************************')
    print('训练集准确率：', sess.run(accuracy, {xs: x_train, ys: y_train}))
    print('测试集准确率：', sess.run(accuracy, {xs: x_test, ys: y_test}))
    sess.close()


if __name__ == '__main__':
    main()


