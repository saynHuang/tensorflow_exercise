import tensorflow as tf
import pandas as pd
import numpy as np


def main():
    print('--------------正在加载数据--------------')
    # 读取训练集数据
    train = np.array(pd.read_csv('../data/iris_training.csv'))
    x_train = train[:, 0:4]
    rows = train.shape[0]
    y_train = np.array(np.zeros([rows, 3]))  # 将标签转化为one-hot编码
    for r in range(rows):
        label = int(train[r][4])
        y_train[r][label] = 1

    # 读取测试集数据
    test = np.array(pd.read_csv('../data/iris_test.csv'))
    x_test = test[:, 0:4]
    rows = test.shape[0]
    y_test = np.array(np.zeros([rows, 3]))
    for r in range(rows):
        label = int(test[r][4])
        y_test[r][label] = 1
    print('--------------加载完成--------------\n')


    # ================构建LSTM网络================
    # 设置超参数
    num_classes = 3  # 标签类别数
    num_units = 3  # LSTM隐藏状态的大小，等于类别数
    num_features = 4  # 特征维数
    x_train = np.reshape(x_train, [-1, num_features, 1])  # 这里将数据按照特征展开为时间序列
    x_test = np.reshape(x_test, [-1, num_features, 1])

    print('--------------开始构建LSTM网络--------------')
    # 构建graph
    xs = tf.placeholder(tf.float32, [None, num_features, 1])
    ys = tf.placeholder(tf.float32, [None, num_classes])

    # LSTM
    lstm_cell = tf.keras.layers.LSTMCell(num_units)  # 构建一层lstm，隐藏状态数目为3
    lstm_out, states = tf.nn.dynamic_rnn(lstm_cell, xs, dtype=tf.float32)  # 得到输出和隐藏状态
    logit = lstm_out[:, -1, :]  # shape为[num_samples, time_steps, num_units],所以我们获取的是输出的最后一个序列
    access = tf.equal(tf.argmax(logit, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(access, tf.float32))
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(ys, logit))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
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


