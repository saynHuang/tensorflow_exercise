import pandas as pd
import numpy as np
import tensorflow as tf


def main():
    # 读取训练集数据
    train = np.array(pd.read_csv('D:/tensorflow_exercise/data/iris_training.csv'))
    x_train = train[:, 0:4]
    rows = train.shape[0]
    y_train = np.array(np.zeros([rows, 3]))  # 将标签转化为one-hot编码
    for r in range(rows):
        label = int(train[r][4])
        y_train[r][label] = 1

    # 读取测试集数据
    test = np.array(pd.read_csv('D:/tensorflow_exercise/data/iris_test.csv'))
    x_test = test[:, 0:4]
    rows = test.shape[0]
    y_test = np.array(np.zeros([rows, 3]))
    for r in range(rows):
        label = int(test[r][4])
        y_test[r][label] = 1

    with tf.name_scope('input'):
        xs = tf.placeholder(dtype='float', shape=[None, 4], name='xs')
        ys = tf.placeholder(dtype='float', shape=[None, 3], name='ys')
    W = tf.Variable(tf.zeros([4, 3]), name='weights')
    b = tf.Variable(tf.zeros([3]) + 0.01, name='bias')
    with tf.name_scope('prediction'):
        output = tf.nn.softmax(tf.matmul(xs, W) + b, name='output')  # 输出加个softmax层
    with tf.name_scope('loss'):
        loss = -tf.reduce_sum(ys * tf.log(output + 1e-10))  # 损失函数用交叉熵
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # 梯度下降法最小化损失函数
    with tf.name_scope('accuracy'):
        access = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(access, "float"), name='accuracy')

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print('--------------------开始训练模型--------------------')
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
        if i % 100 == 0:
            print('Loss（train set）:%.2f' % (sess.run(loss, feed_dict={xs: x_train, ys: y_train})))

    print('--------------------训练结束--------------------\n\n')
    print('************************性能评价************************')
    print('训练集准确率：', sess.run(accuracy, {xs: x_train, ys: y_train}))
    print('测试集准确率：', sess.run(accuracy, {xs: x_test, ys: y_test}))
    saver.save(sess, '../model/iris_model.ckpt')
    print('模型已保存')
    sess.close()

if __name__ == '__main__':
    main()