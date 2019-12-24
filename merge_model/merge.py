import tensorflow as tf
import numpy as np
import re


class ImportGraph():
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def train(self, data, label, model_name, save_dir):
        '''
        参数说明
        :param data: 输入数据
        :param label: 数据对应的标签
        :param model_name: 此次训练的模型名字，不要重复，因为会影响后续合并模型的操作
        :param save_dir: 模型保存的路径
        '''
        features = 1
        classes = 1
        epoch = 100
        lr = 0.01

        with self.graph.as_default():
            print('开始训练模型%s\n' % model_name)

            xs = tf.placeholder(dtype=tf.float32, shape=[None, features], name='data')
            ys = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='label')

            weights_name = 'weights'
            bias_name = 'bias'

            with tf.name_scope(model_name):
                W = tf.Variable(tf.random_normal(shape=[features, classes]), name=weights_name)
                b = tf.Variable(tf.constant(0.1, shape=[classes]), name=bias_name)

            with tf.name_scope('train'):
                out = tf.matmul(xs, W) + b
                loss = tf.losses.absolute_difference(ys, out)

                tf.add_to_collection('output', out)  # 把网络输出output存起来
                tf.add_to_collection('loss', loss)  # 把网络损失loss存起来

                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

            init = tf.global_variables_initializer()
            self.sess.run(init)
            saver = tf.train.Saver()
            print('initial---W=%.4f  b=%.4f' % (self.sess.run(W), self.sess.run(b)))
            for ep in range(epoch):
                self.sess.run(train_step, feed_dict={xs: data, ys: label})
                if (ep+1) % 10 == 0:
                    temp_loss = self.sess.run(loss, feed_dict={xs: data, ys: label})
                    # print('epoch %d: %.4f' % (ep, temp_loss))
                    weights, bias = self.sess.run([W, b])
                    print('epoch %2d: %.4f\tW=%.4f  b=%.4f' % (ep+1, temp_loss, weights, bias))
            saver.save(self.sess, save_dir + model_name)
            print('模型%s训练完毕，并保存至%s\n' % (model_name, save_dir))
        self.sess.close()

    def load(self, meta, ckpt, data, label=None):
        '''
       :param meta: meta文件的路径
       :param ckpt: checkpoint所在文件夹的路径
       :param data: 数据
       :param label: 标签
       '''

        with self.graph.as_default():
            # 从指定路径加载模型到局部图中
            print('加载模型参数...')
            saver = tf.train.import_meta_graph(meta)
            saver.restore(self.sess, tf.train.latest_checkpoint(ckpt))

            self.loss = tf.get_collection('loss')
            self.output = tf.get_collection('output')

        output = self.sess.run(self.output, feed_dict={"data:0": data})[0]
        if label is not None:
            loss = self.sess.run(self.loss, feed_dict={'data:0': data, 'label:0': label})[0]
        else:
            loss = None
        self.sess.close()
        print('计算结果已返回')
        return output, loss


def train_multi_models():
    # 训练多个模型
    ImportGraph().train(x, y1, 'model1', 'models/model1/')  # W=2.0069  b=0.0420
    ImportGraph().train(x, y2, 'model2', 'models/model2/')  # W=3.0125  b=0.8580


def load_multi_models():
    x_test = np.reshape(np.array([11, 12, 13, 14, 15]), [-1, 1])
    # 加载多个模型
    result1, loss1 = ImportGraph().load('models/model1/model1.meta', 'models/model1/', x_test)
    result2, loss2 = ImportGraph().load('models/model2/model2.meta', 'models/model2/', x_test)
    print(' x_test:', end='')
    for i in x_test:
        print('%4d' % i, end='')
    print('\n--------------------------------------------------')
    print('model_1(y=2x+0):', end='')
    for i in result1:
        print('%.2f  ' % i, end='')
    print('\nmodel_2(y=3x+1):', end='')
    for i in result2:
        print('%.2f  ' % i, end='')


def merge_models():
    # 合并模型
    features = 1
    classes = 1
    epoch = 100
    lr = 0.05
    xs = tf.placeholder(dtype=tf.float32, shape=[None, features], name='data')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='label')

    with tf.name_scope('model1'):
        W1 = tf.Variable(tf.random_normal(shape=[features, classes]), trainable=False, name='weights')
        b1 = tf.Variable(tf.constant(0.1, shape=[classes]), trainable=False, name='bias')
        out1 = tf.matmul(xs, W1) + b1

    with tf.name_scope('model2'):
        W2 = tf.Variable(tf.random_normal(shape=[features, classes]), name='weights')
        b2 = tf.Variable(tf.constant(0.1, shape=[classes]), name='bias')
        out2 = tf.matmul(out1, W2) + b2

    loss = tf.losses.absolute_difference(ys, out2)

    tf.add_to_collection('output', out2)
    tf.add_to_collection('loss', loss)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # tf.reset_default_graph()
    variables = tf.contrib.framework.get_variables_to_restore()
    model1_vars = [v1 for v1 in variables if re.search('model1', v1.name) is not None]
    model2_vars = [v2 for v2 in variables if re.search('model2', v2.name) is not None]

    # tf.train.import_meta_graph('models/model1/model.meta')
    saver1 = tf.train.Saver(model1_vars)
    saver2 = tf.train.Saver(model2_vars)
    with tf.Session() as sess:
        saver1.restore(sess, tf.train.latest_checkpoint('models/model1'))
        saver2.restore(sess, tf.train.latest_checkpoint('models/model2'))
        weight1, bias1, weight2, bias2 = sess.run([W1, b1, W2, b2])
        print('initial variables: W1=%.4f  b1=%.4f  W2=%.4f  b2=%.4f' % (weight1, bias1, weight2, bias2))
        for ep in range(epoch):
            sess.run(train_step, {xs: x, ys: y3})
            if (ep+1) % 10 == 0:
                weight1, bias1, weight2, bias2 = sess.run([W1, b1, W2, b2])
                print('ep %3d: W1=%.4f  b1=%.4f  W2=%.4f  b2=%.4f' % (ep+1, weight1, bias1, weight2, bias2))
        print(sess.run(out2, {xs: x}))


if __name__ == '__main__':
    # 生成数据
    x = np.array([i for i in range(10)])
    y1 = np.array(list(map(lambda i: 2*i, x)))
    y2 = np.array(list(map(lambda i: 3*i + 1, x)))

    x = x.reshape([10, 1])
    y1 = y1.reshape([10, 1])
    y2 = y2.reshape([10, 1])

    # train_multi_models()  # 训练多模型
    # load_multi_models()  # 加载模型

    # 合并模型并微调
    label = np.array(list(map(lambda i: 8 * i + 3, x)))
    y3 = label.reshape([10, 1])
    merge_models()
    print(label)

