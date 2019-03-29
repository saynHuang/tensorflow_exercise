import tensorflow as tf
import numpy as np
import pandas as pd


# 读取测试集数据
test = np.array(pd.read_csv('D:/tensorflow_exercise/data/iris_test.csv'))
x_test = test[:, 0:4]
rows = test.shape[0]
y_test = np.array(np.zeros([rows, 3]))
for r in range(rows):
    label = int(test[r][4])
    y_test[r][label] = 1


sess = tf.Session()
# 加载模型
saver = tf.train.import_meta_graph('../model/iris_model.ckpt.meta')  # 先加载meta文件，具体到文件名
saver.restore(sess, tf.train.latest_checkpoint('../model'))  # 加载检查点文件，具体到文件夹即可
graph = tf.get_default_graph()
xs = graph.get_tensor_by_name('input/xs:0')  # 获取占位符xs
ys = graph.get_tensor_by_name('input/ys:0')  # 获取占位符ys
acc = graph.get_tensor_by_name('accuracy/accuracy:0')
print(sess.run(acc, feed_dict={xs:x_test, ys:y_test}))
sess.close()




