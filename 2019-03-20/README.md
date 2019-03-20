## 鸢尾花预测模型
### 1、数据处理
数据集包含150个数据集(其中120个是训练集`iris_training.csv`，30个是测试集`iris_test.csv`)，分为3类（Setosa，Versicolour，Virginica），每类50个数据，每个数据包含4个属性：花萼长度，花萼宽度，花瓣长度，花瓣宽度。

    120,4,setosa,versicolor,virginica  
    6.4,2.8,5.6,2.2,2  
    5.0,2.3,3.3,1.0,1  
    4.9,2.5,4.5,1.7,2  
     .   .   .   .  . 
     .   .   .   .  . 
     .   .   .   .  . 
    4.4,2.9,1.4,0.2,0
    4.8,3.0,1.4,0.1,0
    5.5,2.4,3.7,1.0,1
    
由于标签是鸢尾花的类别，因此将标签转换成独热编码[1, 0, 0], [0, 1, 0], [0, 0, 1]  

### 2、建立模型
采用tensorflow建立一个简单的线性模型：

    W = tf.Variable(tf.zeros([4, 3]))
    b = tf.Variable(tf.zeros([3]) + 0.01)
    output = tf.nn.softmax(tf.matmul(xs, W) + b)
    
输入通过一层网络后直接接入一个`softmax`函数后输出  
损失函数为交叉熵：

    loss = -tf.reduce_sum(ys * tf.log(output + 1e-10))
    
采用梯度下降法最小化`loss`，学习率设置为`0.001`：

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    
### 3、模型训练
模型建立好后，通过`tf.global_variables_initializer()`对变量进行初始化  
模型总共训练1000次，每100次输出`loss`查看训练过程

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
    if i % 100 == 0:
        print('Loss（train set）:%.2f' % (sess.run(loss, feed_dict={xs: x_train, ys: y_train})))
        

        
### 4、鸢尾花种类预测
模型训练完毕之后，即可将测试集输入模型进行预测。由于预测结果是独热编码，所以准确率计算使用`tf.argmax()`函数来实现。返回值是预测结果中最大值的索引，由于独热编码的性质，返回的索引值即为类别。
然后使用`tf.equal()`判断是否与实际类别一致（返回值为bool型）。所以需要通过一个`tf.cast()`函数来转换为[0, 1]值，最后取平均值求出准确率。

    access = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(access, "float"))
    
### 5、结果
最后我们在训练集上以及测试集都得到一个较满意的结果

    --------------------开始训练模型----------------
    Loss（train set）:125.14
    Loss（train set）:67.55
    Loss（train set）:30.55
    Loss（train set）:23.07
    Loss（train set）:20.45
    Loss（train set）:18.60
    Loss（train set）:17.22
    Loss（train set）:16.14
    Loss（train set）:15.28
    Loss（train set）:14.57
    --------------------训练结束--------------------
    
    
    ********************性能评价********************
    训练集准确率： 0.975
    测试集准确率： 0.96666664
