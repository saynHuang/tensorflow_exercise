# 保存训练好的模型以及加载模型进行预测
保存模型主要是进行命名，方便我们找到张量的接口，比如：  
```
    with tf.name_scope('input'):
        xs = tf.placeholder(dtype='float', shape=[None, 4], name='xs')
        ys = tf.placeholder(dtype='float', shape=[None, 3], name='ys')
```
这样我们定义的占位符，在后面加载模型的时候就可以使用函数  
```
xs_ph = graph.get_tensor_by_name('input/xs:0')  # 获取占位符xs
ys_ph = graph.get_tensor_by_name('input/ys:0')  # 获取占位符ys
```
来获取到他们的位置，之后我们把数据feed到xs_ph和ys_ph就可以了。然后获取输出，也是调用同样的函数  
```
acc = graph.get_tensor_by_name('accuracy/accuracy:0')
```
注意此时的xs_ph、ys_ph和acc都是张量，至于tensor之间的计算我们不需要关心，因为它已经在我们之前的模型中定义好了。
