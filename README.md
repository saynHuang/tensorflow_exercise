# 说明文档
`iris.py`是鸢尾花预测模型  
`cali_house.py`是加州房价预测模型  

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


## 加州房价预测模型
### 1、数据预处理
加州房价的数据有20640个样本，特征值有9个，median_house_value作为输出结果。首先使用`describe()`来初步观察数据。

    import pandas as pd

    features = pd.read_csv('D:/tensorflow_exercise/data/housing.csv')
    print(features.describe())

    >>>
             total_bedrooms    population    households  median_income  
    count        20433           20640         20640        20640  
    mean       537.870553   1425.476744    499.539680       3.870671   
    std        421.385070   1132.462122    382.329753       1.899822   
    min          1.000000      3.000000      1.000000       0.499900   
    
以上只展示了部分统计数据，不过可以看到`total_bedrooms`有一部分缺省值，因此我们删去有缺省值的数据。考虑到可能会有重复的数据，所以还需要去掉重复的样本。

    nan = features.dropna(subset=['total_bedrooms'], axis=0)  # 去除缺省值
    repeat = nan.drop_duplicates()  # 去掉重复值样本

### 2、特征工程
在`housing.csv`里面，前面两个特征是经度(longitude)和纬度(latitude)，属于数值型特征。离散的经度纬度对于房价预测似乎没什么重要信息，因此我们对经度纬度进行分箱并合并为独热编码。
我们可以看看之前的统计信息：

              longitude      latitude  
    count        20640        20640 
    mean    -119.569704     35.631861
    std        2.003532      2.135952
    min     -124.350000     32.540000
    max     -114.310000     41.950000  
    
可以知道经度的范围大概在(-124.35, -114.31)之间，纬度的范围大概在(32.54, 41.95)之间。我们以1°为区间分别对经度和纬度进行分箱。
    
    pd.cut(longitude, range(-125, -112), right=False)
    pd.cut(latitude, range(31, 43), right=False)
然而单独考虑经度或是纬度都没有太大的意义，因此我们将这两个特征交叉组合成一个特征，这个特征仍采用独热编码，它的长度为132。

在这些特征中，`ocean_proximity`是字符型特征：NEAR BAY, NEAR OCEAN, ISLAND, INLAND, <1H OCEAN。因此通过`get_dummies()`将其转换为独热编码。

再者，对于`total_rooms`，`population`我们可以用人均房间数`rooms_per_person`来表达他们之间的关系，通过简单的运算即可求出：

    def rooms_per_person(data):  # 合成新特征：人均房间数 = 总房间数 / 总人数
        rooms_per_person = data.apply(lambda x: x['total_rooms'] / x['population'], axis=1)  # 计算特征值
        rooms_per_person[np.abs(rooms_per_person) > 5] = 5  # 对异常值进行截断处理
        rooms_per_person = rooms_per_person.rename('rooms_per_person')  # 特征名称
        return rooms_per_person

计算新特征之后我们重新进行统计，发现有些特征值不合常理，因此对于新特征做截断处理，把人均房间数限制在(0, 5)之间。


### 3、特征值归一化
再来观察数据情况：

              households    median_income  
    count       20640             20640 
    mean      499.539680       3.870671   
    std       382.329753       1.899822   
    min         1.000000       0.499900   
    25%       280.000000       2.563400   
    50%       409.000000       3.534800   
    75%       605.000000       4.743250   
    max       6082.000000      15.000100 
    
可以看到对于不同特征，他们的值可能差别会非常大，如果直接建立模型，可能造成模型在数值大的特征上投入更多精力，会造成结果偏拟合。  
因此需要对特征进行归一化：

    normalized = （value - mean） / std
    
### 4、建立模型
同样采用线性层：

    W = tf.Variable(tf.zeros([142, 1]))
    b = tf.Variable(tf.zeros([1]) + 0.01)
    output = tf.matmul(xs, W) + b
    
损失函数用平方误差：

    loss = tf.reduce_mean(tf.square(ys - output))
   
梯度下降法(learning rate = 0.003)最小化损失函数：

     train_step = tf.train.GradientDescentOptimizer(0.003).minimize(loss)
     
### 5、训练模型
进行特征工程之后，得到一个更为精炼的数据集，将数据集随机划分为训练集和测试集：

    from sklearn.model_selection import train_test_split
    
    data = np.array(pd.read_csv('D:/tensorflow_exercise/data/housing_features.csv'))
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.25, random_state=2018)
    
划分比例为3：1，得到的训练集有15325个样本，测试集有5108个样本。  
训练次数20000次，每1000次输出误差：

        for i in range(20000):
            sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
            if i % 1000 == 0:
                print('Loss（train set）:%.2f' % (sess.run(loss, feed_dict={xs: x_train, ys: y_train})))

### 6、房价预测
最终的训练结果：`Loss（train set）:0.32`
将训练集输入模型，得到预测结果，通过真实-预测关系图来反应模型的性能，同时得到`Loss（test set）:0.31`
![result](house_value_prediction.PNG)
