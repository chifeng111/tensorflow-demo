import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data

data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(data_sets.train.images)#训练集的图片， 55000个样本，每个数据784列。
# print(data_sets.train.labels)  # 训练集标签，55000x10

# 定义网络结构
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
y_ = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

# 训练网络
sess = tf.InteractiveSession()
init.run()
for i in range(1000):
    xs, ys = data_sets.train.next_batch(100)
    train.run({x: xs, y: ys})

# 预测网络
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: data_sets.test.images, y: data_sets.test.labels}))
