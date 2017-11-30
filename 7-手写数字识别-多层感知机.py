import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data

data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 搭建网络
w1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b1 = tf.Variable(tf.zeros([300]))
w2 = tf.Variable(tf.zeros([300, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # 加入Dropout，解决过拟合

hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)  # dropout的作用是随机将某些数据设置为0
y_pre = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre), reduction_indices=[1]))
train = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 训练网络
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(3000):
    xs, ys = data_sets.train.next_batch(100)
    train.run({x: xs, y: ys, keep_prob: 0.75})

# 预测网络
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pre, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: data_sets.test.images, y: data_sets.test.labels, keep_prob: 1.0}))
