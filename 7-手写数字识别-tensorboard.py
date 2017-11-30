import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据并设置输出日志目录
data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)
log_dir = "logs/7"

# 设置输入数据以及将图片给tensorboard展示
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1], name='image_data')  # 将输入的1D向量转换为2D图片显示
    tf.summary.image('input', image_shaped_input, 10)

# 隐藏层
with tf.name_scope('layer1'):
    w1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1), name='weight1')
    b1 = tf.Variable(tf.zeros([300]), name='bias1')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # 加入Dropout，解决过拟合
    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')
    # tf.summary.scalar('w1', w1)
    # tf.summary.scalar('b1', b1)

with tf.name_scope('drop_out'):
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob, name='hidden1_drop')  # dropout的作用是随机将某些数据设置为0

# 输出层
with tf.name_scope('layer2'):
    w2 = tf.Variable(tf.zeros([300, 10]), name='weight2')
    b2 = tf.Variable(tf.zeros([10]), name='bias2')
    y_pre = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2, name='y_prediction')
    # tf.summary.scalar('w2', w2)
    # tf.summary.scalar('b2', b2)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 训练网络
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(log_dir, sess.graph)
tf.global_variables_initializer().run()
for i in range(3000):
    xs, ys = data_sets.train.next_batch(100)
    train.run({x: xs, y: ys, keep_prob: 0.75})
    if i % 10 == 0:
        summary = sess.run(tf.summary.merge_all(), feed_dict={x: xs, y: ys, keep_prob: 0.75})
        writer.add_summary(summary, i)

writer.close()

# 预测网络
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pre, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: data_sets.test.images, y: data_sets.test.labels, keep_prob: 1.0}))
