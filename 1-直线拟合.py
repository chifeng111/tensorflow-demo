import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###create tensorflow structure start ###
weights = tf.Variable(tf.zeros([1, 1]))
biases = tf.Variable(tf.zeros([1, 1]))

x = tf.placeholder(tf.float32, [None, 1])  # 输入数据行不限，只有1列。
y = tf.placeholder(tf.float32, [None, 1])

y_ = tf.matmul(x, weights) + biases
loss = tf.reduce_mean(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
###create tensorflow structure end ###

### creat data start ###
x_data = np.random.rand(100).reshape([100, 1])
y_data = x_data * 0.1 + 0.3
### creat data end ###

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data, color='black')
plt.pause(0.3)

sess = tf.Session()
sess.run(init)
for _ in range(101):
    sess.run(train, feed_dict={x: x_data, y: y_data})
    print("loss: {}, w: {}, b: {}".format(sess.run(loss, feed_dict={x: x_data, y: y_data}), \
                                          sess.run(weights, feed_dict={x: x_data, y: y_data}), \
                                          sess.run(biases, feed_dict={x: x_data, y: y_data})))
    if _ % 10 == 0:
        predict = sess.run(y_, feed_dict={x: x_data, y: y_data})
        lines = ax.plot(x_data, predict, 'r-')
        plt.pause(0.3)
plt.show()
