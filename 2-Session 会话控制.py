# coding:utf-8
import tensorflow as tf

matrix1 = tf.constant([[2, 3]])
matrix2 = tf.constant([[4],
                       [5]])
product = tf.matmul(matrix1, matrix2)

# # method 1
# sess = tf.Session()
# print(sess.run(product))
# sess.close()

# method 2
with tf.Session() as sess:
    print(sess.run(product))
