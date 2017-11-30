import tensorflow as tf

state = tf.Variable(1, name='counter')  # 变量
# print(state.name)
one = tf.constant(1)  # 常量

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer() # 变量的初始化定义

with tf.Session() as sess:
    sess.run(init)
    for _ in range(100):
        sess.run(update)
        print(sess.run(state))
