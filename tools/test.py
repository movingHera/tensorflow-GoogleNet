import tensorflow as tf

a = tf.get_variable("a", shape=())

sess = tf.Session()

with tf.variable_scope("", reuse=True):
    var = tf.get_variable("a")
    sess.run(var.assign(1.0))

#sess.run(tf.global_variables_initializer())

print sess.run(a)


