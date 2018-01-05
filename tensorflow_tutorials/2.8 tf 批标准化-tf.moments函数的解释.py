import tensorflow as tf

out_size=3
img = tf.Variable(tf.random_normal([2, out_size],mean=5))


fc_mean, fc_var = tf.nn.moments(img,axes=[0],)
scale = tf.Variable(tf.ones([out_size]))
shift = tf.Variable(tf.zeros([out_size]))
epsilon = 0.001

Wx_plus_b = tf.nn.batch_normalization(img, fc_mean, fc_var, shift, scale, epsilon)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print (sess.run(img))
print (sess.run(Wx_plus_b))


"""
输出 ： 
$ python tf.moments.py

[[ 6.1853323   5.69743299  6.43799973]
 [ 5.59796476  4.8997345   4.77300262]]
[[ 0.99425316  0.99687099  0.9992795 ]
 [-0.99425125 -0.9968729  -0.9992795 ]]

"""