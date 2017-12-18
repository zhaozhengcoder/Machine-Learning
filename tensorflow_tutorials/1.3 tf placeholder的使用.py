import tensorflow as tf 

# placeholder 是 Tensorflow 中的占位符
# 如果想要从外部传入data, 那就需要用到 tf.placeholder()
# 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)


# 需要传入的值放在了feed_dict={} 并一一对应每一个 input. 
# placeholder 与 feed_dict={} 是绑定在一起出现的。

#这里没有变量，就不需要 init =tf.global_variables_initializer() 这一步了
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))