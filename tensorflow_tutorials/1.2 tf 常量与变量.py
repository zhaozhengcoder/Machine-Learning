import tensorflow as tf 

## 定义
#定义一个变量
var =tf.Variable(0,name="myvar")

#定义一个常量
con_var=tf.constant(1)

#定义一个加法
new_var=tf.add(var,con_var)

## 开始计算

#初始化，在初始化之前是变量是没有值的
init =tf.global_variables_initializer()

#这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.
sess=tf.Session()

#计算
sess.run(init)

#输出
print ('var : ',sess.run(var))
print ('con_var : ',sess.run(con_var))
print ('new_var : ',sess.run(new_var))

# 关闭会话
sess.close()
"""
# 另一种写法
with tf.Session() as sess:
    sess.run(init)
    print ('var : ',sess.run(var))
    print ('con_var : ',sess.run(con_var))
    print ('new_var : ',sess.run(new_var))
"""