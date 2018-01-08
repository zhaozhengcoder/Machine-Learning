import tensorflow as tf 

A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
B = tf.Variable(tf.constant(1.0), dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print ("A :",sess.run(A))
    print ("B :",sess.run(B))
    
    ## 两种assign的方式，但是必须加上sess.run ,否则赋值无效
    #sess.run(tf.assign(B,A))
    #sess.run(B.assign(A))

    #没有加run
    #tf.assign(B,A)  # b的值就不会改变，仍然是1
    B.assign(A)     # 同理，b的值就不会改变，仍然是1


    print ("A :",sess.run(A))
    print ("B :",sess.run(B))

"""
A : 0.0
B : 1.0
A : 0.0
B : 0.0

"""