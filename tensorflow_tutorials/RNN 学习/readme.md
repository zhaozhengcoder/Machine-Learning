## RNN 读《伊索寓言》 来预测下一个词汇

用示例短文（出自《伊索寓言》）训练一个RNN来预测下一个单词（就像输入法里常见的“联想”功能。

1. 这个是从 github ( https://github.com/roatienza/Deep-Learning-Experiments/tree/master/Experiments/Tensorflow/RNN ) 的地方拉下来的代码，使用lstm写得一个预测下一个单词的序列网络。

2. 参考的教程是 ：https://jizhi.im/blog/post/1hour_lstm  

3. 结构：

    rnn_word.py 是别的写得代码 

    belling_the_cat.txt 是训练集

    text1.py 是我读原来的代码的时候，把一部分代码copy出来，这样可能会更好的理解原来的代码

    text2.py 是另外一个部分的核心代码

    rnn_demo.py 是我根据上述的代码自己写得，是一个rnn的网络模型。这个是核心代码。建议先看这个。

    rnn_demo2.py 是完整的，可以运行的，完整的代码。



PS :

参考GitHub ： https://github.com/roatienza/Deep-Learning-Experiments/tree/master/Experiments/Tensorflow/RNN

参考文章 ： https://jizhi.im/blog/post/1hour_lstm