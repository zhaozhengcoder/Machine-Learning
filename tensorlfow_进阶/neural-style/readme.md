## my-neural-stye


* 如何执行
    ```
    # 进入 my-neural-style文件夹 ，执行：
    python3 neural-style.py
    ```

    执行的时候，需要加载预训练模型imagenet-vgg-verydeep-19.mat ， 这个文件比较大，没有办法push到github里面。解决办法：

    1. 去官网下载一下
    
         http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

    2. 如果不能翻墙，我也在百度云上存了一份

        https://pan.baidu.com/s/1V-qnokC3vX9uIyoA17oIhg




* 文件及文件夹的作用

    my-neural-style 文件夹下面 ： 
        image/    输入图片，包括内容图片，风格图片

        results_docker/   存放输出的图片

        neural-sytle.py   完整版的代码
        
        test1.py 和 test2.py 是checkout的两个分支，我也没删除，留下来了。以后没事，看看实现的过程。

* Blog 

    实现的原理写在blog上面了 ： https://www.jianshu.com/p/25036ca64408

* 环境
    ```
    python3.5
    tensorflow 1.2
    ```