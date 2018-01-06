# VGG19_with_tensorflow
An easy implement of VGG19 with tensorflow, which has a detailed explanation.

<img src="https://raw.githubusercontent.com/hjptriplebee/VGG19_with_tensorflow/master/testModel/005525.jpg" width = "200" height = "150" alt="alexnet" /><img src="https://raw.githubusercontent.com/hjptriplebee/VGG19_with_tensorflow/master/testModel/002689.jpg" width = "200" height = "150" alt="alexnet" /><img src="https://raw.githubusercontent.com/hjptriplebee/VGG19_with_tensorflow/master/testModel/000018.jpg" width = "200" height = "150" alt="alexnet" />

<img src="https://raw.githubusercontent.com/hjptriplebee/VGG19_with_tensorflow/master/demo1.png" width = "200" height = "150" alt="tensorflow" /><img src="https://raw.githubusercontent.com/hjptriplebee/VGG19_with_tensorflow/master/demo2.png" width = "200" height = "150" alt="tensorflow" /><img src="https://raw.githubusercontent.com/hjptriplebee/VGG19_with_tensorflow/master/demo3.png" width = "200" height = "150" alt="tensorflow" />

The code is an implement of VGG19 with tensorflow. The detailed explanation can be found [here](http://blog.csdn.net/accepthjp/article/details/70170217).

Before running the code, you should confirm that you have :

- Python (2 and 3 is all ok, 2 need a little change on function"print()")
- tensorflow 1.0
- opencv

Then, you should download the model file "vgg19.npy" which can be found [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) or [here](http://pan.baidu.com/s/1eRLSwwE)(for users in china).

Finally, run the test file with "**python3 testModel.py folder testModel**", you will see some images with the predicted label (press any key to move on to the next image).

The command also **supports url**. 

For eg. "**python3 testModel.py url http://www.cats.org.uk/uploads/images/featurebox_sidebar_kids/Cat-Behaviour.jpg**"

You can also use tensorboard to monitor the process. Remeber to see [detailed explanation](http://blog.csdn.net/accepthjp/article/details/70170217).

<br />
<br />

If you have any problem, please contact me!

blog  ：[http://blog.csdn.net/accepthjp](http://blog.csdn.net/accepthjp)

email ：huangjipengnju@gmail.com
