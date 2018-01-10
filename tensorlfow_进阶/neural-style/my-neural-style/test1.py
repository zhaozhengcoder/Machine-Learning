
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os


IMAGE_W = 800 
IMAGE_H = 600 
CONTENT_IMG =  './images/Taipei101.jpg'
STYLE_IMG = './images/StarryNight.jpg'
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION = 5000

CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]
MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))


def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i,):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][0][0][1]
    print ("--------------")
    print (bias)
    #print (bias.shape)
    #print (len(bias))
    print ("--------------")
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias

def build_vgg19(path):
  # net是一个字典　，net 里面记录了每一层的输出
  net = {}
  vgg_rawnet = scipy.io.loadmat(path)
  vgg_layers = vgg_rawnet['layers'][0]
  # 创建输入的变量，但是全为0
  net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
  
  net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
  net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
  net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
  net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
  net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
  net['pool3']   = build_net('pool',net['conv3_4'])
  net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19))
  net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
  net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
  net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
  net['pool4']   = build_net('pool',net['conv4_4'])
  net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28))
  net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
  net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
  net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
  net['pool5']   = build_net('pool',net['conv5_4'])
  return net

def build_content_loss(p, x): #p_shape -(1,75,100,512)
  M = p.shape[1]*p.shape[2]  #7500
  N = p.shape[3]             #512
  loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))  
  return loss


def main():
  #vgg 模型
  net = build_vgg19(VGG_MODEL)

  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
  
  # CONTENT_IMG 内容的图片 ; STYLE_IMG 风格的图片
  content_img = read_image(CONTENT_IMG)
  style_img = read_image(STYLE_IMG)

  print ("content img shape : ",content_img.shape )
  print ("style img shape : ",style_img.shape)
  print ("noise img shape : ",noise_img.shape)


  #for key in net:
  #    print (key," output shape is ---> ",net[key].shape)



  sess.run([net['input'].assign(content_img)])
  # map (f  , [1,2,3,4])
  # map ( lambda 参数 ：参数 + 干一堆事 ,   CONTENT_LAYERS )
  # 这里的干一堆事 指的 --> l[1]* build_content_loss(  sess.run(net[l[0]]) ,net[l[0]])
  # CONTENT_LAYERS =[('conv4_2',1.)] 这里的l[0] 指的是 ：conv4_2 ， l[1] 指的是 ：1
  # cost_content = sum(map(lambda l,: l[1]* build_content_loss(  sess.run(net[l[0]]) ,net[l[0]]),  CONTENT_LAYERS))
  #print (sess.run(build_content_loss(  sess.run(net[CONTENT_LAYERS[0][0]]) ,net[CONTENT_LAYERS[0][0]])))
  #cost_content = sum(map(lambda l,: l[1]* build_content_loss(  sess.run(net[l[0]]) ,net[l[0]]),  CONTENT_LAYERS))
  #print (sess.run(cost_content))
  cost_content = sum(map(lambda l,: l[1]* build_content_loss(  sess.run(net[l[0]]) ,net[l[0]]),  CONTENT_LAYERS))
  sess.run([net['input'].assign(style_img)])


  #sess.close()



def read_image(path):
  image = scipy.misc.imread(path)
  image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
  image = image[np.newaxis,:,:,:] 
  image = image - MEAN_VALUES
  return image

if __name__ == '__main__':
      main()