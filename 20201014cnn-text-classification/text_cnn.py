import tensorflow as tf
import numpy as np
import pickle

#网络结构设计
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    # sequence_length-最长词汇数
    # num_classes-分类数
    # vocab_size-总词汇数
    # embedding_size-词向量长度
    # filter_sizes-卷积核尺寸3，4，5
    # num_filters-卷积核数量
    # l2_reg_lambda-l2正则化系数
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        #定义输入，输出数据的placeholder和dropout。
        #tf.placeholder相当于占一个空间。第二个参数是tensor的shape，None表示这个维度可以是任意数。
        #第一个维度是批数据的大小，用None表示，可以是任意大小的批数据
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") 
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #dropout系数
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer 词嵌入层
        #这里用的随机生成的vector。最后的tf.expand_dims()来扩充数据，生成[[None, sequence_length, embedding_size
        #W就是所有词的词向量矩阵，然后embedded_chars是输入句子所对应的的词向量矩阵，
        #返回的就是[batch_siz sequence_length,embedding_size]，然后手动为其添加了一个维度。
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #vocab_size词典大小  返回一个维度为[rows, colomns]，范围为[low, high]的均匀分布随机浮点数张量
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            #tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
            #根据input_x中的id，查找embedding-W中对应的元素,返回的还是一个tensor
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 添加一个维度，[batch_size, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size  卷积池化层
        pooled_outputs = [] #空的list
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        #0-3 1-4 2-5
        for i, filter_size in enumerate(filter_sizes):
            #通过不同的卷积核大小，进行卷积层操作
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 高*宽*通道*卷积个数
                #卷积窗口filter_size（3,4,5）,embedding_size（128），在一个平面，num_filters（128）个卷积核
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #卷积参数
                #变量维度filter_shape的tensor
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #变量维度卷积核个数的tensor
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                """
                tf.nn.conv2d函数,2d二维
                x是一个tensor，形状是[batch,in_height,in_width,in_channels]NHWC关系，
                分别是批次大小(本例 batch_size=100)，图片高度，图片宽度，通道数(黑白照片是1，彩色是3)
                w是一个滤波器，tensor，形状是[filter_height,filter_width,in_channels,out_channels]，
                滤波器长，宽，输入和输出通道数
                步长参数，strides[0]=strides[3]=1,strides[1]代表x方向的步长，strides[2]代表y方向的步长
                padding:一个字符串，要么是'SAME'要么是'VALID'，对应两种卷积方法，前者补零,后者不会超出平面外部
                """
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,#需要卷积的矩阵
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity 激活函数
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs 池化输出
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                #每个卷积核和pool处理一个样本后得到一个值，这里维度如batchsize*1*1*卷积核个数，三次
                pooled_outputs.append(pooled)

        # Combine all the pooled features 卷积核数量*卷积核种类
        num_filters_total = num_filters * len(filter_sizes)
        #三个池化层的输出拼接
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 把池化层输出变成一维向量
        #tf.reshape(tensor,shape,name=None) 将tensor（h_pool）变为参数shape（num_filters_total）形式
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Add dropout 防止过拟合，训练时以一定概率的神经元 停止工作
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions 全连接 预测
        with tf.name_scope("output"):
            #系数tensor
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes], #num_classes-分类数 [64*3,2]
                initializer=tf.contrib.layers.xavier_initializer())  #初始化权重矩阵
            #偏置tensor
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #加入W的l2正则化
            l2_loss += tf.nn.l2_loss(W)
            #加入b的l2正则化
            l2_loss += tf.nn.l2_loss(b)
            #计算全连接的输出
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            #计算预测值，输出最大值的索引，0或者1，
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #交叉熵+全连接w和b的正则化
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            #根据input_y和predictions是否相同，得到一个矩阵batchsize大小的tensor
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #计算均值即为准确率
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
