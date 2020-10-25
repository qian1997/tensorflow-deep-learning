import tensorflow as tf  
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

#网络训练
# Data loading params
"""
flags命令参数
tf.flags.DEFINE_xxx() FLAGS = tf.flags.FLAGS FLAGS._parse_flags()
第一个是参数名称，第二个参数是默认值，第三个是参数描述
用于帮助我们添加命令行的可选参数。
利用该函数我们可以实现在命令行中选择需要设定的参数来运行程序，
可以不用反复修改源代码中的参数，直接在命令行中进行参数的设定。
"""
#数据集占比
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#正样本
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#负样本
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
#词向量长度
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
#卷积核大小
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#每一种卷积核个数
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
#dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#L2正则化参数
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
#批次大小
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
#迭代周期
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
#多少step测试一次
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#多少step保存一次
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
#最多保存多少模型
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
#tensorflow会自动选择一个存在并且支持的设备来运行
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#获取你的operation和tensor被指派在哪个设备上运行
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#flags解析，可以在命令行改参数
FLAGS = tf.flags.FLAGS
#如果报错 AttributeError: _parse_flags 更改FLAGS.flag_values_dict()
#FLAGS._parse_flags
#打印所有参数  参数名=值
#print("\nParameters:")
#sorted() 对可迭代的对象进行排序操作，返回一个新的list
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")


# Data Preparation
# ==================================================
def preprocess():
    # Load data
    print("Loading data...")
    #调用data_helpers中的load_data_and_labels函数
    #通过导入文件的函数来下载数据 x_text二维列表，存储每个review的每个word，
    #它们对应的labels也组合在一起，labels实际对应的是二分类输出层的两个神经元，因此用one-hot编码成0/1和1/0，然后返回y。
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    #TextCNN模型中的input_x对应的是tf.placeholder，是一个tensor，shape已经固定好，要找到最长的句子长度
    max_document_length = max([len(x.split(" ")) for x in x_text])
    #根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #通过fit_trainform方法将我们的文本数据fit到这个对象中，最终才能学习到这个文本对应的词汇表并返回单词对应的编号。
    #使用这些编号做embedding np.array()将列表转换为数组
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data 随机排序数据
    #设置相同的seed，则每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样
    np.random.seed(10)
    #np.random.permutation() 随机排列序列  np.arange()函数返回一个有终点和起点的固定步长的排列
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    #随机排序了前面x,y得到的索引数组，打乱数据
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set 训练集与测试集，90%训练，10%验证
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    del x, y, x_shuffled, y_shuffled
    
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev
    

# Training正式训练主要部分
# ==================================================
#tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图
def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        #tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置：
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        #创建一个默认的对话
        with sess.as_default():
            #调用text_cnn中的TextCNN，TextCNN是一个类，输入参数，得到一个CNN结构
            cnn = TextCNN(
                sequence_length=x_train.shape[1],#sequence_length:最长词汇数
                num_classes=y_train.shape[1],#num_classes:分类数
                vocab_size=len(vocab_processor.vocabulary_),#vocab_size:总词汇数
                embedding_size=FLAGS.embedding_dim,#embedding_size:词向量长度
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),#filter_sizes:卷积核的尺寸3 4 5
                num_filters=FLAGS.num_filters,#num_filters：卷积核的数量
                l2_reg_lambda=FLAGS.l2_reg_lambda)#l2_reg_lambda_l2正则化系数

            # Define Training procedure
            #全局变量step
            global_step = tf.Variable(0, name="global_step", trainable=False)
            #选择训练优化器，学习率
            optimizer = tf.train.AdamOptimizer(1e-3)
            #选择目标函数，计算梯度；返回的是梯度和变量
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            #运应梯度，将compute_gradients()返回的值作为输入参数对variable进行更新
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    #tf.summary显示梯度的直方图
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g) 
                    #tf.nn.zero_fraction计算所0在tensor类型的占比
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            #相当于把信息合并        
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time())) #得到当前的时间
            ##根据时间定义了一个路径
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy  显示标量信息
            loss_summary = tf.summary.scalar("loss", cnn.loss) #生成损失标量图
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy) #生成准确率标量图

            # Train Summaries
            # 合并信息
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            #定义训练可视化信息保存的路径
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            #用于将Summary写入磁盘，需要制定存储路径logdir
            #如果传递了Graph对象，则在Graph Visualization会显示Tensor Shape Information
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            #返回绝对路径
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            #保存模型
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            #保存生成的词库
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables 初始化所有变量
            sess.run(tf.global_variables_initializer())

            #定义函数，输入一个batch
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                #梯度更新（更新模型），步骤加一，存储数据，计算一个batch的损失，计算一个batch的准确率
                #获取当前的时间
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            #定义了一个函数，用于验证集，输入为一个batch
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            # Generate batches（生成器），得到一个generator，
            #每一次返回一个batch，没有构成list[batch1,batch2,batch3,...]
            #调用data_helpers中的batch_iter函数
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch) #将配对的样本，分离出来data和label
                train_step(x_batch, y_batch) #训练，输入batch样本，更新模型
                current_step = tf.train.global_step(sess, global_step)
                #每隔多少步的训练后，评估一次
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                #每隔多少步的训练后,保存模型
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()