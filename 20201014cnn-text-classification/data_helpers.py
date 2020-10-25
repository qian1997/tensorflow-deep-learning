import numpy as np
import re
import itertools
from collections import Counter


#数据预处理
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # re.sub正则表达 不是特定字符（大小写数字）都变成空格
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # 按照每个需要分割的部分加上空格
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    # 匹配2个或多个空白字符变成一个" "空格
    string = re.sub(r"\s{2,}", " ", string)
    # 去掉句子首尾的空白符，再转小写
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #从文件中国加载positive和negative数据，将文件中的每一行读取成一个list元素
    positive_examples = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    #strip去掉每个sentence结尾的换行符
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    #x_text是一个二维列表，存储了每个review的每个word，分词
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    #对应的labels也组合在一起，labels实际对应的是二分类输出层的两个神经元
    #因此用one-hot编码成0/1和1/0，然后返回y
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    #数组拼接函数concatenate axis=0 按0轴纵向拼接
    """
    [[0,1],
     [0,1],
     [1,0]]
    """
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


#batch样本生成器 输入数据，批次大小，迭代周期
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    定义一个函数，输出batch样本，参数为data（包括feature和label），batchsize，epoch
    """
    data = np.array(data) #全部数据转化为array
    data_size = len(data) #数组长度
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 #每个epoch有多少个batch，个数
    print("num_batches_per_epoch:",num_batches_per_epoch)
    #迭代周期
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices] # shuffled_data按照上述乱序得到新的样本
        else:
            shuffled_data = data
        #开始生成batch
        #每个周期有num个批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            #这里主要是最后一个batch可能不足batchsize的处理
            end_index = min((batch_num + 1) * batch_size, data_size)
            #yield，在for循环执行时，每次返回一个batch的data，占用的内存为常数
            yield shuffled_data[start_index:end_index]
