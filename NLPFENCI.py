import jieba
import pandas as pd
import numpy as np

from random import randint
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from PIL import Image


# 读取文件
def readfile(path):
    read_file = open(path)
    content = read_file.read()
    read_file.close()
    return content


# 创建停用词列表
def get_stopwords_list(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords


# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.lcut(sentence.strip())
    return sentence_depart


# 去除停用词
def move_stopwords(sentence_list, stopwords_list):
    # 去停用词
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            out_list.append(word)
    return out_list


# 设置颜色
def random_color(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(randint(0, 360))
    s = int(randint(85, 95))
    l = int(randint(35, 45))
    return "hsl({}, {}%, {}%)".format(h, s, l)


# 读取文件
quanzhi_sentence = readfile('quanzhi.txt')
# 去除 \n \u3000 与空格
quanzhi_sentence = quanzhi_sentence.replace('\n', '').replace('\u3000', '').replace(' ', '')
# 使用自定义词典
jieba.load_userdict('quanzhi_ciku.txt')
# 对句子进行中文分词
quanzhi_sentence_depart = seg_depart(quanzhi_sentence)
# 获取停用词词表
quanzhi_stopwords = get_stopwords_list('quanzhi_stopwords.txt')
# 去除停用词 得出分词的结果
quanzhi_sentence_depart = move_stopwords(quanzhi_sentence_depart, quanzhi_stopwords)

# 计算词频
df = pd.DataFrame(quanzhi_sentence_depart, columns=['word'])
result = df.groupby(['word']).size()
freq_list = result.sort_values(ascending=False)
# 输出频率最高的前40
print(freq_list[:40])

# 绘制词云
word_cloud = WordCloud(font_path='msyh.ttc',
                       width=2400, height=1600,
                       background_color='white',
                       min_font_size=4,
                       color_func=random_color,
                       mask=np.array(Image.open("NLP.png")))
word_cloud.generate(' '.join(quanzhi_sentence_depart))

# 展示
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

# 保存至文件
# word_cloud.to_file("quanzhi_nlp.png")

