#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import jieba
import jieba.analyse
import os
import codecs
import util_path


class SentenceSegmentation(object):
    '''
    分句类
    '''

    def __init__(self, delimiters=u'?!？！。\n'):
        """
        初始化
        :param delimiters: 分割句子的符号集,也可选（delimiters=u'?!;？！。；…\n'）
        :type delimiters: unicode string
        """
        self.delimiters = delimiters

    @staticmethod
    def has_chinese_character(content):
        """
        判断字符串是否含有中文（静态方法）
        python判断是否是中文需要满足u'[\u4e00-\u9fa5]+'，
        需要注意如果正则表达式的模式中使用unicode，那么
        要匹配的字符串也必须转换为unicode，否则肯定会不匹配。
        """
        # content = unicode(content)
        zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        match = zh_pattern.search(content)
        res = False
        if match:
            res = True
        return res

    @staticmethod
    def has_number_character(content):
        """
        判断字符串是否含有纯数字（静态方法）
        需要注意如果正则表达式的模式中使用unicode，那么
        要匹配的字符串也必须转换为unicode，否则肯定会不匹配。
        """
        # content = unicode(content)
        zh_pattern = re.compile(u'^[0-9]*$')
        match = zh_pattern.search(content)
        res = False
        if match:
            res = True
        return res

    def __sentence_segmentation(self, text, delimiters):
        """
        实现对文章分句(私有)
        :param text: 输出的文本字符串，支持中英文输入
        :type text: unicode string
        :param delimiters: 分割句子的符号集
        :type delimiters: unicode string
        :return: 对文章分好句的句子列表
        :rtype: list
        """
        text = text.strip()  # 去掉头尾空格和换行符
        # text = text.decode('utf8')  # 如果是从编码为 utf8 的 txt 文本中直接输入的话，需要先把文本解码成 unicode 来处理
        if not SentenceSegmentation.has_chinese_character(text):
            delimiters += u'.'
        start = 0
        # i = 0
        quotation_marks = u'""”'
        sents = []
        token = ''
        dot = u'.'
        # for word in text:
        #     if word in delimiters and token not in delimiters:  # 检查标点符号下一个字符是否还是标点
        #         sents.append(text[start:i + 1])
        #         start = i + 1  # start标记到下一句的开头
        #         i += 1
        #     else:
        #         i += 1  # 若不是标点符号，则字符位置继续前移
        #         token = list(text[start:i + 2]).pop()  # 取下一个字符
        for i in range(len(text)):
            if i < len(text) - 1:
                if text[i] in dot:
                    if u'\u4e00' <= text[i - 1] <= u'\u9fa5' and u'\u4e00' <= text[i + 1] <= u'\u9fa5':  # 判断.前后是否为汉字
                        # text[i+1]>= u'\u4e00' and text[i+1]<=u'\u9fa5':
                        sents.append(text[start:i + 1])
                        start = i + 1  # start标记到下一句的开头
                        i += 1
                    elif text[i - 1].strip() == '' and u'\u4e00' <= text[i - 2] <= u'\u9fa5' and text[
                                i + 1].strip() == '' and u'\u4e00' <= text[
                                i + 2] <= u'\u9fa5':  # 判断.前为空格时，空格前面是否为汉字且.后面一个字符是汉字
                        sents.append(text[start:i + 1])
                        start = i + 1  # start标记到下一句的开头
                        i += 1
                elif text[i] in delimiters and text[i + 1] in quotation_marks and token not in delimiters:
                    sents.append(text[start:i + 2])
                    start = i + 2  # start标记到下一句的开头
                    i += 2
                elif text[i] in delimiters and text[i + 1] not in quotation_marks and token not in delimiters:
                    sents.append(text[start:i + 1])
                    start = i + 1  # start标记到下一句的开头
                    i += 1

                else:
                    i += 1  # 若不是标点符号，则字符位置继续前移
                    token = list(text[start:i + 2]).pop()  # 取下一个字符
            elif i == len(text) - 1:
                if text[i] in dot and u'\u4e00' <= text[i - 1] <= u'\u9fa5':
                    # text[i+1]>= u'\u4e00' and text[i+1]<=u'\u9fa5' :
                    sents.append(text[start:i + 1])
                    start = i + 1  # start标记到下一句的开头
                    i += 1
                elif text[i] in delimiters and token not in delimiters:
                    sents.append(text[start:i + 1])
                    start = i + 1  # start标记到下一句的开头
                    i += 1
                    # else:
                    #     i += 1  # 若不是标点符号，则字符位置继续前移
                    #     token = list(text[start:i + 2]).pop()  # 取下一个字符
        if start < len(text):
            sents.append(text[start:])  # 这是为了处理文本末尾没有标点符号的情况
        return sents

    def __sentence_segmentation_based_length(self, text, delimiters, min_length=8):
        """
        基于分句后句子长度判断是否句子需要向后归并
        :param text: 需要分句的文本
        :type text: unicode string
        :param delimiters: 分割符号集
        :type delimiters: string or sequence of string
        :return: list
        :rtype: 结果集
        """
        text = text.strip()  # 去掉头尾空格和换行符
        delimiters += u'.'
        start = 0
        i = 0
        sents = []
        token = ''
        for word in text:
            if word in delimiters and token not in delimiters:  # 检查标点符号下一个字符是否还是标点
                sents.append(text[start:i + 1])
                start = i + 1  # start标记到下一句的开头
                i += 1
            else:
                i += 1  # 若不是标点符号，则字符位置继续前移
                token = list(text[start:i + 2]).pop()  # 取下一个字符
        if start < len(text):
            sents.append(text[start:])  # 这是为了处理文本末尾没有标点符号的情况
        i_list = []
        swap_sents = []
        for i in range(len(sents)):
            if len(sents[i]) == 2 or len(sents[i]) < min_length:
                i_list.append(i)
                sents[i + 1] = sents[i] + sents[i + 1]
        for i in range(len(sents)):
            if i not in i_list:
                swap_sents.append(sents[i])
        return swap_sents

    def segment(self, text, mode='normal'):
        """
        执行文本分句
        :param text: 输入的文档字符串，支持中英文
        :type text: unicode string
        :param mode: 分句模式(normal, other)。normal为普通模式分句；other为基于分句后句子长度判断是否句子需要向后归并。默认选择的是normal模式分句
        :type mode: unicode string
        :return: 返回分句后的句子列表
        :rtype: list
        """
        if mode == 'normal':
            return self.__sentence_segmentation(text, self.delimiters)
        else:
            return self.__sentence_segmentation_based_length(text, self.delimiters)


class WordSegmentation(object):
    '''
    分词
    '''

    stop_words_file = {}.fromkeys([line.strip()
                                   for line in open(util_path.stop_words_path)])  # 加载停用词典
    jieba.set_dictionary(util_path.jieba_dict_path)  # 加载专业词jieba词典
    jieba.initialize()


    def addotherdics(self):
        dfolder=util_path.otherdict_folder
        othdpath=[os.path.join(dfolder,i) for i in os.listdir(dfolder)]
        for inf in othdpath:
            print("add words from %s" %inf)
            with codecs.open(inf,'rU',encoding='utf8') as f:
                for w in f:
                    w=w.strip()
                    if w:
                        jieba.add_word(w)



    def segment(self, sent, stop_words_file=stop_words_file, mode='normal',addotherdic=False):
        """
        对输入文本进行分词处理，可选择性加载停用词，以及选择分词模式
        :param sent: 需分词处理的句子
        :type sent: unicode string
        :param stop_words_file: 停用词表，设置stopwords=None为不过滤停用词。默认加载自带停用词表
        :type stop_words_file: dict
        :param mode: 分词模式(normal,tf-idf，TextRank)。normal为普通模式分词；tf-idf为基于TF-IDF算法的关键词抽取；TextRank为基于TextRank算法的关键词抽取。默认选择的是normal模式分词
        :type mode: unicode string
        :return: 经过分词处理过后的句子列表
        :rtype: list
        """
        sentence_words = []
        # 去除标点符号及空格
        # punct = set(
        #     u''' :!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”
        #     ([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
        # filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
        # sent = filterpunt(sent)
        if addotherdic:
            self.addotherdics()
        if mode == 'tf-idf':
            sent = jieba.analyse.extract_tags(sent.strip(' \t\n\r'))  # 基于 TF-IDF 算法的关键词抽取
        elif mode == 'TextRank':
            tr = jieba.analyse.TextRank()  # 基于 TextRank 算法的关键词抽取
            tr.span = 2
            sent = tr.textrank(sent.strip(' \t\n\r'), topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        else:
            sent = jieba.cut(sent.strip(' \t\n\r'))  # 格式化，去换行符等. 普通模式分词
        for w in sent:
            # seg = str(w.encode('utf-8').strip())
            seg = w.strip()
            if stop_words_file is None:
                sentence_words.append(seg)
            else:
                res = SentenceSegmentation.has_number_character(seg)
                if not seg in stop_words_file and not res:  # 正文需要去停用词（将原来puct符号集移至停用词字典，摒除数字单个成字的可能）
                    sentence_words.append(seg)
        return sentence_words


class Segmentation(object):
    '''
    对文本进行分句和分词两步操作
    '''

    def __init__(self, delimiters=u'?!;？！。；…\n'):
        """
        初始化
        :param delimiters: 分割符号集
        :type delimiters: unicode string
        """
        self.ss = SentenceSegmentation(delimiters)
        self.ws = WordSegmentation()

    def segment(self, text):
        """
        执行分词分句等操作
        :param text: 输入的文本字符串，支持中英文输入
        :type text: unicode string
        :returns: 分好句的文本以及分好次的每个句子
        :rtype:unicode string & list
        """
        sentences = self.ss.segment(text)
        sentence_words = self.ws.segment(sentences)
        return sentences, sentence_words


# 测试
if __name__ == '__main__':
    # import uniout

    ws = WordSegmentation()
    ee = u'445我爱祖国天安门DNA1234'
    ss = u'典型周期谱密度ＤＮＡ123>34，T－DNA，HBV＿DNA,DNA分型方法,ＴｈｅｍａｘｉｍｕｍｐｒｅｃｉｐｉｔａｔｉｏｎａｎｄｗｉｎｄｓｐｅｅｄｏｆＳｏｕｔｈ-ＣｈｉｎａａｎｄＥａｓｔ-Ｃｈｉｎａｃｏａｓｔａｌｌａｎｄｆａｌｌｔｙｐｈｏｏｎ，关于工業企業电力负荷的确定問題，2016年12月20日，磷酸二氢钾１５０克,pm2.5。'
    # print ss
    # ss = IOTools.str_full2half(ss)
    # ss = IOTools.Converter('zh-hans').convert(ss)
    # print len(ss.lower())
    my_list = ws.segment(ss.lower())
    print(my_list)
    # ws.addotherdics()

