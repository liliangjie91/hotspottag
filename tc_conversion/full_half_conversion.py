#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wx on 2016/12/15 


def str_full2half(ustring):
    """
    把字符串全角转半角
    :param ustring: 待转换字符串
    :type ustring: string or sequence of string or unicode string
    :return: 经过转换后的字符串
    :rtype: string or sequence of string or unicode string
    """
    switch_string = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        try:
            # if inside_code < 0x0020 or inside_code > 0x7e:   # 0xFF01~0x5FFE全角字符
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                switch_string += uchar
            switch_string += unichr(inside_code)
        except ValueError:
            pass
    return switch_string


def str_half2full(ustring):
    """
    把字符串半角转全角
    :param ustring: 待转换字符串
    :type ustring: string or sequence of string or unicode string
    :return: 经过转换后的字符串
    :rtype: string or sequence of string or unicode string
    """
    switch_string = ""
    for uchar in ustring:
        try:
            inside_code = ord(uchar)
            if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
                switch_string += uchar
            if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
                inside_code = 0x3000
            else:
                inside_code += 0xfee0
            switch_string += unichr(inside_code)
        except ValueError:
            pass
    return switch_string
