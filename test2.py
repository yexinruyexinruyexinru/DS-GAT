#!/user/bin/env python3
# -*- coding: utf-8 -*-
import sys
# def main(n_pkt, n_hh, n_sfw, p_hh):


if __name__ == '__main__':
    print(len(sys.argv))
    print(type(sys.argv))
    for i in range(0, len(sys.argv)):
        print('参数 %s 为：%s' % (i, sys.argv[i]))
