#!/user/bin/env python3
# -*- coding: utf-8 -*-
alist = ['北京大学','我','爱','北京','生日','快乐']
asent = '北京大学生日快乐'

# 窗口方法，考虑并不全面
def string_match(alist,asent):
    window=[]
    for c in asent:
        if window:
            sub_str=().join(window)
        if sub_str in alist:
            window=[]
        window.append(c)
    if window:
        sub_str = ().join(window)
    if sub_str in asent:
        return True
    else:
        return False

flag=string_match(alist,asent)
print(flag)



