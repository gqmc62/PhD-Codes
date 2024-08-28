# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:05:40 2024

@author: gqmc62
"""

home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\2024-08-06'
folder1 = home + '//' + '2024-08-02-15-21-56'
folder2 = home + '//' + '2024-08-02-15-22-35'

import os
import glob

os.chdir(home)

if not os.path.exists(home + '/' + 'merged'):
    os.mkdir('merged')

# A = glob.glob(folder1 + '//*')
# B = glob.glob(folder2 + '//*')

A = os.listdir(folder1)
B = os.listdir(folder2)

if A != B:
    print('oh shit')
    
    
file_list = A

