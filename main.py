import os
from kaist_eval_full import *
# writened by nh,2021.7.26
if __name__ == '__main__':
    print('test')
    gtDir = 'E:/data/New KAIST/test_annos'

    dtDir = 'E:/Workspace/Python/Object Detection/MBNet-master/data/result'
    kaist_eval_full(dtDir, gtDir, False, True)