import os
from bbGt import *
import numpy as np
# writened by nh,2021.7.26


def kaist_eval_full(dtDir, gtDir, reval=True, writeRes=True):
    tname = os.path.split(dtDir)[-1]

    bbsNms, isEmpty = aggreg_dets(dtDir, reval, tname)
    if ~isEmpty[0] or ~isEmpty[1] or ~isEmpty[2]:
        # no detection
        return None
    exps = [
        ['Reasonable-all', 'test-all', [55, float("inf")], ['none', 'partial']],
        ['Reasonable-day', 'test-day', [55, float("inf")], ['none', 'partial']],
        ['Reasonable-night', 'test-night', [55, float("inf")], ['none', 'partial']],
        ['Scale=near', 'test-all', [115, float("inf")], ['none']],
        ['Scale=medium', 'test-all', [45, 115], ['none']],
        ['Scale=far', 'test-all', [1, 45], ['none']],
        ['Occ=none', 'test-all', [1, float("inf")], ['none']],
        ['Occ=partial', 'test-all', [1, float("inf")], ['partial']],
        ['Occ=heavy', 'test-all', [1, float("inf")], ['heavy']],
        ['all', 'test-all', [1,float("inf")], ['none', 'partial']],
        ['all-day', 'test-day', [1,float("inf")], ['none', 'partial']],
        ['all-night', 'test-night', [1,float("inf")], ['none', 'partial']],
    ]

    res = []

    for exp in exps:
        res = run_exp(res, exp, gtDir, bbsNms)

    if writeRes:
        # 原先默认最后是det结尾，现在改为任意的上一级目录
        # 保存结果
        pass

    return res


# return aggregated files
# bbsNm.test-all
def aggreg_dets(dtDir, reval, tname):
    bbsNms = {}
    isEmpty = []
    conditons = ['test-all', 'test-day', 'test-night']
    for cond in conditons:
        desName = tname + '-' + cond + '.txt'
        split_end = os.path.split(dtDir)[-1]
        le = len(split_end)
        tmpstr = dtDir[:-le]
        # desName = os.path.join(tmpstr,desName)
        bbsNms[cond] = desName
        if os.path.exists(desName) and (not reval):
            continue

        if cond == 'test-all':
            setIds = [6, 7, 8, 9, 10, 11]
            skip = 20
            vidIds = [4, 2, 2, 0, 1, 1]
        if cond == 'test-day':
            setIds = [6, 7, 8]
            skip = 20
            vidIds = [4, 2, 2]
        if cond == 'test-night':
            setIds = [9, 10, 11]
            skip = 20
            vidIds = [0, 1, 1]

        fidA = open(desName, 'w+');
        num = 0

        def getRange(num):
            list_r = []
            for i in range(num + 1):
                list_r.append(i)
            return list_r
        num_bbox_det = 0
        for s in range(len(setIds)):
            list_vidIds = getRange(vidIds[s])
            for v in list_vidIds:
                for i in range(skip - 1, 99999, skip):
                    detName = 'set%02d_V%03d_I%05d.txt' % (setIds[s], v, i)
                    detName = os.path.join(dtDir, detName)
                    if not os.path.exists(detName):
                        continue
                    num = num + 1
                    x1, y1, x2, y2, score = [], [], [], [], []
                    detfile = open(detName)
                    for line in detfile:
                        line = line.strip()
                        datas = line.split(' ')
                        if len(datas) == 5:
                            x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, datas))
                        else:
                            x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, datas[1:]))
                        x1.append(x1_t), x2.append(x2_t)
                        y1.append(y1_t), y2.append(y2_t)
                        score.append(score_t)
                    lens = min(20, len(score))
                    for j in range(len(score)):
                        strinput = '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n' % (
                        num, x1[j] + 1, y1[j] + 1, x2[j] - x1[j], y2[j] - y1[j], score[j])
                        num_bbox_det +=1
                        fidA.write(strinput)
        fidA.close()
        if num_bbox_det == 0:
            isEmpty.append(True)
        else:
            isEmpty.append(False)
    return bbsNms,isEmpty


def run_exp(res, iexp, gtDir, bbsNms):
    res_e = {}
    thr = 0.5
    mul = 0
    pows = np.arange(-2, 0.25, 0.25)
    ref = np.power(10, pows)
    pLoad = {'lbls': ['person', ], 'ilbls': ['people', 'person?', 'cyclist']}
    # pLoad0 = {'lbls':['person', 'people', 'person?'], 'ilbls':"['cyclist']};
    pLoad.update({'hRng': iexp[2], 'vType': iexp[3], 'xRng': [5, 635], 'yRng': [5, 507]})

    # res(end + 1).name = iexp{1};
    res_e['name'] = iexp[0]
    # bbsNms.(sprintf('%s', strrep(iexp{2}, '-', '_')))
    bbsNm = bbsNms[iexp[1]]
    # original annotations
    annoDir = os.path.join(gtDir, iexp[1], 'arcnn_vis')
    gt, dt = loadAll(annoDir, bbsNm, pLoad)
    # 给每一帧计算
    gt_n = []
    dt_n = []
    for i in range(len(gt)):
        gtn, dtn = evalRes(gt[i], dt[i], thr, mul)
        gt_n.append(gtn)
        dt_n.append(dtn)
    fp, tp, score, miss = compRoc(gt_n, dt_n, 1, ref.copy())
    miss_ori = np.exp(np.mean(np.log(np.maximum(1-miss,1e-10))))
    roc_ori = [score, fp, tp]

    res_e['ori_miss'] = miss
    res_e['ori_mr'] = miss_ori
    res_e['roc'] = roc_ori

    # improved annotations
    annoDir = os.path.join(gtDir, iexp[1], 'arcnn_lwir')
    gt, dt = loadAll(annoDir, bbsNm, pLoad)
    # 给每个gt计算
    gt_n = []
    dt_n = []
    for i in range(len(gt)):
        gtn, dtn = evalRes(gt[i], dt[i], thr, mul)
        gt_n.append(gtn)
        dt_n.append(dtn)
    fp, tp, score, miss = compRoc(gt_n, dt_n, 1, ref.copy())
    miss_imp = np.exp(np.mean(np.log(np.maximum(1-miss,1e-10))))
    roc_imp = [score, fp, tp]

    res_e['imp_miss'] = miss
    res_e['imp_mr'] = miss_imp
    res_e['imp_roc'] = roc_imp

    strshow = '%-30s \t log-average miss rate = %02.2f%% (%02.2f%%) recall = %02.2f%% (%02.2f%%)' % \
              (iexp[0], miss_ori * 100, miss_imp * 100, roc_ori[2][-1] * 100, roc_imp[2][-1] * 100)
    print(strshow)
    res.append(res_e)
    return res


if __name__ == '__main__':
    dtDir = 'E:/Workspace/Python/Object Detection/AR-CNN-master/ARCNN-officialDet/det/aligned_rcnn_vgg16_0_3'
    tname = 'aligned_rcnn_vgg16_0_3'
    aggreg_dets(dtDir, True, tname)
