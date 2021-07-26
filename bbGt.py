import os
import numpy as np
# writened by nh,2021.7.26

class OBJ(object):
    # o=struct('lbl','','bb',[0 0 0 0],'occ',0,'bbv',[0 0 0 0],'ign',0,'ang',0);
    def __init__(self):
        self.lbl = ''
        self.bb = [0, 0, 0, 0]
        self.occ = 0
        self.bbv = [0, 0, 0, 0]
        self.ign = 0
        self.ang = 0


# function [gt0,dt0] = loadAll( gtDir, dtDir, pLoad )
# 将gt和dt表示为numpy数组
def loadAll(gtDir, dtDir, pLoad):
    gt0 = []
    dt0 = []

    if dtDir is None:
        fs = getFiles([gtDir])
        gtFs = fs[0]
    else:
        dtFile = len(dtDir) > 4 and '.txt' == dtDir[-4:]
        if dtFile:
            dirs = [gtDir]
        else:
            dirs = [gtDir, dtDir]
        fs = getFiles(dirs)
        gtFs = fs[0][0]
        if dtFile:
            dtFs = dtDir
        else:
            dtFs = fs[0][1]

    #print(len(gtFs))

    # load ground truth persistent keyPrv gtPrv;
    # key = {gtDir, pLoad};
    n = len(gtFs)
    # if (isequal(key, keyPrv)), gt0=gtPrv; else gt0=cell(1, n);
    for i in range(n):
        _, listbb = bbLoad(gtFs[i], pLoad)
        gt0.append(listbb)
    # gtPrv = gt0;
    # keyPrv = key;
    # end
    #
    # #% load detections
    if os.path.exists(dtDir):
        detdata = np.loadtxt(dtDir, delimiter=',')
        id = detdata[:, 0]
        # if n<= np.max(id):
        # throw exception
        for num in range(n):
            idx = num + 1
            idex_n = detdata[:, 0] == idx
            temp = detdata[idex_n][:, 1:]
            dt0.append(detdata[idex_n][:, 1:])

    else:
        dt0 = None
    # if (isempty(dtDir) | | nargout <= 1), dt0=cell(0); return; end
    # if (iscell(dtFs)), dt0=cell(1, n);
    # for i=1:n, dt1 = load(dtFs{i}, '-ascii');
    #     if (numel(dt1) == 0), dt1=zeros(0, 5); end; dt0{i}=dt1(:,1: 5); end
    #     else
    #     dt1 = load(dtFs, '-ascii'); if (numel(dt1) == 0), dt1 = zeros(0, 6);
    #     end
    #     ids = dt1(:, 1); assert (max(ids) <= n);
    #     dt0 = cell(1, n); for i = 1:n, dt0{i} = dt1(ids == i, 2:6); end
    # end

    return gt0, dt0


# function [gt,dt] = evalRes( gt0, dt0, thr, mul )
def evalRes(gt0, dt0, thr=0.5, mul=0):
    """ Evaluates detections against ground truth data.(from matlab)
     Uses modified Pascal criteria that allows for "ignore" regions. The
     Pascal criteria states that a ground truth bounding box (gtBb) and a
     detected bounding box (dtBb) match if their overlap area (oa):
      oa(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(union(gtBb,dtBb))
     is over a sufficient threshold (typically .5). In the modified criteria,
     the dtBb can match any subregion of a gtBb set to "ignore". Choosing
     gtBb' in gtBb that most closely matches dtBb can be done by using
     gtBb'=intersect(dtBb,gtBb). Computing oa(gtBb',dtBb) is equivalent to
      oa'(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(dtBb)
     For gtBb set to ignore the above formula for oa is used.

     Highest scoring detections are matched first. Matches to standard,
     (non-ignore) gtBb are preferred. Each dtBb and gtBb may be matched at
     most once, except for ignore-gtBb which can be matched multiple times.
     Unmatched dtBb are false-positives, unmatched gtBb are false-negatives.
     Each match between a dtBb and gtBb is a true-positive, except matches
     between dtBb and ignore-gtBb which do not affect the evaluation criteria.

     In addition to taking gt/dt results on a single image, evalRes() can take
     cell arrays of gt/dt bbs, in which case evaluation proceeds on each
     element. Use bbGt>loadAll() to load gt/dt for multiple images.

     Each gt/dt output row has a flag match that is either -1/0/1:
      for gt: -1=ignore,  0=fn [unmatched],  1=tp [matched]
      for dt: -1=ignore,  0=fp [unmatched],  1=tp [matched]

     USAGE
      matlab:[gt, dt] = bbGt( 'evalRes', gt0, dt0, [thr], [mul] )
      python:gt, dt = evalRes( gt0, dt0, [thr], [mul] )
     INPUTS
      gt0  - [mx5] ground truth array with rows [x y w h ignore]
      dt0  - [nx5] detection results array with rows [x y w h score]
      thr  - [.5] the threshold on oa for comparing two bbs
      mul  - [0] if true allow multiple matches to each gt

     OUTPUTS
      gt   - [mx5] ground truth results [x y w h match]
      dt   - [nx6] detection results [x y w h score match]
    """
    if gt0 is None:
        gt0 = np.zeros((0, 5))
    if dt0 is None:
        dt0 = np.zeros((0, 5))
    # assert (size(dt0, 2) == 5);
    nd = dt0.shape[0]
    # assert (size(gt0, 2) == 5);
    ng = gt0.shape[0]

    # sort dt highest score first, sort gt ignore last
    t = dt0[:, -1]
    idx = np.argsort(-dt0[:, -1])
    dt = dt0[idx, :].copy()
    idx = np.argsort(gt0[:, -1])
    gt = gt0[idx, :].copy()
    # [~, ord] = sort(dt0(:, 5), 'descend'); dt0 = dt0(ord,:);
    # [~, ord] = sort(gt0(:, 5), 'ascend'); gt0 = gt0(ord,:);
    gt[:, 4] = -gt[:, 4]
    zs = np.zeros(nd)
    dt = np.column_stack((dt, zs))

    # Attempt to match each (sorted) dt to each (sorted) gt
    # oa size:col:num_gt row:num_dt
    oa = compOas(dt[:, 0:5], gt[:, 0:5], gt[:, 4] == -1)

    for d in range(nd):
        bstOa = thr
        bstg = 0
        bstm = 0  # info about best match so far
        for g in range(ng):
            # if this gt already matched, continue to next gt
            m = gt[g, 4]
            if m == 1 and ~mul:
                continue
            # if dt already matched, and on ignore gt, nothing more to do
            if bstm != 0 and m == -1:
                break
            # compute overlap area,continue to next gt unless better match made
            if oa[d, g] < bstOa:
                continue
            # match successful and best so far, store appropriately
            bstOa = oa[d, g]
            bstg = g
            if m == 0:
                bstm = 1
            else:
                bstm = -1

        g = bstg
        m = bstm
        # store type of match for both dt and gt
        if m == -1:
            dt[d, 5] = m
        elif m == 1:
            gt[g, 4] = m
            dt[d, 5] = m

    return gt, dt


# function [xs,ys,score,ref] = compRoc( gt, dt, roc, ref )
def compRoc(gt, dt, roc=1, ref=[]):
    """
    % Compute ROC or PR based on outputs of evalRes on multiple images.
    %
    % ROC="Receiver operating characteristic"; PR="Precision Recall"
    % Also computes result at reference points (ref):
    %  which for ROC curves is the *detection* rate at reference *FPPI*
    %  which for PR curves is the *precision* at reference *recall*
    % Note, FPPI="false positive per image"
    %
    % USAGE
    %  xs,ys,score,ref = compRoc( gt, dt, roc, ref )
    %
    % INPUTS
    %  gt         - nx first output of evalRes()
    %  dt         - nx second output of evalRes()
    %  roc        - [1] if 1 compue ROC else compute PR
    %  ref        - [] reference points for ROC or PR curve
    %
    % OUTPUTS
    %  xs         - x coords for curve: ROC->FPPI; PR->recall
    %  ys         - y coords for curve: ROC->TP; PR->precision
    %  score      - detection scores corresponding to each (x,y)
    %  ref        - recall or precision at each reference point
    %
    % EXAMPLE
    """
    xs, ys, score = [], [], []

    #  convert to single matrix, discard ignore bbs
    nImg = len(gt)
    # assert (len(dt) == nImg)
    # for s_gt in gt:
    gt = np.concatenate(gt)
    gt = gt[gt[:, 4] != -1, :]
    dt = np.concatenate(dt)
    dt = dt[dt[:, 5] != -1, :]
    # compute    results
    if dt.shape[0] == 0:
        xs = 0
        ys = 0
        score = 0
        ref = ref * 0
        return
    m = len(ref)
    nap = gt.shape[0]
    score = dt[:, 4]
    tp = dt[:, 5]
    # [score, order] = sort(score, 'descend');
    order = np.argsort(-score)
    score = score[order]

    tp = tp[order]
    fp = (tp != 1).astype(float)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    if roc:
        xs = fp / nImg
        ys = tp / nap
        # xs1=[-inf; xs];
        # ys1 = [0;ys]
        t = np.array(float('inf'))
        xs1 = np.concatenate([[-float('inf')], xs])
        ys1 = np.concatenate([[0], ys])

        for i in range(m):
            # j = find(xs1 <= ref(i))
            j = np.argwhere(xs1 <= ref[i])
            ref[i] = ys1[j[-1]]
    else:
        xs = tp / nap
        # ys = tp. / (fp + tp);
        ys = tp / (fp + tp)
        # xs1 = [xs; inf]
        # ys1 = [ys; 0]
        xs1 = np.concatenate([xs, [float('inf')]])
        ys1 = np.concatenate([ys, [0]])
        for i in range(m):
            # j = find(xs1 >= ref(i))
            j = np.argwhere(xs1 >= ref[i])
            ref[i] = ys1[j[1]]

    return xs, ys, score, ref


# function [objs,bbs] = bbLoad( fName, pLoad )
def bbLoad(fName, pLoad):
    objs, bbs = [], []
    squarify, aRng, arRng, oRng, vRng, wRng = None, None, None, None, None, None
    format_d = 0
    ellipse = 1
    lbls, ilbls = pLoad['lbls'], pLoad['ilbls']
    hRng, xRng, yRng = pLoad['hRng'], pLoad['xRng'], pLoad['yRng']
    vType = pLoad['vType']

    # For KAIST - MultispectralDB
    vVal = 0
    if 'none' in vType: vVal = vVal + 1
    if 'partial' in vType: vVal = vVal + 2
    if 'heavy' in vType: vVal = vVal + 4

    if format_d == 0:
        # load objs stored in default format
        fId = open(fName)
        # error(['unable to open file: ' fName]);
        v = 0
        line = fId.readline()
        line = line.strip()
        if 'bbGt' in line:
            v = int(line[-1])
        else:
            v = 0
        for line in fId:
            line = line.strip()
            obj = OBJ()
            # read in annotation(m is number of fields for given version v)
            # if (all(v~=[0 1 2 3])), error('Unknown version %i.', v); end
            frmt = '%s %d %d %d %d %d %d %d %d %d %d %d';
            ms = [10, 10, 11, 12]
            m = ms[v]
            frmt = frmt[:2 + (m - 1) * 3]
            splt_line = line.split()
            bb = [float(strs) for strs in splt_line[1:5]]
            bbv = [float(strs) for strs in splt_line[6:10]]
            lbl = splt_line[0]
            occ = int(splt_line[5])
            obj.bb = bb
            obj.bbv = bbv
            obj.lbl = lbl
            obj.occ = occ

            if m >= 11:
                obj.ign = int(splt_line[10])
            if m >= 12:
                obj.ang = splt_line[11]

            # only choose known lbl's objects
            def isIn(tar, list_str):
                for str in list_str:
                    if tar == str:
                        return True
                return False

            T = obj.lbl not in lbls
            T = isIn(obj.lbl, lbls)
            if (not isIn(obj.lbl, ilbls)) and (not isIn(obj.lbl, lbls)):
                continue
            if ilbls is not None:
                obj.ign = obj.ign or (isIn(obj.lbl, ilbls))
            if xRng is not None:
                obj.ign = obj.ign or (obj.bb[0] < xRng[0]) or (obj.bb[0] > xRng[1])
                x2 = obj.bb[0] + obj.bb[2]
                obj.ign = obj.ign or (x2 < xRng[0]) or (x2 > xRng[1])
            if yRng is not None:
                obj.ign = obj.ign or (obj.bb[1] < yRng[0]) or (obj.bb[1] > yRng[1])
                y2 = obj.bb[1] + obj.bb[3]
                obj.ign = obj.ign or (y2 < yRng[0]) or (y2 > yRng[1])
            if wRng is not None:
                obj.ign = obj.ign or (obj.bb[2] < wRng[0]) or (obj.bb[2] > wRng[1])
            if hRng is not None:
                obj.ign = obj.ign or (obj.bb[3] < hRng[0]) or (obj.bb[3] > hRng[1])
            if vType is not None:
                obj.occ = pow(2, obj.occ)
                ak = int(bin(obj.occ & vVal), 2)
                obj.ign = obj.ign or (ak == 0)

            bb = [float(data) for data in obj.bb]
            bb.append(float(obj.ign))
            objs.append(obj)
            bbs.append(bb)
        bbs = np.array(bbs)
        if len(objs) == 0:
            bbs = None
    return objs, bbs


# [fs,fs0] = getFiles( dirs, f0, f1 )
def getFiles(dirs):
    fs, fs0 = [], []
    for dir in dirs:
        listf = os.listdir(dir)
        listf_p = [os.path.join(dir, f) for f in listf]
        fs.append(listf_p)
        fs0.append(listf)

    return fs, fs0


def compOas(dt, gt, ig=None):
    """
    % Computes (modified) overlap area between pairs of bbs.
    %
    % Uses modified Pascal criteria with "ignore" regions. The overlap area
    % (oa) of a ground truth (gt) and detected (dt) bb is defined as:
    %  oa(gt,dt) = area(intersect(dt,dt)) / area(union(gt,dt))
    % In the modified criteria, a gt bb may be marked as "ignore", in which
    % case the dt bb can can match any subregion of the gt bb. Choosing gt' in
    % gt that most closely matches dt can be done using gt'=intersect(dt,gt).
    % Computing oa(gt',dt) is equivalent to:
    %  oa'(gt,dt) = area(intersect(gt,dt)) / area(dt)
    %
    % USAGE
    %  oa = bbGt( 'compOas', dt, gt, [ig] )
    %
    % INPUTS
    %  dt       - [mx4] detected bbs
    %  gt       - [nx4] gt bbs
    %  ig       - [nx1] 0/1 ignore flags (0 by default)
    %
    % OUTPUTS
    %  oas      - [m x n] overlap area between each gt and each dt bb
    %
    % EXAMPLE
    %  dt=[0 0 10 10]; gt=[0 0 20 20];
    %  oa0 = bbGt('compOas',dt,gt,0)
    %  oa1 = bbGt('compOas',dt,gt,1)
    %
    % See also bbGt, bbGt>evalRes
    """
    oa = []
    m = dt.shape[0]
    n = gt.shape[0]
    oa = np.zeros((m, n))
    if ig is None:
        ig = np.zeros((n, 1))
    de = dt[:, 0:2] + dt[:, 2:4]
    da = dt[:, 2] * dt[:, 3]
    ge = gt[:, 0:2] + gt[:, 2:4]
    ga = gt[:, 2] * gt[:, 3]
    for i in range(m):
        for j in range(n):
            tmp1 = min(de[i, 0], ge[j, 0])
            tmp2 = max(dt[i, 0], gt[j, 0])
            w = np.min(tmp1) - np.max(tmp2)
            if w <= 0:
                continue
            h = min(de[i, 1], ge[j, 1]) - max(dt[i, 1], gt[j, 1])
            if h <= 0:
                continue
            t = w * h
            if ig[j]:
                u = da[i]
            else:
                u = da[i] + ga[j] - t
            oa[i, j] = t / u
    return oa


def compOa(dt, gt, ig):
    oa = []
    pass
    return oa
