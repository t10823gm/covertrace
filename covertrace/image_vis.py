import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
# Add visualizing division and unconfident tracking.

class ImageVis(object):
    def __init__(self, images, data, state):
        self.images = images
        self.data = data
        self.state = state

    def mark_prop(self, frame=0, pid=0):
        ch = getattr(self.images, self.state[1])
        ch_img = ch(frame=frame, rgb=True)

        # label_id_arr = self.data.__getitem__((self.state[0], self.state[1], 'label_id'))
        label_id_arr = self.data.__getitem__('cell_id')

        prop_cell = set(label_id_arr[self.data.prop[:, frame] == pid, frame])
        obje = getattr(self.images, self.state[0])
        obj_img = obje(frame=frame)
        for cell in prop_cell:
            bool_img = obj_img == cell
            ch_img[bool_img, 1] = 255
        return ch_img

    def show_single_cell(self, label_id=1, MARGIN=30, frame=0):
        ch = getattr(self.images, self.state[1])
        label_id_arr = self.data.__getitem__('cell_id')
        idx = np.where(label_id_arr == label_id)[0][0]
        x_vec = self.data.__getitem__((self.state[0], self.state[1], 'x'))[idx, :]
        y_vec = self.data.__getitem__((self.state[0], self.state[1], 'y'))[idx, :]
        x, y = x_vec[frame], y_vec[frame]
        ch_img = ch(frame=frame)
        y_ran = slice_adjust_margin(y, ch_img.shape[0], MARGIN)
        x_ran = slice_adjust_margin(x, ch_img.shape[1], MARGIN)
        img = ch_img[y_ran, x_ran]
        return img


def slice_adjust_margin(x, maximum, MARGIN):
    """Return a slice defined by x-MARGIN and x+MARGIN.
    If x is below 0, it uses 0 instead of x-MARGIN.
    If x is above maximum, it uses maximum instead of x-MARGIN.

    Examples:

        >>> slice_adjust_margin(20, 50, 30)
        slice(0, 50, None)
    """
    x = int(x)
    LOW = x - MARGIN if x-MARGIN >= 0 else 0
    HIGH = x + MARGIN if x+MARGIN < maximum else maximum
    return slice(LOW, HIGH)

def slect_divided_cell(site, location, label):
    tmp_cellid = site[location, label, 'cell_id']
    tmp_parent = site[location, label, 'parent']
    list_cellid = []
    list_parent = []
    for i in range(tmp_parent.shape[0]):
        tmp = tmp_parent[i, :]
        div_idx = np.where(~np.isnan(tmp))
        div_idx = list(div_idx[0])
        if div_idx != []:
            list_cellid.append(np.unique(tmp_cellid[i, :][~np.isnan(tmp_cellid[i, :])]))
            list_parent.append(tmp[div_idx])
            #print 'cell_id: '+ str(np.unique(tmp_cellid[i,:][~np.isnan(tmp_cellid[i,:])]))# return a certain cell_id which is in currrent column
            #print 'division_timepoint: ' +str(div_idx) # timepoint of division detection
            #print 'parent_id: ' + str(tmp[div_idx]) # parental cell_id
            #div_list.append((tmp[div_idx], div_idx, np.unique(tmp_cellid[i,:][~np.isnan(tmp_cellid[i,:])])))
    seq_cell = []
    for i, j in enumerate(list_parent):
        #print j
        if j in list_cellid:
            #print list_cellid.index(j)
            seq_cell.append([list_parent[list_cellid.index(j)], j, list_cellid[i]])
        else:
            seq_cell.append([j, list_cellid[i]])
    return seq_cell

def mergePlots(site, location, label, seq_cell):
    """Merge intensity information
    site:
    location: 'nuc' or 'cyto'
    label: Gem, SLBP, ...
    seq_cell: [22,165, 398]
    """
    tmp_intensity = site[location, label, 'mean_intensity']
    tmp_parent = site[location, label, 'parent'] # seq_cell=[22,165, 398]
    seq_idx = []
    tmp_cellid = site[location, label, 'cell_id']
    tmp_cellid2 = np.nanmin(tmp_cellid,axis=1)
    for i in seq_cell:
        for (j, k) in enumerate(tmp_cellid2):
            if i == k:
                seq_idx.append(j)
    acell = np.zeros([tmp_parent.shape[1]])
    acell[:] = np.nan
    for l in seq_idx:
        tmp_idx = np.where(~np.isnan(tmp_intensity[l, :]))
        acell[min(tmp_idx[0]):max(tmp_idx[0])] = tmp_intensity[l, :][min(tmp_idx[0]):max(tmp_idx[0])]
    return acell

def plotSingleTrace(seq_cell, sub_folders):
    AktCyt = mergePlots(site, 'cyto', 'Akt-KTR', seq_cell)
    AktNuc = mergePlots(site, 'nuc', 'Akt-KTR', seq_cell)
    ERKCyt = mergePlots(site, 'cyto', 'ERK-KTR', seq_cell)
    ERKNuc = mergePlots(site, 'nuc', 'ERK-KTR', seq_cell)
    Gem = mergePlots(site, 'nuc', 'Gem', seq_cell)
    nGem = Gem.copy()
    nGem = (nGem - np.sort(nGem[~np.isnan(Gem)])[1]) / \
            (np.sort(nGem[~np.isnan(Gem)])[-3]  - np.sort(nGem[~np.isnan(Gem)])[1])
    SLBP = mergePlots(site, 'nuc', 'SLBP', seq_cell)
    nSLBP = SLBP.copy()
    nSLBP = (nSLBP - np.sort(nSLBP[~np.isnan(SLBP)])[1]) / \
            (np.sort(nSLBP[~np.isnan(SLBP)])[-3]  - np.sort(nSLBP[~np.isnan(SLBP)])[1])
    fig, ax1 = plt.subplots()
    ax1.plot(AktCyt/AktNuc)
    ax1.plot(ERKCyt/ERKNuc)
    ax2 = ax1.twinx()
    ax2.plot(nGem, 'r')
    ax2.plot(nSLBP, 'y')
    ax1.set_title(str(sub_folders) + str(seq_cell), fontsize=20)
    ax1.set_xlabel("Timepoints", fontsize=20)
    ax1.set_ylabel("C/N ratio", fontsize=20)
    ax2.set_ylabel("Normalized Intensity", fontsize=20)
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    pos=str(seq_cell).replace('[','')
    pos=pos.replace(']','')
    pos = pos.replace(', ', '_')
    plt.savefig(sub_folders[0]+'_'+pos+'.png')

def min_max(x, axis=None):
    min = np.nanmin(x, axis=axis, keepdims=True)
    max = np.nanmax(x, axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def detect_onset(FP_int, thres, window_length=15, polyorder=5):
    sv = signal.savgol_filter(FP_int, window_length, polyorder)
    mmsv= min_max(sv)
    for i, v in enumerate(mmsv[0]):
        rel = i + 10
        tmp = mmsv[0][i:rel]
        n = np.where( np.array(tmp) <  thres)
        if len(n[0]) > 8and tmp[9] > thres:
            break
    return i, tmp[9]

def detect_peak(FP_int, window_length=15, polyorder=5):
    sg = signal.savgol_filter(FP_int, window_length, polyorder)
    maxindx_sg = signal.argrelmax(sg[0], order=10) 
    return maxindx_sg

def add_intdata(site, id_list, np_dataarray, location, label, type = 0):
    """
    id_list : id list of cell division data
    np_dataarray : data array for quantified data
    location : "nuc" or "cyto"
    label : e.g. H2B, ERK-KTR
    type : 0-> connect all data in id strip, 1 -> get data of daugter cell data after 1st cell division
    """
    pre = 0
    for i in id_list:
        if pre != i[1]:
            if type == 0:
                tmp = mergePlots(site, location, label, i)
                np_dataarray.append(tmp)
            elif type == 1:
                #tmp = site[location, label, 'mean_intensity']
                tmp = mergePlots(site, location, label, [i[1]])
                np_dataarray.append(tmp)
                #print pre, i[1]
                pre = i[1]
    return np_dataarray
