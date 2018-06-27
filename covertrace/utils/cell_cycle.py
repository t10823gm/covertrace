import numpy as np

def division_count(celllist, single_idlist, double_idlist):
    """Count a number of cell division
    :param celllist:
    :param single_idlist: assign null vector
    :param double_idlist: assign null vector
    :return: single_idlist, double_idlist
    """
    for i in celllist:
        # single cell division
        if len(i) == 2:
            single_idlist.append(i)

        #double cell division
        elif len(i) == 3:
            double_idlist.append(i)

    return single_idlist, double_idlist


def shift2zero(np_dataarray):
    """
    :param np_dataarray:
    :return: Left-justified numpy_dataarray
    """
    shifted_int=[]
    for j, data in enumerate(np_dataarray):
        tmp_data = np.empty(len(data))
        tmp_data[:] = np.nan
        #tmp = np.isnan(data)
        tmp_ind = np.argwhere(~np.isnan(data))
        if tmp_ind[0][0] + 1 == tmp_ind[1][0] and sum(~np.isnan(data)) > 50:
            tmp_data[0:tmp_ind[-1]-tmp_ind[0]] = data[tmp_ind[0]:tmp_ind[-1]]
            shifted_int.append(tmp_data)
    return shifted_int

def thres_filter(ref_dataarray, thres=0.1, frame_start=36, frame_end=48, *ref_dataarray2):
    """
    Cleaning data by thresholding
    :param ref_dataarray:
    :param thres:
    :param frame_start:
    :param frame_end:
    :param ref_dataarray2:
    :return:
    """
    for m, data  in enumerate(ref_dataarray):
        if np.mean(data[frame_start:frame_end]) > thres:
            for n in ref_dataarray2:
                n[m][:] = np.nan
    return

def detect_onsets(FP_array, thres, window_length = 25, polyorder=5):
    """
    
    """
    index = []
    onset_val = []
    for tmp_strip in FP_array:
        #print(tmp_strip)
        tmp_ind = np.argwhere(~np.isnan(tmp_strip))
        #print(tmp_ind.size)
        if len(tmp_ind) != 0:
            sg = signal.savgol_filter(tmp_strip[tmp_ind[0]:tmp_ind[-1]], window_length, polyorder) 
            mmsv= min_max(sg)
            r_indx = tmp_ind[::-1] #inverted index
            r_sg = mmsv[::-1] # inverted signal
            for i in range(tmp_ind.size):
                tmp = r_sg[i:i+10] # data window for detect change point 
                n = np.where( np.array(tmp) <  thres)
                #print n, tmp
                if len(n[0]) > 8 and tmp[0] < thres:
                    #print tmp, n
                    index.append(r_indx[i])
                    onset_val.append(r_sg[i])
                    break
        else:
            index.append([])
            onset_val.append([])
    return index, onset_val

def set_phase(FP_array, S_start, S_end):
    '''
    
    '''
    g1phase = []
    sphase =[]
    g2mphase =[]
    wholecycle = []
    for i, tmp in enumerate(S_start):
        tmp_FP = FP_array[i]
        tmp_ind = np.argwhere(~np.isnan(tmp_FP))
        if len(tmp_ind) != 0:
            g1phase.append([tmp_ind[0][0], S_start[i][0]])
            sphase.append([S_start[i][0], S_end[i][0]])
            g2mphase.append([S_end[i][0], tmp_ind[-1][0]])
            wholecycle.append([tmp_ind[0][0], tmp_ind[-1][0]])
            #print tmp_ind[0], S_start[i], S_end[i], tmp_ind[-1]
        else:
            g1phase.append([])
            sphase.append([])
            g2mphase.append([])
            wholecycle.append([])
    return g1phase, sphase, g2mphase, wholecycle

def rm_short_trace(ref_dataarray, thres=200, *ref_dataarray2):
    """
    ref_dataarray : referernce
    thres : threshold for discarding data
    ref_dataarray2 : arrays to apply thresholding
    """
    for m, data  in enumerate(ref_dataarray):
        if sum(np.isnan(data)) > thres:
            for n in ref_dataarray2:
                n[m][:] = np.nan
    return 
