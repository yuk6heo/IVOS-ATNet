import os
import numpy as np
import cv2

from davisinteractive.utils.operations import bresenham

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


class logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def printNlog(self,str2print):
        print(str2print)
        with open(self.log_file, 'a') as f:
            f.write(str2print + '\n')
            f.close()


def apply_pad(img, padinfo=None):
    if padinfo: # ((hpad,hpad),(wpad,wpad))
        (hpad, wpad) = padinfo
        if len(img.shape)==3 :  pad_img = np.pad(img, (hpad, wpad, (0, 0)), mode='reflect') # H,W,3
        else: pad_img = np.pad(img, (hpad, wpad), mode='reflect') #H,W
        return pad_img
    else:
        h, w = img.shape[0:2]
        new_h = h + 32 - h % 32
        new_w = w + 32 - w % 32
        # print(new_h, new_w)
        lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
        lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        if len(img.shape)==3 :  pad_img = np.pad(img, ((lh, uh), (lw, uw), (0, 0)), mode='reflect') # H,W,3
        else: pad_img = np.pad(img, ((lh, uh), (lw, uw)), mode='reflect') # H,W
        info = ((lh, uh), (lw, uw))

        return pad_img, info


def get_prop_list(annotated_frames, annotated_now, num_frames, proportion = 1.0, get_close_anno_frames = False):

    aligned_anno = sorted(annotated_frames)
    overlap = aligned_anno.count(annotated_now)
    for i in range(overlap):
        aligned_anno.remove(annotated_now)

    start_frame, end_frame = 0, num_frames -1
    for i in range(len(aligned_anno)):
        if aligned_anno[i] > annotated_now:
            end_frame = aligned_anno[i] - 1
            break
    aligned_anno.reverse()
    for i in range(len(aligned_anno)):
        if aligned_anno[i] < annotated_now:
            start_frame = aligned_anno[i]+1
            break

    if get_close_anno_frames:
        close_frames_round=dict() # 1st column: iaction idx, 2nd column: the close frames
        annotated_frames.reverse()
        try: close_frames_round["left"] = len(annotated_frames) - annotated_frames.index(start_frame-1) - 1
        except: print('No left annotated fr')
        try: close_frames_round["right"] = len(annotated_frames) - annotated_frames.index(end_frame) - 1
        except: print('No right annotated fr')

    if proportion != 1.0:
        if start_frame!=0:
            start_frame = annotated_now - int((annotated_now-start_frame)*proportion + 0.5)
        if end_frame != num_frames-1:
            end_frame = annotated_now + int((end_frame - annotated_now) * proportion + 0.5)
    prop_list = list(range(annotated_now,start_frame-1,-1)) + list(range(annotated_now,end_frame+1))
    if len(prop_list)==0:
        prop_list = [annotated_now]

    if not get_close_anno_frames:
        return prop_list

    else:
        return prop_list, close_frames_round


def scribble_to_image(scribbles, currentframe, obj_id, prev_mask, dilation=5,
                      nocare_area=None, bresenhamtf=True, blur=True, singleimg=False, seperate_pos_neg = False):
    """ Make scrible to previous mask shaped numpyfile


    """
    h,w = prev_mask.shape
    regions2exclude_on_maskneg = prev_mask!=obj_id
    mask = np.zeros([h,w])
    mask_neg = np.zeros([h,w])
    if singleimg:
        scribbles=scribbles
    else: scribbles = scribbles[currentframe]

    for scribble in scribbles:
        points_scribble = np.round(np.array(scribble['path']) * np.array((w, h))).astype(np.int)
        if bresenhamtf and len(points_scribble) > 1:
            all_points = bresenham(points_scribble)
        else:
            all_points = points_scribble

        if obj_id==0:
            raise NotImplementedError
        else:
            if scribble['object_id'] == obj_id:
                mask[all_points[:, 1] - 1, all_points[:, 0] - 1] = 1
            else:
                mask_neg[all_points[:, 1] - 1, all_points[:, 0] - 1] = 1
        # else:
        #     mask_neg[all_points[:, 1] - 1, all_points[:, 0] - 1] = 1

    scr_gt, _ = scrimg_postprocess(mask, dilation=dilation, nocare_area=nocare_area, blur=blur, blursize=(5, 5))
    scr_gt_neg, _ = scrimg_postprocess(mask_neg, dilation=dilation, nocare_area=nocare_area, blur=blur, blursize=(5, 5))
    scr_gt_neg[regions2exclude_on_maskneg] = 0

    if seperate_pos_neg:
        return scr_gt.astype(np.float32), scr_gt_neg.astype(np.float32)
    else:
        scr_img = scr_gt - scr_gt_neg
        return scr_img.astype(np.float32)


def scrimg_postprocess(scr, dilation=7, nocare_area=21, blur = False, blursize=(5, 5), var = 6.0, custom_blur = None):

    # Compute foreground
    if scr.max() == 1:
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        fg = cv2.dilate(scr.astype(np.uint8), kernel=kernel_fg).astype(scr.dtype)
    else:
        fg = scr

    # Compute nocare area
    if nocare_area is None:
        nocare = None
    else:
        kernel_nc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nocare_area, nocare_area))
        nocare = cv2.dilate(fg, kernel=kernel_nc) - fg
    if blur:
        fg = cv2.GaussianBlur(fg,ksize=blursize,sigmaX=var)
    elif custom_blur:
        c_kernel = np.array([[1,2,3,2,1],[2,4,9,4,2],[3,9,64,9,3],[2,4,9,4,2],[1,2,3,2,1]])
        c_kernel = c_kernel/np.sum(c_kernel)
        fg = cv2.filter2D(fg,ddepth=-1,kernel = c_kernel)

    return fg, nocare
