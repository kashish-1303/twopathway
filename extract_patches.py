

# final code
import random
import numpy as np
from glob import glob
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split

class Pipeline(object):
    
    def __init__(self, list_train, Normalize=True):
        self.scans_train = list_train
        self.train_im = self.read_scans(Normalize)
        print("Shape of self.train_im:", self.train_im.shape)
        
    def read_scans(self, Normalize):
        train_im = []

        print("Number of scans to process:", len(self.scans_train))

        for i in range(len(self.scans_train)):
            if i % 10 == 0:
                print('iteration [{}]'.format(i))

            patient_dir = self.scans_train[i]
            
            flair = glob(os.path.join(patient_dir, '*Flair*.mha'))
            t1 = glob(os.path.join(patient_dir, '*T1*.mha'))
            t1c = glob(os.path.join(patient_dir, '*T1c*.mha'))
            t2 = glob(os.path.join(patient_dir, '*T2*.mha'))
            gt = glob(os.path.join(patient_dir, '*more*.mha'))

            t1s = [scan for scan in t1 if scan not in t1c]

            if not all([flair, t1s, t2, gt]):
                print(f"Missing files for patient: {patient_dir}")
                continue

            if len(flair) != 1 or len(t2) != 1 or len(gt) != 1 or len(t1s) != 1 or len(t1c) != 1:
                print(f"Unexpected number of files for patient: {patient_dir}")
                continue

            scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
            
            tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]

            z0, y0, x0 = 1, 29, 42
            z1, y1, x1 = 147, 221, 194
            tmp = np.array(tmp)
            tmp = tmp[:, z0:z1, y0:y1, x0:x1]

            if Normalize:
                tmp = self.norm_slices(tmp)

            train_im.append(tmp)
            del tmp    

        print("Number of successfully processed scans:", len(train_im))
        return np.array(train_im)
    
    def sample_patches_randomly(self, num_patches, d, h, w):
        patches, labels = [], []
        count = 0
        print(self.train_im.shape)

        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        tmp_shp = gt_im.shape

        gt_im = gt_im.reshape(-1).astype(np.uint8)
        msk = msk.reshape(-1).astype(np.float32)

        indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
        del msk

        np.random.shuffle(indices)

        gt_im = gt_im.reshape(tmp_shp)

        i = 0
        pix = len(indices)
        while (count < num_patches) and (pix > i):
            ind = indices[i]
            i += 1
            ind = np.unravel_index(ind, tmp_shp)
            patient_id, slice_idx = ind[0], ind[1]
            p = ind[2:]
            p_y = (p[0] - h//2, p[0] + h//2)
            p_x = (p[1] - w//2, p[1] + w//2)
            p_x = list(map(int, p_x))
            p_y = list(map(int, p_y))
            
            tmp = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
            lbl = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]

            if tmp.shape != (d, h, w):
                continue
            patches.append(tmp)
            labels.append(lbl)
            count += 1
        return np.array(patches), np.array(labels)
        
    def norm_slices(self, slice_not):
        normed_slices = np.zeros((5, 146, 192, 152)).astype(np.float32)
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(146):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        normed_slices[-1] = slice_not[-1]
        return normed_slices    
   
    def _normalize(self, slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    path_HGG = glob(os.path.join(current_dir, 'BRATS2015', 'training', 'HGG', '*'))
    path_LGG = glob(os.path.join(current_dir, 'BRATS2015', 'training', 'LGG', '*'))
    path_all = path_HGG + path_LGG

    np.random.seed(2022)
    np.random.shuffle(path_all)

    np.random.seed(1555)
    start, end = 0, 5
    num_patches = 146 * (end - start) * 3
    h, w, d = 128, 128, 4 

    pipe = Pipeline(list_train=path_all[start:end], Normalize=True)
    Patches, Y_labels = pipe.sample_patches_randomly(num_patches, d, h, w)

    Patches = np.transpose(Patches, (0, 2, 3, 1)).astype(np.float32)

    Y_labels[Y_labels == 4] = 3

    shp = Y_labels.shape[0]
    Y_labels = Y_labels.reshape(-1)
    Y_labels = to_categorical(Y_labels).astype(np.uint8)
    Y_labels = Y_labels.reshape(shp, h, w, 4)

    shuffle = list(zip(Patches, Y_labels))
    np.random.seed(180)
    np.random.shuffle(shuffle)
    Patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    del shuffle
    
    print("Size of the patches : ", Patches.shape)
    print("Size of their corresponding targets : ", Y_labels.shape)

    np.save("x_training", Patches.astype(np.float32))
    np.save("y_training", Y_labels.astype(np.uint8))
