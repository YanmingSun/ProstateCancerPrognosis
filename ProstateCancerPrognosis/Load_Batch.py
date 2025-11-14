from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import Config


class LoadBatch(Dataset):
    def __init__(self, img_t2w, img_hbv, img_adc, clinical_features, reference, repeat=1):

        self.config_job = Config.ConfigProject()

        self.img_t2w = img_t2w
        self.img_hbv = img_hbv
        self.img_adc = img_adc
        self.clinical_features = clinical_features
        self.reference = reference

        self.len = np.shape(self.reference)[0]
        self.repeat = repeat

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))

        temp_t2w = np.zeros((1, self.config_job.shape_t2w[0], self.config_job.shape_t2w[1], self.config_job.shape_t2w[2]), dtype=np.float32)
        temp_hbv = np.zeros((1, self.config_job.shape_hbv_adc[0], self.config_job.shape_hbv_adc[1], self.config_job.shape_hbv_adc[2]), dtype=np.float32)
        temp_adc = np.zeros((1, self.config_job.shape_hbv_adc[0], self.config_job.shape_hbv_adc[1], self.config_job.shape_hbv_adc[2]), dtype=np.float32)

        temp_clinical_features = np.zeros(18, dtype=np.float32)
        temp_reference = np.zeros(2, dtype=np.float32)

        temp_t2w[0, :, :, :] = self.img_t2w[index, 0, :, :, :]
        temp_hbv[0, :, :, :] = self.img_hbv[index, 0, :, :, :]
        temp_adc[0, :, :, :] = self.img_adc[index, 0, :, :, :]
        temp_clinical_features[:] = self.clinical_features[index, :]
        temp_reference[:] = self.reference[index, :]

        temp_t2w, temp_hbv, temp_adc = self.aug_sample(temp_t2w, temp_hbv, temp_adc)

        return temp_t2w.astype(np.float16).copy(), temp_hbv.astype(np.float16).copy(), temp_adc.astype(np.float16).copy(), temp_clinical_features.astype(np.float16).copy(), temp_reference.astype(np.float16).copy()


    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len * self.repeat
        return data_len


    def aug_sample(self, in_t2w, in_hbv, in_adc):
        temp_t2w = np.copy(in_t2w)
        temp_t2w = temp_t2w.astype(np.float32)
        temp_hbv = np.copy(in_hbv)
        temp_hbv = temp_hbv.astype(np.float32)
        temp_adc = np.copy(in_adc)
        temp_adc = temp_adc.astype(np.float32)

        flip_length = random.choice([0, 0, 1])
        if flip_length == 1:
            temp_t2w = np.flip(temp_t2w, axis=1)
            temp_hbv = np.flip(temp_hbv, axis=1)
            temp_adc = np.flip(temp_adc, axis=1)

        flip_high = random.choice([0, 0, 1])
        if flip_high == 1:
            temp_t2w = np.flip(temp_t2w, axis=2)
            temp_hbv = np.flip(temp_hbv, axis=2)
            temp_adc = np.flip(temp_adc, axis=2)

        flip_width = random.choice([0, 0, 1])
        if flip_width == 1:
            temp_t2w = np.flip(temp_t2w, axis=3)
            temp_hbv = np.flip(temp_hbv, axis=3)
            temp_adc = np.flip(temp_adc, axis=3)

        noise = random.choice([0, 0, 1])
        if noise == 1:
            mean_gn = 0.0
            sigma_gn = np.random.uniform(low=0.15, high=0.5)
            temp_t2w = temp_t2w + np.random.normal(loc=mean_gn, scale=sigma_gn, size=np.shape(temp_t2w)).astype(np.float32)
            temp_hbv = temp_hbv + np.random.normal(loc=mean_gn, scale=sigma_gn, size=np.shape(temp_hbv)).astype(np.float32)
            temp_adc = temp_adc + np.random.normal(loc=mean_gn, scale=sigma_gn, size=np.shape(temp_adc)).astype(np.float32)

        return temp_t2w, temp_hbv, temp_adc

