import numpy as np


class ConfigProject:
    def __init__(self):
        self.validation_fold = 0  # five-fold cross-validation: 0, 1, 2, 3, 4

        self.path_mpMRI = 'C:/Users/XXXXXX/Desktop/M31_Project/Codes/radiology/'
        self.path_clinical_data = 'C:/Users/XXXXXX/Desktop/M31_Project/Codes/clinical_data/'

        patients_id = np.loadtxt('Patient_ID_5Fold.txt').astype(np.int16)
        # self.num_all_cases = np.shape(patients_id)[0]

        self.train_patients_id = []
        self.validation_patients_id = []

        for i in range(np.shape(patients_id)[0]):
            if patients_id[i, 1] == self.validation_fold:
                self.validation_patients_id.append(str(patients_id[i, 0]))
            else:
                self.train_patients_id.append(str(patients_id[i, 0]))

        # self.num_train_cases = len(self.train_patients_id)
        # self.num_validation_cases = len(self.validation_patients_id)

        self.shape_t2w = [26, 640, 640]
        self.shape_hbv_adc = [26, 128, 120]

