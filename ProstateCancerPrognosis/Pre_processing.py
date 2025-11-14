# import os
import numpy as np
import Config
import datetime
import SimpleITK as sitk
import json
from scipy.ndimage.morphology import binary_dilation


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return_data = data - np.min(data)
    else:
        return_data = (data - np.min(data)) / _range
    return return_data


def normalization_signal(data):
    mask_data = data.copy()
    mask_data[mask_data != 0] = 1

    _range = np.max(data[data != 0]) - np.min(data[data != 0])
    if _range == 0:
        return_data = data - np.min(data[data != 0])
    else:
        return_data = (data - np.min(data[data != 0])) / _range
    return return_data * mask_data


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        return_data = (data - mu)
    else:
        return_data = (data - mu) / sigma
    return return_data


def standardize_signal(data):
    mean_data = np.mean(data[data != 0])
    std_data = np.std(data[data != 0])

    if std_data == 0:
        standard_data = data - mean_data
    else:
        standard_data = (data - mean_data) / std_data

    return standard_data


def read_mha_image(path):
    image_mha = sitk.ReadImage(path)
    image_data = sitk.GetArrayFromImage(image_mha)

    return image_data.astype(np.float32)


def read_patient_images(path_img, patinet_id, shape_t2w, shape_hbv_adc):
    path_t2w = path_img + 'mpMRI/' + patinet_id + '/' + patinet_id + '_0001_t2w.mha'
    path_hbv = path_img + 'mpMRI/' + patinet_id + '/' + patinet_id + '_0001_hbv.mha'
    path_adc = path_img + 'mpMRI/' + patinet_id + '/' + patinet_id + '_0001_adc.mha'
    path_mask_t2w = path_img + 'prostate_mask_t2w/' + patinet_id + '_0001_mask.mha'

    image_data_t2w = read_mha_image(path_t2w)
    image_data_hbv = read_mha_image(path_hbv)
    image_data_adc = read_mha_image(path_adc)
    image_data_mask_t2w = read_mha_image(path_mask_t2w)

    image_data_mask_t2w_exd = np.float32(binary_dilation(image_data_mask_t2w, iterations=3))

    image_data_t2w = standardize_signal(image_data_t2w * image_data_mask_t2w_exd)
    image_data_hbv = standardize_signal(image_data_hbv)
    image_data_adc = standardize_signal(image_data_adc)

    original_shape_t2w = np.shape(image_data_t2w)
    original_shape_hbv_adc = np.shape(image_data_hbv)

    out_t2w = np.zeros((shape_t2w[0], shape_t2w[1], shape_t2w[2]), dtype=np.float32)
    out_hbv = np.zeros((shape_hbv_adc[0], shape_hbv_adc[1], shape_hbv_adc[2]), dtype=np.float32)
    out_adc = np.zeros((shape_hbv_adc[0], shape_hbv_adc[1], shape_hbv_adc[2]), dtype=np.float32)

    out_t2w[0:original_shape_t2w[0], 0:original_shape_t2w[1], 0:original_shape_t2w[2]] = image_data_t2w[:, :, :]
    out_hbv[0:original_shape_hbv_adc[0], 0:original_shape_hbv_adc[1], 0:original_shape_hbv_adc[2]] = image_data_hbv[:, :, :]
    out_adc[0:original_shape_hbv_adc[0], 0:original_shape_hbv_adc[1], 0:original_shape_hbv_adc[2]] = image_data_adc[:, :, :]

    return out_t2w.astype(np.float16), out_hbv.astype(np.float16), out_adc.astype(np.float16)


def read_quantify_clinical_data(path_data, patient_id):
    path_clinical_data = path_data + patient_id + '.json'

    try:
        with open(path_clinical_data, 'r') as file:
            data = json.load(file)
            # print(type(data))  # This will likely be <class 'dict'> or <class 'list'>
    except FileNotFoundError:
        print("Error: The file was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file. Check for malformed JSON.")

    numeric_features = np.zeros(3, dtype=np.float32)  # age_at_prostatectomy, pre_operative_PSA, BCR_PSA,
    ordinal_features = np.zeros(5, dtype=np.float32)  # primary_gleason, secondary_gleason, tertiary_gleason, ISUP, pT_stage
    binary_features = np.zeros(5, dtype=np.float32)  # positive_lymph_nodes, capsular_penetration, positive_surgical_margins, invasion_seminal_vesicles, lymphovascular_invasion
    text_features = np.zeros(int(1 * 5), dtype=np.float32)  # earlier_therapy: None, unknow, radiotherapy, hormones, cryotherapy
    reference = np.zeros(2, dtype=np.float32)  # BCR, time_to_follow-up/BCR

    ############################################################################## numeric_features
    age_at_prostatectomy = data.get('age_at_prostatectomy', 'NotFound')
    if age_at_prostatectomy == 'NotFound' or age_at_prostatectomy == '' or age_at_prostatectomy == ' ':
        numeric_features[0] = 0.0
    else:
        numeric_features[0] = np.float32(age_at_prostatectomy)

    pre_operative_PSA = data.get('pre_operative_PSA', 'NotFound')
    if pre_operative_PSA == 'NotFound' or pre_operative_PSA == '' or pre_operative_PSA == ' ':
        numeric_features[1] = 0.0
    else:
        numeric_features[1] = np.float32(pre_operative_PSA)

    BCR_PSA = data.get('BCR_PSA', 'NotFound')
    if BCR_PSA == 'NotFound' or BCR_PSA == '' or BCR_PSA == ' ':
        numeric_features[2] = 0.0
    else:
        numeric_features[2] = np.float32(BCR_PSA)

    ############################################################################## ordical_features
    primary_gleason = data.get('primary_gleason', 'NotFound')
    if primary_gleason == 'NotFound' or primary_gleason == '' or primary_gleason == ' ':
        ordinal_features[0] = 0.0
    else:
        ordinal_features[0] = np.float32(primary_gleason)

    secondary_gleason = data.get('secondary_gleason', 'NotFound')
    if secondary_gleason == 'NotFound' or secondary_gleason == '' or secondary_gleason == ' ':
        ordinal_features[1] = 0.0
    else:
        ordinal_features[1] = np.float32(secondary_gleason)

    tertiary_gleason = data.get('tertiary_gleason', 'NotFound')
    if tertiary_gleason == 'NotFound' or tertiary_gleason == '' or tertiary_gleason == ' ':
        ordinal_features[2] = 0.0
    else:
        ordinal_features[2] = np.float32(tertiary_gleason)

    ISUP = data.get('ISUP', 'NotFound')
    if ISUP == 'NotFound' or ISUP == '' or ISUP == ' ':
        ordinal_features[3] = 0.0
    else:
        ordinal_features[3] = np.float32(ISUP)

    pT_stage = data.get('pT_stage', 'NotFound')
    if pT_stage == 'NotFound' or pT_stage == '' or pT_stage == ' ':
        ordinal_features[4] = 0.0
    else:
        if len(pT_stage) == 1:
            ordinal_features[4] = np.float32(pT_stage)
        else:
            if pT_stage[1] == 'a':
                ordinal_features[4] = np.float32(pT_stage[0]) + 0.25
            elif pT_stage[1] == 'b':
                ordinal_features[4] = np.float32(pT_stage[0]) + 0.5
            elif pT_stage[1] == 'c':
                ordinal_features[4] = np.float32(pT_stage[0]) + 0.75
            else:
                ordinal_features[4] = np.float32(pT_stage[0])

    ############################################################################## binary_features
    positive_lymph_nodes = data.get('positive_lymph_nodes', 'NotFound')
    if positive_lymph_nodes == '1' or positive_lymph_nodes == '1.0':
        binary_features[0] = 1.0
    elif positive_lymph_nodes == '0' or positive_lymph_nodes == '0.0':
        binary_features[0] = -1.0
    else:
        binary_features[0] = 0.0

    capsular_penetration = data.get('capsular_penetration', 'NotFound')
    if capsular_penetration == '1' or capsular_penetration == '1.0':
        binary_features[1] = 1.0
    elif capsular_penetration == '0' or capsular_penetration == '0.0':
        binary_features[1] = -1.0
    else:
        binary_features[1] = 0.0

    positive_surgical_margins = data.get('positive_surgical_margins', 'NotFound')
    if positive_surgical_margins == 1 or positive_surgical_margins == 1.0:
        binary_features[2] = 1.0
    elif positive_surgical_margins == 0 or positive_surgical_margins == 0.0:
        binary_features[2] = -1.0
    else:
        binary_features[2] = 0.0

    invasion_seminal_vesicles = data.get('invasion_seminal_vesicles', 'NotFound')
    if invasion_seminal_vesicles == '1' or invasion_seminal_vesicles == '1.0':
        binary_features[3] = 1.0
    elif invasion_seminal_vesicles == '0' or invasion_seminal_vesicles == '0.0':
        binary_features[3] = -1.0
    else:
        binary_features[3] = 0.0

    lymphovascular_invasion = data.get('lymphovascular_invasion', 'NotFound')
    if lymphovascular_invasion == '1' or lymphovascular_invasion == '1.0':
        binary_features[4] = 1.0
    elif lymphovascular_invasion == '0' or lymphovascular_invasion == '0.0':
        binary_features[4] = -1.0
    else:
        binary_features[4] = 0.0

    ############################################################################## text_features
    earlier_therapy = data.get('earlier_therapy', 'NotFound')
    if ('NotFound' in earlier_therapy) or ('unknow' in earlier_therapy):
        text_features[0] = 1.0
    if 'none' in earlier_therapy:
        text_features[1] = 1.0
    if 'radiotherapy' in earlier_therapy:
        text_features[2] = 1.0
    if 'hormones' in earlier_therapy:
        text_features[3] = 1.0
    if 'cryotherapy' in earlier_therapy:
        text_features[4] = 1.0

    ############################################################################## reference
    BCR = data.get('BCR', 'NotFound')
    reference[0] = np.float32(BCR)
    time_to_follow_up_BCR = data.get('time_to_follow-up/BCR', 'NotFound')
    reference[1] = np.float32(time_to_follow_up_BCR)

    #################
    #################
    features_all = np.zeros(18, dtype=np.float32)
    features_all[0:3] = numeric_features[:]
    features_all[3:8] = ordinal_features[:]
    features_all[8:13] = binary_features[:]
    features_all[13:18] = text_features[:]

    return features_all.astype(np.float16), reference.astype(np.float16)


class LoadDataset(object):
    def __init__(self): #(self, training_location, roi_location, patient_ID):

        self.config_job = Config.ConfigProject()


    def load_data(self, patients_id):
        # print('Start loading samples:')
        # start_time = datetime.datetime.now()

        num_cases = len(patients_id)

        img_t2w = np.zeros((num_cases, 1, self.config_job.shape_t2w[0], self.config_job.shape_t2w[1], self.config_job.shape_t2w[2]), dtype=np.float16)
        img_hbv = np.zeros((num_cases, 1, self.config_job.shape_hbv_adc[0], self.config_job.shape_hbv_adc[1], self.config_job.shape_hbv_adc[2]), dtype=np.float16)
        img_adc = np.zeros((num_cases, 1, self.config_job.shape_hbv_adc[0], self.config_job.shape_hbv_adc[1], self.config_job.shape_hbv_adc[2]), dtype=np.float16)

        clinical_features = np.zeros((num_cases, 18), dtype=np.float16)
        reference = np.zeros((num_cases, 2), dtype=np.float16)

        for i in range(num_cases):
            img_t2w[i, 0, :, :, :], img_hbv[i, 0, :, :, :], img_adc[i, 0, :, :, :] = read_patient_images(self.config_job.path_mpMRI, patients_id[i], self.config_job.shape_t2w, self.config_job.shape_hbv_adc)
            clinical_features[i, :], reference[i, :] = read_quantify_clinical_data(self.config_job.path_clinical_data, patients_id[i])

        # print('Data-loading finished')
        # end_time = datetime.datetime.now()
        # print('Total time for data loading: ', (end_time - start_time).seconds)

        return img_t2w.astype(np.float16), img_hbv.astype(np.float16), img_adc.astype(np.float16), clinical_features.astype(np.float16), reference.astype(np.float16)




