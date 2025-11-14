import os
import Pre_processing
import Model
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime
import Config
import numpy as np
from lifelines.utils import concordance_index


def test_validation_set(network):

    device = torch.device('cuda')

    config_job = Config.ConfigProject()

    validation_data_obj = Pre_processing.LoadDataset()
    img_t2w, img_hbv, img_adc, clinical_features, reference = validation_data_obj.load_data(config_job.validation_patients_id)

    num_cases = np.shape(reference)[0]
    # print('Total validation samples: ', num_cases)

    network.to(device)
    network.eval()

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            img_t2w_test_batch_tensor = torch.tensor(img_t2w, device=device)
            img_hbv_test_batch_tensor = torch.tensor(img_hbv, device=device)
            img_adc_test_batch_tensor = torch.tensor(img_adc, device=device)
            clinical_features_batch_tensor = torch.tensor(clinical_features, device=device)

            risk = network(img_t2w_test_batch_tensor, img_hbv_test_batch_tensor, img_adc_test_batch_tensor, clinical_features_batch_tensor)

    risk = risk.detach().cpu().numpy()
    risk = np.squeeze(risk).astype(np.float32)

    c_index = concordance_index(reference[:, 1], -risk, event_observed=reference[:, 0])
    # print('C-index: ', c_index)

    print('########################################')
    print('C-index: ', c_index)
    print('########################################')
    # print('PatientID, Fold, BCR, Time_to_follow-up/BCR, Risk, Median')
    for i in range(num_cases):
        print('PatientID: %d' % np.int16(config_job.validation_patients_id[i]), ', Fold: %d' % config_job.validation_fold, ', BCR: %d' % reference[i, 0],
              ', Time_to_follow-up/BCR: %.1f' % reference[i, 1], ', Proportional_hazards: %.2f' % risk[i])

    print('########################################')

    return c_index, risk


if __name__ == "__main__":

    num_epoch = 15

    print('pytorch version: ', torch.__version__)

    start_time = datetime.datetime.now()
    print('Start time: ', start_time)

    config_job = Config.ConfigProject()
    branch_name = 'Fold' + str(config_job.validation_fold)

    device = torch.device('cuda')

    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

    net_test = Model.MultimodalNetwork()
    net_test.to(device)
    net_test.load_state_dict(torch.load('./CheckPoint_' + branch_name + '/' + 'Model_' + str(num_epoch - 1) + '.pkl', map_location=device))
    net_test.eval()

    c_index, proportional_hazards = test_validation_set(net_test)

    end_time = datetime.datetime.now()
    print('End time: ', end_time)
    print('Total test time: ', (end_time - start_time).seconds)
