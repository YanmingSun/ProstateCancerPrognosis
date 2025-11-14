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


def predict_median_time_from_cox(risk, time, event):
    '''
    Input:
        risk: np.array, Cox predict score (x_i^T beta)
        time: np.array,
        event: np.array, state (1=recurrence, 0=censored)
    Output:
        predicted_median: list, median recurrent time
    '''
    # 1. sort

    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    risk_sorted = risk[order]

    # 2. calculate Breslow baseline cumulatve hazards
    H0 = np.zeros_like(time_sorted, dtype=float)
    for i, t in enumerate(time_sorted):
        at_risk = np.where(time_sorted >= t)[0]
        if np.sum(np.exp(risk_sorted[at_risk])) == 0:
            H0[i] = 0
        else:
            H0[i] = event_sorted[i] / np.sum(np.exp(risk_sorted[at_risk]))
    H0 = np.cumsum(H0)  # cumulative hazards

    # 3. baseline recurrent function
    S0 = np.exp(-H0)

    # 4. predict case-wise median recurrent time
    predicted_median = []
    for r in risk:
        S_target = 0.5 ** (1/np.exp(r + 1.0))
        # find the nearest S0
        idx = np.argmin(np.abs(S0 - S_target))
        predicted_median.append(time_sorted[idx])

    return np.array(predicted_median, dtype=np.float32)


def inference_overall(network0, network1, network2, network3, network4):

    device = torch.device('cuda')

    patients_id = np.loadtxt('Patient_ID_5Fold.txt').astype(np.int16)
    num_all_cases = np.shape(patients_id)[0]

    load_data_obj = Pre_processing.LoadDataset()

    out_data_cases = np.zeros((num_all_cases, 6), dtype=np.float32)
    out_data_cases[:, 0:2] = np.copy(patients_id[:, :]).astype(np.float32)

    for i in range(num_all_cases):
        current_case = [str(patients_id[i, 0])]
        img_t2w, img_hbv, img_adc, clinical_features, reference = load_data_obj.load_data(current_case)
        out_data_cases[i, 2:4] = reference[0, :]

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                img_t2w_test_batch_tensor = torch.tensor(img_t2w, device=device)
                img_hbv_test_batch_tensor = torch.tensor(img_hbv, device=device)
                img_adc_test_batch_tensor = torch.tensor(img_adc, device=device)
                clinical_features_batch_tensor = torch.tensor(clinical_features, device=device)

                if patients_id[i, 1] == 0:
                    risk = network0(img_t2w_test_batch_tensor, img_hbv_test_batch_tensor, img_adc_test_batch_tensor, clinical_features_batch_tensor)
                elif patients_id[i, 1] == 1:
                    risk = network1(img_t2w_test_batch_tensor, img_hbv_test_batch_tensor, img_adc_test_batch_tensor, clinical_features_batch_tensor)
                elif patients_id[i, 1] == 2:
                    risk = network2(img_t2w_test_batch_tensor, img_hbv_test_batch_tensor, img_adc_test_batch_tensor, clinical_features_batch_tensor)
                elif patients_id[i, 1] == 3:
                    risk = network3(img_t2w_test_batch_tensor, img_hbv_test_batch_tensor, img_adc_test_batch_tensor, clinical_features_batch_tensor)
                elif patients_id[i, 1] == 4:
                    risk = network4(img_t2w_test_batch_tensor, img_hbv_test_batch_tensor, img_adc_test_batch_tensor, clinical_features_batch_tensor)

        risk = risk.detach().cpu().numpy()
        risk = np.squeeze(risk).astype(np.float32)
        out_data_cases[i, 4] = risk

    c_index_risk = concordance_index(out_data_cases[:, 3], -out_data_cases[:, 4], event_observed=out_data_cases[:, 2])
    # print('C-index: ', c_index)

    out_data_cases[:, 5] = predict_median_time_from_cox(risk=out_data_cases[:, 4], time=out_data_cases[:, 3], event=out_data_cases[:, 2])

    c_index_time = concordance_index(out_data_cases[:, 3], out_data_cases[:, 5], event_observed=out_data_cases[:, 2])
    # print('C-index-time: ', c_index_time)

    return c_index_risk, c_index_time, out_data_cases


def load_model(fold, epoch):
    device = torch.device('cuda')

    net_test = Model.MultimodalNetwork()
    net_test.to(device)
    net_test.load_state_dict(torch.load('./CheckPoint_Fold' + str(fold) + '/' + 'Model_' + str(epoch - 1) + '.pkl', map_location=device))
    net_test.eval()

    return net_test


if __name__ == "__main__":

    print('pytorch version: ', torch.__version__)

    start_time = datetime.datetime.now()
    print('Start time: ', start_time)

    ##################################################
    num_epoch = 15

    net_0 = load_model(0, num_epoch)
    net_1 = load_model(1, num_epoch)
    net_2 = load_model(2, num_epoch)
    net_3 = load_model(3, num_epoch)
    net_4 = load_model(4, num_epoch)

    c_index_risk, c_index_time, out_data_cases = inference_overall(net_0, net_1, net_2, net_3, net_4)

    print('########################################')
    print('C-index Proportional_Hazards: ', c_index_risk)
    print('C-index Median_Recurrent_Time: ', c_index_time)
    print('########################################')
    # print('PatientID, Fold, BCR, Time_to_follow-up/BCR, Risk, Median')
    for i in range(np.shape(out_data_cases)[0]):
        print('PatientID: %d' % out_data_cases[i, 0], ', Fold: %d' % out_data_cases[i, 1], ', BCR: %d' % out_data_cases[i, 2],
              ', Time_to_follow-up/BCR: %.1f' % out_data_cases[i, 3], ', Proportional_Hazards: %.2f' % out_data_cases[i, 4], ', Median_Recurrent_Time: %.1f' % out_data_cases[i, 5])

    np.savetxt('inference_case_wise.txt', out_data_cases)
    ##################################################

    end_time = datetime.datetime.now()
    print('End time: ', end_time)
    print('Total test time: ', (end_time - start_time).seconds)
