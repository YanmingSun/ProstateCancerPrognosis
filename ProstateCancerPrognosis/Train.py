import os
import Pre_processing
import Load_Batch
import Model
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime
import Loss_Function
import Config
import numpy as np
import wandb
from lifelines.utils import concordance_index


if __name__ == "__main__":

    batch_size = 16
    num_epoch = 15
    plot_step = 1
    config_job = Config.ConfigProject()
    branch_name = 'Fold' + str(config_job.validation_fold)

    wandb.login(key='xxxxxxxxxxxxxxxxxxxxxxxxxxx')  # sign in wandb and copy your own API key
    wandb.init(project='ProstateCancerPrognosis_v3_Github', name=branch_name, config={'epochs': num_epoch})  # , config={'learning_rate': 0.001, 'epochs': num_epoch}
    config_wandb = wandb.config  # Access the logged hyperparameters

    print('pytorch version: ', torch.__version__)

    start_time = datetime.datetime.now()
    print('Start time: ', start_time)

    device = torch.device('cuda')

    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder = './CheckPoint_' + branch_name + '/'
    if not os.path.exists(checkpoint_folder):
        try:
            os.makedirs(checkpoint_folder)
        except:
            folder_exist = True

    # training
    #################################################################
    print('Start loading data and training: ')

    training_data_obj = Pre_processing.LoadDataset()
    img_t2w_train, img_hbv_train, img_adc_train, clinical_features_train, reference_train = training_data_obj.load_data(config_job.train_patients_id)
    img_t2w_val, img_hbv_val, img_adc_val, clinical_features_val, reference_val = training_data_obj.load_data(config_job.validation_patients_id)

    total_training_samples = np.shape(reference_train)[0]
    print('Total training samples: ', total_training_samples)

    train_dataset = Load_Batch.LoadBatch(img_t2w_train, img_hbv_train, img_adc_train, clinical_features_train, reference_train, repeat=1)
    train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ####################################################################################################################

    net_train = Model.MultimodalNetwork()
    net_train = net_train.to(device)
    net_train = torch.nn.DataParallel(net_train)
    print('Model:', type(net_train))
    print('Devices:', net_train.device_ids)
    net_train.train()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(net_train.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int((0.5 + total_training_samples / batch_size) * num_epoch), eta_min=0.0)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    scaler = torch.cuda.amp.GradScaler()

    loss_function_cox = Loss_Function.CoxLoss().to(device)

    #########################################################################################

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            tensor_img_t2w_train = torch.tensor(img_t2w_train, device=device)
            tensor_img_hbv_train = torch.tensor(img_hbv_train, device=device)
            tensor_img_adc_train = torch.tensor(img_adc_train, device=device)
            tensor_clinical_features_train = torch.tensor(clinical_features_train, device=device)
            tensor_reference_train = torch.tensor(reference_train, device=device)

            tensor_img_t2w_val = torch.tensor(img_t2w_val, device=device)
            tensor_img_hbv_val = torch.tensor(img_hbv_val, device=device)
            tensor_img_adc_val = torch.tensor(img_adc_val, device=device)
            tensor_clinical_features_val = torch.tensor(clinical_features_val, device=device)
            tensor_reference_val = torch.tensor(reference_val, device=device)

    for epoch in range(num_epoch):
        start = time.time()

        #####################################################
        net_train.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                risk_train = net_train(tensor_img_t2w_train, tensor_img_hbv_train, tensor_img_adc_train, tensor_clinical_features_train)
                loss_train = loss_function_cox(risk_train, tensor_reference_train[:, 1], tensor_reference_train[:, 0])

                risk_val = net_train(tensor_img_t2w_val, tensor_img_hbv_val, tensor_img_adc_val, tensor_clinical_features_val)
                loss_val = loss_function_cox(risk_val, tensor_reference_val[:, 1], tensor_reference_val[:, 0])

        risk_train = risk_train.detach().cpu().numpy()
        risk_train = np.squeeze(risk_train).astype(np.float32)
        c_index_train = concordance_index(reference_train[:, 1], -risk_train, event_observed=reference_train[:, 0])

        risk_val = risk_val.detach().cpu().numpy()
        risk_val = np.squeeze(risk_val).astype(np.float32)
        c_index_val = concordance_index(reference_val[:, 1], -risk_val, event_observed=reference_val[:, 0])

        wandb.log({
            'train_loss': loss_train.item(),
            'train_c_index': c_index_train,
            'val_loss': loss_val.item(),
            'val_c_index': c_index_val,
            'epoch': epoch
        })

        print('epoch: ', epoch, ', train_loss: %.4f' % loss_train.item(), ', train_c_index: %.4f' % c_index_train, ', val_loss: %.4f' % loss_val.item(), ', val_c_index: %.4f' % c_index_val)

        ############################################################

        net_train.train()

        for step, (batch_img_t2w, batch_img_hbv, batch_img_adc, batch_clinical_features, batch_reference) in enumerate(train_dataset_loader):
            batch_img_t2w = batch_img_t2w.to(device)
            batch_img_hbv = batch_img_hbv.to(device)
            batch_img_adc = batch_img_adc.to(device)
            batch_clinical_features = batch_clinical_features.to(device)
            batch_reference = batch_reference.to(device)

            optimizer.zero_grad()
            backward_loss = 'CoxLoss'

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                risk_step = net_train(batch_img_t2w, batch_img_hbv, batch_img_adc, batch_clinical_features)
                loss_step = loss_function_cox(risk_step, batch_reference[:, 1], batch_reference[:, 0])

            scaler.scale(loss_step).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # loss_total.backward()
            # optimizer.step()
            # scheduler.step()

            '''
            if step % plot_step == 0:
                lr_present = optimizer.param_groups[0]['lr']
                print('Epoch: ', epoch, ', Step: ', step, ', LR: ', lr_present, ', CoxLoss: %.4f' % (loss_total.item()))
            '''

        if num_epoch - epoch <= 3:
            # torch.save(net.state_dict(), checkpoint_path + str(epoch) + '.pkl')
            torch.save(net_train.module.state_dict(), checkpoint_folder + 'Model_' + str(epoch) + '.pkl')

        '''
        if epoch <= 5:
            duration = time.time() - start
            print('Training duation: %.4f' % duration)
        '''

    ##########################################################
    net_train.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            risk_train = net_train(tensor_img_t2w_train, tensor_img_hbv_train, tensor_img_adc_train, tensor_clinical_features_train)
            loss_train = loss_function_cox(risk_train, tensor_reference_train[:, 1], tensor_reference_train[:, 0])

            risk_val = net_train(tensor_img_t2w_val, tensor_img_hbv_val, tensor_img_adc_val, tensor_clinical_features_val)
            loss_val = loss_function_cox(risk_val, tensor_reference_val[:, 1], tensor_reference_val[:, 0])

    risk_train = risk_train.detach().cpu().numpy()
    risk_train = np.squeeze(risk_train).astype(np.float32)
    c_index_train = concordance_index(reference_train[:, 1], -risk_train, event_observed=reference_train[:, 0])

    risk_val = risk_val.detach().cpu().numpy()
    risk_val = np.squeeze(risk_val).astype(np.float32)
    c_index_val = concordance_index(reference_val[:, 1], -risk_val, event_observed=reference_val[:, 0])

    wandb.log({
        'train_loss': loss_train.item(),
        'train_c_index': c_index_train,
        'val_loss': loss_val.item(),
        'val_c_index': c_index_val,
        'epoch': num_epoch
    })

    print('epoch: ', num_epoch, ', train_loss: %.4f' % loss_train.item(), ', train_c_index: %.4f' % c_index_train, ', val_loss: %.4f' % loss_val.item(), ', val_c_index: %.4f' % c_index_val)

    ##########################################################

    print('Training finished')
    end_time = datetime.datetime.now()
    print('Start time: ', start_time)
    print('End time: ', end_time)
    print('Total training duration: ', (end_time - start_time).seconds)

    wandb.finish()
