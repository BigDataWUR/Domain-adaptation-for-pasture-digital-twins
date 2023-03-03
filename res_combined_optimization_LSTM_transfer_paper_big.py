import torch as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys
import pandas as pd
from early_stopping import EarlyStopping
from helper_functions import get_datetime, plot_training_validation_losses, create_path, columns_to_drop, make_jointplot, get_r2, plot_monthly_residuals
import random

class CustomDataset(Dataset):

    def __init__(self, data_df, timesteps, drop_columns, drop_sims_NRR_leq2):
        if drop_sims_NRR_leq2==True: 
            data_df = data_df[data_df['target_var']>2] #this should move to another place
        self.y_nrr = T.tensor(data_df.drop(drop_columns, axis=1).iloc[:, 2].values)
        self.x_timeseries = T.tensor(data_df.drop(drop_columns, axis=1).iloc[:, 3:-8].values)
        self.x_scalars = T.tensor(data_df.drop(drop_columns, axis=1).iloc[:, -8:-5].values)          
        self.rest = T.tensor(data_df[['SoilFertility', 'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year']].values)
        
        self.timesteps = timesteps
        self.n_samples = data_df.shape[0]
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return T.reshape(self.x_timeseries[index], (self.timesteps, self.x_timeseries[index].shape[0]//self.timesteps)), self.x_scalars[index], self.y_nrr[index], self.rest[index]

class NRRRegressor(T.nn.Module):
    
    def __init__(self, timesteps, timeseries_n, scalars_n, regressor_dropout_rate):
        super().__init__()
        bottleneck_features = 3
        total_inputs = bottleneck_features * timesteps + scalars_n
        self.linear1 = T.nn.Linear(total_inputs, total_inputs)
        self.linear2 = T.nn.Linear(total_inputs, 1)
        self.dropout = T.nn.Dropout(p=regressor_dropout_rate)
        
        T.nn.init.kaiming_uniform_(self.linear1.weight)
        T.nn.init.kaiming_uniform_(self.linear2.weight)
        T.nn.init.ones_(self.linear1.bias)
        T.nn.init.ones_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        
        return x

class Encoder(T.nn.Module):

    def __init__(self, timeseries_n):
        super().__init__()
        self.lstm1 = T.nn.LSTM(input_size = timeseries_n, 
                                hidden_size = int(np.ceil(timeseries_n*0.5)),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
        self.lstm2 = T.nn.LSTM(input_size = int(np.ceil(timeseries_n*0.5)), 
                                hidden_size = int(np.ceil(timeseries_n*0.25)),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)                        

        for lstm in [self.lstm1, self.lstm2]:
            T.nn.init.kaiming_uniform_(lstm.weight_ih_l0)
            T.nn.init.kaiming_uniform_(lstm.weight_hh_l0)
            T.nn.init.ones_(lstm.bias_ih_l0)
            T.nn.init.ones_(lstm.bias_hh_l0)

    def forward(self, x):
        
        #batch size x sequence length x number of features 
        x, (hn, cn) = self.lstm1(x)
        rescon1 = x        
        x, (hn, cn) = self.lstm2(x)
        
        return [x, rescon1]
        
class Decoder(T.nn.Module):

    def __init__(self, timeseries_n):
        super().__init__()
        self.lstm1 = T.nn.LSTM(input_size = int(np.ceil(timeseries_n*0.25)), 
                                hidden_size = int(np.ceil(timeseries_n*0.50)),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
        self.lstm2 = T.nn.LSTM(input_size = int(np.ceil(timeseries_n*0.50)), 
                                hidden_size = timeseries_n,
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
                                
        for lstm in [self.lstm1, self.lstm2]:
            T.nn.init.kaiming_uniform_(lstm.weight_ih_l0)
            T.nn.init.kaiming_uniform_(lstm.weight_hh_l0)
            T.nn.init.ones_(lstm.bias_ih_l0)
            T.nn.init.ones_(lstm.bias_hh_l0)
        
    def forward(self, x, rescon1):
                
        x, (hn, cn) = self.lstm1(x)
        x = x + rescon1
        x = F.leaky_relu(x)
        x, (hn, cn) = self.lstm2(x)
                
        return x

class dual_head_Autoencoder(T.nn.Module):
    
    def __init__(self, timesteps, timeseries_n, scalars_n, regressor_dropout_rate):
        super().__init__()
        self.encoder = Encoder(timeseries_n)
        self.decoder = Decoder(timeseries_n)
        self.regressor = NRRRegressor(timesteps, timeseries_n, scalars_n, regressor_dropout_rate)

    def forward(self, x_timeseries, x_scalars):
        [enc_out, rescon1] = self.encoder(x_timeseries)
        dec_out = self.decoder(enc_out, rescon1)
        reshaped_bottleneck = T.reshape(enc_out, (enc_out.shape[0], enc_out.shape[1]*enc_out.shape[2])) #flatten but keep batches
        regressor_input = T.cat((reshaped_bottleneck, x_scalars), 1)
        
        nrr_out = self.regressor(regressor_input)
        
        return [dec_out, nrr_out]
    
def validate(device, net, dataloader, loss):
    
    validation_losses = np.array([])
        
    with T.no_grad():    
            net.train(mode=False)
            for i, (x_timeseries, x_scalars, y_nrr, rest) in enumerate(dataloader):
                
                x_timeseries = x_timeseries.float().to(device)
                x_scalars = x_scalars.float().to(device)                
                y_nrr = y_nrr.float().to(device)               
                
                dec_out, nrr_out = net(x_timeseries, x_scalars)
                l = loss(dec_out, x_timeseries) + loss(nrr_out, y_nrr.unsqueeze(1))

                validation_losses = np.append(validation_losses, l.cpu().numpy())
                           
    return validation_losses

def train(device, net, train_dataloader, validation_dataloader, num_epochs, lr, early_stopping_patience, model_save_path):
    
    #helper operations
    early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=True, save_path = model_save_path)
    
    avg_train_losses = np.array([])
    avg_valid_losses = np.array([])
    
    #loss
    loss = T.nn.MSELoss()
    
    optimizer = T.optim.AdamW(net.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        
        net.train(mode=True) #this is needed since we have dropout layers and they will be deactivated after validation where we do 'net.train(mode=False)'
        
        training_losses = np.array([]) 
        
        for i, (x_timeseries, x_scalars, y_nrr, rest) in enumerate(train_dataloader):
            
            x_timeseries = x_timeseries.float().to(device)
            x_scalars = x_scalars.float().to(device)
            y_nrr = y_nrr.float().to(device)       
                        
            dec_out, nrr_out = net(x_timeseries, x_scalars)
            l = loss(dec_out, x_timeseries) + loss(nrr_out, y_nrr.unsqueeze(1))

            net.zero_grad()
            l.backward()
            optimizer.step()
            
            training_losses = np.append(training_losses, l.detach().cpu().numpy())
            
        validation_losses = validate(device, net, validation_dataloader, loss)                         
        
        train_loss = np.average(training_losses)
        valid_loss = np.average(validation_losses)
        avg_train_losses = np.append(avg_train_losses, train_loss)
        avg_valid_losses = np.append(avg_valid_losses, valid_loss)
                      
        early_stopper(valid_loss, net, epoch, num_epochs)
        if early_stopper.early_stop == True:
            print('Early stopping triggered')
            break
        
    return net, avg_train_losses, avg_valid_losses    
 
#'type' is used to seperate between training and test, in order to avoid saving all the predictions for the big training dataset 
def test(device, net, dataloader):
    
    df = pd.DataFrame(columns=['SoilFertility', 'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year', 'target_var', 'NRR', 'Residual'])
    results_tensor = T.tensor([])
        
    with T.no_grad():
        
        net.train(mode=False)
        
        errors_dec = np.array([])
        errors_nrr = np.array([])
        
        for i, (x_timeseries, x_scalars, y_nrr, rest) in enumerate(dataloader):
                                
            x_timeseries = x_timeseries.float().to(device)            
            x_scalars = x_scalars.float().to(device)
            y_nrr = y_nrr.float().to(device)       
                              
            dec_pred, nrr_pred = net(x_timeseries, x_scalars)          
            
            squared_error_dec = (dec_pred-x_timeseries)**2
            squared_error_nrr = (nrr_pred-y_nrr)**2
 
            errors_dec = np.append(errors_dec, squared_error_dec.cpu())
            errors_nrr = np.append(errors_nrr, squared_error_nrr.cpu())
                       
            results_tensor = T.cat((results_tensor, 
                                    T.cat((rest, 
                                            y_nrr.unsqueeze(1).cpu(),
                                            nrr_pred.cpu(),
                                            T.abs(y_nrr.unsqueeze(1).cpu()-nrr_pred.cpu())), dim=1)), dim = 0)
            
        error_dec = np.sqrt(errors_dec.mean())
        error_nrr = np.sqrt(errors_nrr.mean())
        
    return [error_dec, error_nrr, pd.concat([df, pd.DataFrame(results_tensor, columns=df.columns).astype('float')], ignore_index=False)]
    

def run(device, training_type, pretraining_climate, runid, seed, variability_setup, year_run, batch_size, lr, epochs, patience, timesteps, timeseries_n, scalars_n, regressor_dropout_rate, train_clim, other_clim, training_path, validation_path, test_path, other_clim_test_path, drop_columns, base_path, drop_sims_NRR_leq2, network_topology, pretrained_model_path, tuning_model_path, pretrained_model_exists):
  
    ############ Data loading ################
    #load dataframes
    training_df = pd.read_csv(training_path)
    validation_df = pd.read_csv(validation_path)
    pretraining_test_df = pd.read_csv(test_path)
    tuning_test_df = pd.read_csv(other_clim_test_path)
    
    #create Dataset objects
    train_dataset = CustomDataset(training_df, timesteps, drop_columns, drop_sims_NRR_leq2)
    validation_dataset = CustomDataset(validation_df, timesteps, drop_columns, drop_sims_NRR_leq2)
    pretraining_test_dataset =  CustomDataset(pretraining_test_df, timesteps, drop_columns, drop_sims_NRR_leq2)
    tuning_test_dataset =  CustomDataset(tuning_test_df, timesteps, drop_columns, drop_sims_NRR_leq2)
    
    #create Dataloader objects
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=3, persistent_workers=True, pin_memory=False)
    validation_dataloader = DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, persistent_workers=True, pin_memory=False)
    pretraining_test_dataloader = DataLoader(dataset = pretraining_test_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, persistent_workers=True, pin_memory=False)
    tuning_test_dataloader = DataLoader(dataset = tuning_test_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, persistent_workers=True, pin_memory=False)
    
    ############ Training ################
    if training_type == 'pretraining':
        #if the model doesn't exist pretrain, else will just want to have its results on the tuning datast of the other climate
        if pretrained_model_exists == False:
            print('* * * P R E T R A I N I N G * * *')        
            #initialize the net
            net = dual_head_Autoencoder(timesteps, timeseries_n, scalars_n, regressor_dropout_rate)
            net.to(device)
            #train
            net, avg_train_losses, avg_valid_losses = train(device, net, train_dataloader, validation_dataloader, epochs, lr, patience, pretrained_model_path)
            #load best network state
            checkpointed_net = dual_head_Autoencoder(timesteps, timeseries_n, scalars_n, regressor_dropout_rate)
            checkpointed_net.load_state_dict(T.load(pretrained_model_path))
            checkpointed_net.to(device)
        else:
            checkpointed_net = dual_head_Autoencoder(timesteps, timeseries_n, scalars_n, regressor_dropout_rate)
            checkpointed_net.load_state_dict(T.load(pretrained_model_path))
            checkpointed_net.to(device)
    else:
        #in case it's 'tuning' or 'hyperparameter tuning'
        print('* * * T U N I N G * * *')
        #initialize the model
        net = dual_head_Autoencoder(timesteps, timeseries_n, scalars_n, regressor_dropout_rate)
        net.load_state_dict(T.load(pretrained_model_path)) #load the pretrained model
        net.to(device)
        
        #train
        net, avg_train_losses, avg_valid_losses = train(device, net, train_dataloader, validation_dataloader, epochs, lr, patience, tuning_model_path)
        #load best network state
        checkpointed_net = dual_head_Autoencoder(timesteps, timeseries_n, scalars_n, regressor_dropout_rate)
        checkpointed_net.load_state_dict(T.load(tuning_model_path))
        checkpointed_net.to(device)
        
    
    ############ Calculate error metrics ################
    training_error_dec, train_rmse, training_predictions_df = test(device, checkpointed_net, train_dataloader)
    print('Checkpointed net train rmse:' + 'decoder ' + str(training_error_dec) + ', regressor ' + str(train_rmse) + ', regressor r2 ' + str(get_r2(training_predictions_df)))
    
    validation_error_dec, validation_rmse, validation_predictions_df = test(device, checkpointed_net, validation_dataloader)
    print('Checkpointed net validation rmse:' + 'decoder ' + str(validation_error_dec) + ', regressor ' + str(validation_rmse) + ', regressor r2 ' + str(get_r2(validation_predictions_df)))
    
    testing_error_dec, test_rmse, testing_pretrain_predictions_df = test(device, checkpointed_net, pretraining_test_dataloader)
    print('Checkpointed net test (on test dataset) rmse:' + 'decoder ' + str(testing_error_dec) + ' , regressor ' + str(test_rmse) + ', regressor r2 ' + str(get_r2(testing_pretrain_predictions_df)))
    
    testing_error_dec, other_clim_test_rmse, testing_tuning_predictions_df = test(device, checkpointed_net, tuning_test_dataloader)
    print('Checkpointed net test (on other clim test dataset) rmse:' + 'decoder ' + str(testing_error_dec) + ' , regressor ' + str(other_clim_test_rmse) + ', regressor r2 ' + str(get_r2(testing_tuning_predictions_df)))
    
    #for plots and predictions we want things to happen only if we pretrain or tune. NOT in hyperparameter tuning
    if training_type != 'hyperparameter_tuning':
    
        ############ Make plots ################
        #define paths, and then create the hierarchies if they don't exist
        joint_plot_path = base_path + 'plots/' + train_clim + '/' + variability_setup + '/' + year_run + '/' + 'jointplots/' + network_topology + '/seed_' + str(seed) + '/' + runid + '/'
        monthly_residuals_path = base_path + 'plots/' + train_clim + '/' + variability_setup + '/' + year_run + '/' + 'monthly_residuals/' + network_topology + '/seed_' + str(seed) + '/' + runid + '/'
        train_val_loss_path = base_path + 'plots/' + train_clim + '/' + variability_setup + '/' + year_run + '/' + 'train_validation_loss/' + network_topology + '/seed_' + str(seed) + '/' + runid + '/'
        for plot_path in [joint_plot_path, monthly_residuals_path, train_val_loss_path]:
            create_path(plot_path)
        
        #for training set, the 'if ... or ...' are to avoid having duplicate plots when Clim1 goes first to clim3 and then clim7 
        if (pretrained_model_exists == False) or (train_clim != pretraining_climate):
            make_jointplot(joint_plot_path, training_predictions_df, 'training_', train_clim)
            plot_monthly_residuals(monthly_residuals_path, training_predictions_df, 'training_', train_clim)
        #for validation set
        if (pretrained_model_exists == False) or (train_clim != pretraining_climate):
            make_jointplot(joint_plot_path, validation_predictions_df, 'validation_', train_clim)
            plot_monthly_residuals(monthly_residuals_path, validation_predictions_df, 'validation_', train_clim)
        #for pretraining test set
        if (pretrained_model_exists == False) or (train_clim != pretraining_climate):
            make_jointplot(joint_plot_path, testing_pretrain_predictions_df, 'testing_on_', train_clim)
            plot_monthly_residuals(monthly_residuals_path, testing_pretrain_predictions_df, 'testing_on_', train_clim)
        #for tuning test set, here we don't need the 'if ... or ...' because we want it to happen every time
        make_jointplot(joint_plot_path, testing_tuning_predictions_df, 'testing_on_', other_clim)
        plot_monthly_residuals(monthly_residuals_path, testing_tuning_predictions_df, 'testing_on_', other_clim)    
        #training validation losses
        #In case where we have a pretrained model and we just changed climate, we don't want to have this plot again
        if (pretrained_model_exists == False) or (train_clim != pretraining_climate):
            plot_training_validation_losses(train_val_loss_path, epochs, avg_train_losses, avg_valid_losses)
        
        ############ Save predictions ################
        #define and create path
        prediction_save_path = base_path + 'predictions/' + train_clim + '/' + variability_setup + '/' + year_run + '/' + network_topology + '/seed_' + str(seed) + '/' + runid + '/'
        create_path(prediction_save_path)
        #save predictions
        validation_predictions_df.to_csv(prediction_save_path + train_clim + '_validation_set_predictions.csv', index = False)
        testing_pretrain_predictions_df.to_csv(prediction_save_path + train_clim + '_test_set_predictions.csv', index = False)
        testing_tuning_predictions_df.to_csv(prediction_save_path + other_clim + '_test_set_predictions.csv', index = False)
    
    ########### Clean Dataloaders and Datasets ################
    #try to clean those references due to some memory related errors 
    del train_dataloader
    del validation_dataloader
    del pretraining_test_dataloader
    del tuning_test_dataloader
    del train_dataset
    del validation_dataset
    del pretraining_test_dataset
    del tuning_test_dataset
    
    return train_rmse, get_r2(training_predictions_df), validation_rmse, get_r2(validation_predictions_df), test_rmse, get_r2(testing_pretrain_predictions_df), other_clim_test_rmse, get_r2(testing_tuning_predictions_df)
    

def train_a_net(training_type, pretraining_climate, runid, config, seed, variability_setup, year_run, train_clim, other_clim, base_path, training_path, validation_path, test_path, other_clim_test_path, pretrained_model_path, tuning_model_path, pretrained_model_exists, network_topology):
    
    #otherwise it shows 'RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility'
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    device =  T.device('cuda' if T.cuda.is_available() else 'cpu')
    T.use_deterministic_algorithms(True)
    T.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    timeseries_n = 9

    train_rmse, train_r2, validation_rmse, validation_r2, test_rmse, test_r2, other_clim_test_rmse, other_clim_test_r2 = \
    run(
        device,
        training_type = training_type,
        pretraining_climate = pretraining_climate,
        runid = runid,
        seed = seed,
        variability_setup = variability_setup,
        year_run = year_run,
        batch_size = config['batch_size'],
        lr = config['lr'],
        epochs = config['epochs'],
        patience = 100,
        timesteps = 28,
        timeseries_n = timeseries_n,
        scalars_n = 3,
        regressor_dropout_rate = config['regressor_dropout_rate'],
        train_clim = train_clim,
        other_clim = other_clim,
        training_path = training_path, 
        validation_path = validation_path,
        test_path = test_path, 
        other_clim_test_path = other_clim_test_path,
        drop_columns = columns_to_drop(pd.read_csv(training_path), timeseries_n),
        drop_sims_NRR_leq2 = True,
        base_path = base_path + '/',
        network_topology = network_topology,
        pretrained_model_path = pretrained_model_path,
        tuning_model_path = tuning_model_path,
        pretrained_model_exists = pretrained_model_exists)
            
    return train_rmse, train_r2, validation_rmse, validation_r2, test_rmse, test_r2, other_clim_test_rmse, other_clim_test_r2

