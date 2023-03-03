import pickle
from helper_functions import create_path, get_datetime, create_result_dict, process_runs_results
from postprocess_standardize_fert_soilwater_for_lstm import postprocess
from res_combined_optimization_LSTM_transfer_paper_big import train_a_net
from os.path import exists
from linear_model import train_linear_model
from torch.multiprocessing import set_start_method, set_sharing_strategy

#the if name main is to avoid an error about spawing many processes with the dataloaders
if __name__ == '__main__':
    
    #this is to prevent OSError: [Errno 24] Too many open files
    set_sharing_strategy('file_system')
    #this is to prevent dataloaders from randomly freezing during training
    set_start_method('spawn')

    runid = input('Give a runid (experiment identifier): \n')

    base_path = '...' #contains a directorry 'input' with the training data, it is also the place where the results will be saved
    results_path = base_path + '/results/' + runid + '/' + get_datetime()
    create_path(results_path)

    pretraining_climate = 'Clim4'
    tuning_climates = ['Clim7', 'Clim6']
    year_runs = ['year_run1', 'year_run2', 'year_run3', 'year_run4', 'year_run5']
    variability_setups = ['little_weather_little_factorials', 'more_weather_little_factorials', 'little_weather_more_factorials', 'more_weather_more_factorials']
    seeds = [63, 3, 1000, 1001, 5] 
    network_topology = 'res_autoencoder_multitask_adamw_fert_soilwater_transfer_1'
    
    do_postprocessing = False
    
    total_runs = len(year_runs) * len(variability_setups) * len(seeds) * len(tuning_climates)
    current_run = 1
    
    for tuning_climate in tuning_climates:
        for year_run in year_runs:

            #we store the results per year run
            intermediate_results = []
                    
            for variability_setup in variability_setups:
                
                #this is to check progression from the console
                print('Now doing', tuning_climate, year_run, variability_setup)
                
                ######## Standardization ########
                if do_postprocessing:
                    #postprocessing paths
                    pretraining_postprocessing_path = base_path + '/input/' + pretraining_climate + '/' + variability_setup + '/' + year_run           
                    tuning_postprocessing_path = base_path + '/input/' + tuning_climate + '/' + variability_setup + '/' + year_run           
                    
                    
                    #for pretraining
                    pretraining_training_data_scaler_path = postprocess(pretraining_climate, pretraining_postprocessing_path,  None, 'train.csv', 'validation.csv', 'test.csv', 'pretraining', variability_setup)
                    #for tuning
                    postprocess(tuning_climate, tuning_postprocessing_path, pretraining_training_data_scaler_path, 'train.csv', 'validation.csv', 'test.csv', 'tuning', variability_setup)
                
                ######## Define training paths ########
                
                #these paths do not need to have the seed number in the hierarchy, that's why they are out of the seed loop
                #for pretraining
                pretraining_training_path = base_path + '/input/' + pretraining_climate + '/' + variability_setup + '/' + year_run + '/' + 'train_standardized_fert_soilwater.csv'
                pretraining_validation_path = base_path + '/input/' + pretraining_climate + '/' + variability_setup + '/' + year_run + '/' + 'validation_standardized_fert_soilwater.csv'
                pretraining_test_path = base_path + '/input/' + pretraining_climate + '/' + variability_setup + '/' + year_run + '/' + 'test_standardized_fert_soilwater.csv'
                
                #for tuning
                tuning_training_path = base_path + '/input/' + tuning_climate + '/' + variability_setup + '/' + year_run + '/' + 'train_standardized_fert_soilwater.csv'
                tuning_validation_path = base_path + '/input/' + tuning_climate + '/' + variability_setup + '/' + year_run + '/' + 'validation_standardized_fert_soilwater.csv'
                tuning_test_path = base_path + '/input/' + tuning_climate + '/' + variability_setup + '/' + year_run + '/' + 'test_standardized_fert_soilwater.csv'
                              
                for seed in seeds:
                    
                    print('****************************************************************')
                    print('******* ' + str(current_run) + '/' + str(total_runs), tuning_climate, year_run, variability_setup, str(seed) + ' *******')
                    print('****************************************************************')
                    current_run += 1
                    
                    ####### Define model saving paths ####### 
                    #define paths
                    pretrained_model_path = base_path + '/models/' + pretraining_climate + '/' + variability_setup + '/' + year_run + '/' + network_topology + '/seed_' + str(seed) + '/' + runid + '/' + 'model.pt'
                    tuning_model_path = base_path + '/models/' + tuning_climate + '/' + variability_setup + '/' + year_run + '/' + network_topology + '/seed_' + str(seed) + '/' + runid + '/' + 'model.pt'
                    #we have defined the paths (not created them), now we check if the files exist, and after that we will create them anyway
                    pretrained_model_exists = exists(pretrained_model_path)
                    
                    #no matter if they exist or not, try to create those paths
                    #-8 to remove 'model.pt'. We remove it because this function creates path hierachies not the files themselves
                    create_path(pretrained_model_path[:-8]) 
                    create_path(tuning_model_path[:-8])
                    
                        
                    ####### Train #######
                    #run pretraining here, each pretrained model is trained for one seed and from the resulting model a tuned model comes out. So pretrained and tuned models are 1 to 1
                    #pretraining
                    pretraining_results = train_a_net('pretraining', pretraining_climate, runid, {'batch_size':64, 'lr':0.00004, 'epochs':60, 'regressor_dropout_rate':0.5}, seed, variability_setup, year_run, pretraining_climate, tuning_climate, base_path, pretraining_training_path, pretraining_validation_path, pretraining_test_path, tuning_test_path, pretrained_model_path, None, pretrained_model_exists, network_topology)
                                                         
                    intermediate_results.append(create_result_dict(variability_setup, year_run, seed, pretraining_climate, tuning_climate, pretraining_results))
                    
                    #hyperparameter tuning for tuned model
                    best_validation_r2 = -999
                    best_hyperparameters = None
                    
                    for batch_size in [2, 72]:
                        for lr in [0.00004, 0.0001]:
                            for epochs in [7, 15, 30]:
                                hyperparameter_tuning_results = train_a_net('hyperparameter_tuning', pretraining_climate, runid, {'batch_size':batch_size, 'lr':lr, 'epochs':epochs, 'regressor_dropout_rate':0}, seed, variability_setup, year_run, tuning_climate, pretraining_climate, base_path, tuning_training_path, tuning_validation_path, tuning_test_path, pretraining_test_path, pretrained_model_path, tuning_model_path, None, network_topology)
                                
                                if hyperparameter_tuning_results[3] > best_validation_r2:
                                    best_hyperparameters = {'batch_size':batch_size, 'lr':lr, 'epochs':epochs, 'regressor_dropout_rate':0}
                                    best_validation_r2 = hyperparameter_tuning_results[3]
                                
                    #tuning
                    tuning_results = train_a_net('tuning', pretraining_climate, runid, best_hyperparameters, seed, variability_setup, year_run, tuning_climate, pretraining_climate, base_path, tuning_training_path, tuning_validation_path, tuning_test_path, pretraining_test_path, pretrained_model_path, tuning_model_path, None, network_topology)
                    
                    intermediate_results.append(create_result_dict(variability_setup, year_run, seed, tuning_climate, pretraining_climate, tuning_results))                           
                #linear model training
                linear_model_results = train_linear_model(tuning_climate, runid, variability_setup, year_run, base_path, tuning_training_path, tuning_test_path)
                intermediate_results.append(create_result_dict(variability_setup, year_run, None, tuning_climate, None, linear_model_results))
                          
            #save intermediate results
            with open(results_path + '/' + tuning_climate + '_' + year_run + '_' + variability_setup + '.pk', 'wb') as fh:
                pickle.dump(intermediate_results, fh)
    
    #create and save combined results dict
    combined_results = process_runs_results(results_path, year_runs, variability_setups, pretraining_climate, tuning_climates)
    print(combined_results)
    with open(results_path + '/combined_results.pk', 'wb') as fh:
        pickle.dump(combined_results, fh)            
        
        
        