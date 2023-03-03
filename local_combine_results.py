import pickle
import copy, os, numpy as np

def process_runs_results(results_path, year_runs, variability_setups, pretraining_climate, tuning_climates):
    
    #create combine results dict
    metric_dict = {'train_rmse':[], 'train_r2':[], 'validation_rmse':[], 'validation_r2':[], 'test_rmse':[], 'test_r2':[], 'other_clim_test_rmse':[], 'other_clim_test_r2':[],
      'avg_train_rmse':None, 'avg_train_r2':None, 'avg_validation_rmse':None, 'avg_validation_r2':None, 'avg_test_rmse':None, 'avg_test_r2':None, 'avg_other_clim_test_rmse':None, 'avg_other_clim_test_r2':None,
      'std_train_rmse':None, 'std_train_r2':None, 'std_validation_rmse':None, 'std_validation_r2':None, 'std_test_rmse':None, 'std_test_r2':None, 'std_other_clim_test_rmse':None, 'std_other_clim_test_r2':None, 'linear_model_train_rmse':None,  'linear_model_train_r2':None,  'linear_model_test_rmse':None,  'linear_model_test_r2':None}
    
    variability_dict = dict(zip(variability_setups, [copy.deepcopy(metric_dict) for i in range(len(variability_setups))]))  
    year_run_dict = dict(zip(year_runs, [copy.deepcopy(variability_dict) for i in range(len(year_runs))]))   
    #clim 4 has results for testing in the tuning climates, there has to be a disticiton between those results. This doesn't happen for clims 7 and 6
    pretraining_climate_year_run_dict = dict(zip(tuning_climates, [copy.deepcopy(year_run_dict) for i in range(2)]))
    #now we combine everything
    clim_dict = dict(zip([pretraining_climate, *tuning_climates], [pretraining_climate_year_run_dict, copy.deepcopy(year_run_dict), copy.deepcopy(year_run_dict)]))
    #just renaming to understand easier
    combined_results_dict = clim_dict
    
    #we first combine the intermediate results per clim, other_clim, year_run, variability_setup
    for filename in os.listdir(results_path + '/'):
        #don't pick any .txt files
        if filename.endswith('.pk'):
            with (open(results_path + '/' + filename, 'rb')) as openfile:
                
                #load the pickle content into the file
                intermediate_results = pickle.load(openfile)
                
                for result_dict in intermediate_results:
                    #assign some variables here to make it more readable later
                    
                    clim = result_dict['results_clim']
                    other_clim  = result_dict['other_clim']
                    year_run = result_dict['year_run']
                    variability_setup = result_dict['variability_setup']               
                    
                    #then we have a result dict from a linear model
                    if result_dict['seed'] == None:
                        for metric in ['train_rmse', 'train_r2', 'test_rmse', 'test_r2']: 
                            rounded_metric_value = round(result_dict['results'][metric], 3)
                            combined_results_dict[clim][year_run][variability_setup]['linear_model_' + metric] = rounded_metric_value                           
                    #then that's a result dict from a NN
                    else:
                        for metric in ['train_rmse', 'train_r2', 'validation_rmse', 'validation_r2', 'test_rmse', 'test_r2', 'other_clim_test_rmse', 'other_clim_test_r2']:                   
                            rounded_metric_value = round(result_dict['results'][metric], 3)
                            
                            #clim 4 and 7,6 need different handling. The reason is that for the pretraining climate we need a seperation of results of other_clim because it can be for either 7 or 6
                            if clim == tuning_climates[0] or clim == tuning_climates[1]:                       
                                combined_results_dict[clim][year_run][variability_setup][metric] = combined_results_dict[clim][year_run][variability_setup][metric]+[rounded_metric_value]
                            else:
                                combined_results_dict[clim][other_clim][year_run][variability_setup][metric] = combined_results_dict[clim][other_clim][year_run][variability_setup][metric]+[rounded_metric_value]
    
    #then we calculate the averages and stds
    for clim in [pretraining_climate, *tuning_climates]:
        for year_run in year_runs:
            for variability_setup in variability_setups:
                for metric in ['train_rmse', 'train_r2', 'validation_rmse', 'validation_r2', 'test_rmse', 'test_r2', 'other_clim_test_rmse', 'other_clim_test_r2']:
                    
                    #again we need a different handling for clims 4 and 7,6
                    if clim == tuning_climates[0] or clim == tuning_climates[1]:
                        rounded_metric_avg_value = round(np.average(combined_results_dict[clim][year_run][variability_setup][metric]), 3)
                        rounded_metric_std_value = round(np.std(combined_results_dict[clim][year_run][variability_setup][metric]), 3)
                        combined_results_dict[clim][year_run][variability_setup]['avg_' + metric] = rounded_metric_avg_value
                        combined_results_dict[clim][year_run][variability_setup]['std_' + metric] = rounded_metric_std_value
                    else:
                        #we need this loop to calculate avgs and stds for both other_clims of clim 4
                        for other_clim in tuning_climates:
                            rounded_metric_avg_value = round(np.average(combined_results_dict[clim][other_clim][year_run][variability_setup][metric]), 3)
                            rounded_metric_std_value = round(np.std(combined_results_dict[clim][other_clim][year_run][variability_setup][metric]), 3)
                            combined_results_dict[clim][other_clim][year_run][variability_setup]['avg_' + metric] = rounded_metric_avg_value
                            combined_results_dict[clim][other_clim][year_run][variability_setup]['std_' + metric] = rounded_metric_std_value
    
    return combined_results_dict

year_runs = ['year_run1', 'year_run2', 'year_run3', 'year_run4', 'year_run5']
variability_setups = ['little_weather_little_factorials', 'more_weather_little_factorials', 'little_weather_more_factorials', 'more_weather_more_factorials']
    
combined_results = process_runs_results('...', year_runs, variability_setups, 'Clim4', ['Clim7', 'Clim6'])

with open('...pk', 'wb') as fh:
        pickle.dump(combined_results, fh) 
    