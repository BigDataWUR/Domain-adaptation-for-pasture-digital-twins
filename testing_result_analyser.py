import pickle

with (open('...pk', 'rb')) as openfile:
                
    results_dict = pickle.load(openfile)
    #print(results_dict)
    for tuning_climate in ['Clim7', 'Clim6']:
        for year_run in ['year_run1', 'year_run2', 'year_run3', 'year_run4', 'year_run5']:
            for variability_setup in ['little_weather_little_factorials', 'more_weather_little_factorials', 'little_weather_more_factorials', 'more_weather_more_factorials']:
                
                pretraining_avg_train_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['avg_train_r2']
                pretraining_avg_validation_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['avg_validation_r2']
                pretraining_avg_test_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['avg_test_r2']
                pretraining_avg_test_on_tuning_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['avg_other_clim_test_r2']
                
                pretraining_std_train_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['std_train_r2']
                pretraining_std_validation_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['std_validation_r2']
                pretraining_std_test_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['std_test_r2']
                pretraining_std_test_on_tuning_r2 = results_dict['Clim4'][tuning_climate][year_run][variability_setup]['std_other_clim_test_r2']
                
                tuning_avg_train_r2 = results_dict[tuning_climate][year_run][variability_setup]['avg_train_r2']
                tuning_avg_validation_r2 = results_dict[tuning_climate][year_run][variability_setup]['avg_validation_r2']
                tuning_avg_test_r2 = results_dict[tuning_climate][year_run][variability_setup]['avg_test_r2']
                tuning_avg_test_on_pretraining_r2 = results_dict[tuning_climate][year_run][variability_setup]['avg_other_clim_test_r2']
                
                tuning_std_train_r2 = results_dict[tuning_climate][year_run][variability_setup]['std_train_r2']
                tuning_std_validation_r2 = results_dict[tuning_climate][year_run][variability_setup]['std_validation_r2']
                tuning_std_test_r2 = results_dict[tuning_climate][year_run][variability_setup]['std_test_r2']
                tuning_std_test_on_pretraining_r2 = results_dict[tuning_climate][year_run][variability_setup]['std_other_clim_test_r2']
                
                linear_train_r2 = results_dict[tuning_climate][year_run][variability_setup]['linear_model_train_r2']
                linear_test_r2 = results_dict[tuning_climate][year_run][variability_setup]['linear_model_test_r2']
                
                print('-------------------------------------------')
                print(tuning_climate, year_run, variability_setup)
                print('-------------------------------------------')
                print('Pretraining:', pretraining_avg_train_r2, pretraining_avg_validation_r2, pretraining_avg_test_r2, pretraining_avg_test_on_tuning_r2)
                print('Tuning:', tuning_avg_train_r2, tuning_avg_validation_r2, tuning_avg_test_on_pretraining_r2, tuning_avg_test_r2)
                
                # the below are with stds for the test sets
                print('Pretraining:', round(pretraining_std_train_r2, 2), round(pretraining_std_validation_r2, 2), round(pretraining_std_test_r2, 2), round(pretraining_std_test_on_tuning_r2, 2))
                print('Tuning:', round(tuning_std_train_r2, 2), round(tuning_std_validation_r2, 2), round(tuning_std_test_on_pretraining_r2, 2), round(tuning_std_test_r2, 2))
