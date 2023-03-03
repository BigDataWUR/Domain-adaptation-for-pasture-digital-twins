import os
import matplotlib    
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime    
import zipfile
import numpy as np
import time
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import copy

def columns_to_drop(df, timeseries_n):
    
    cols = []
    cols.extend(['Irrigation', 'SoilFertility', 'SoilWater'])
    
    df_columns = list(df.columns)
    
    if timeseries_n == 15:
        pass
    elif timeseries_n == 9:    
        for c in df_columns:
            if ('Wind' in c) or ('VP' in c) or ('PET' in c) or ('RH' in c) or ('SoilTemp300' in c) or ('SoilWater300' in c):
                cols.append(c)
    else:
        raise("The timeseries-columns to drop don't match a known scenario")
    return cols

def get_dfs_from_clims(base_path, train_clims, validation_clims, test_clim, filename_train, filename_validate, filename_test, irrigation=None):
    
    training_paths = [base_path + 'input/' + clim + '/' + filename_train for clim in train_clims]
    validation_paths = [base_path + 'input/' + clim + '/' + filename_validate for clim in validation_clims]
    test_path = base_path + 'input/' + test_clim + '/' + filename_test

    print('Training paths:', training_paths)
    print('Validation paths:', validation_paths)
    print('Test path:', test_path)
    
    if irrigation=='irrigated':
        training_df = pd.concat(pd.read_csv(f) for f in training_paths).loc[lambda df: df.Irrigation == 1]
        validation_df = pd.concat(pd.read_csv(f) for f in validation_paths).loc[lambda df: df.Irrigation == 1]
        test_df = pd.read_csv(test_path).loc[lambda df: df.Irrigation == 1]
    elif irrigation=='non_irrigated':
        training_df = pd.concat(pd.read_csv(f) for f in training_paths).loc[lambda df: df.Irrigation == 0]
        validation_df = pd.concat(pd.read_csv(f) for f in validation_paths).loc[lambda df: df.Irrigation == 0]
        test_df = pd.read_csv(test_path).loc[lambda df: df.Irrigation == 0]
    else:
        #both irrigated and non-irrigated
        training_df = pd.concat(pd.read_csv(f) for f in training_paths)
        validation_df = pd.concat(pd.read_csv(f) for f in validation_paths)
        test_df = pd.read_csv(test_path)
    
    return training_df, validation_df, test_df

def get_df_from_clims(base_path, clims, filename, irrigation=None):
    
    if isinstance(clims, str):
        paths = [base_path + 'input/' + clims + '/' + filename]
    else:
        paths = [base_path + 'input/' + clim + '/' + filename for clim in clims]

    #print('Paths:', paths)

    if irrigation=='irrigated':
        df = pd.concat(pd.read_csv(f) for f in paths).loc[lambda df: df.Irrigation == 1]
    elif irrigation=='non_irrigated':
        df = pd.concat(pd.read_csv(f) for f in paths).loc[lambda df: df.Irrigation == 0]
    else:
        #both irrigated and non-irrigated
        df = pd.concat(pd.read_csv(f) for f in paths)

    return df

def create_path(path):  
    try:
        os.makedirs(path)
    except:
        print('create_path -> ' + path + ' exists')  

def save_errors(scenario_id, errors):

        with open(scenario_id + '.txt', 'w') as f:
            f.write('\n'.join(errors))

def get_datetime():

    date = datetime.datetime.now().strftime('%I:%M%p on %B %d, %Y')
    date = date.replace(':', '')
    
    return date

def create_result_dict(variability_setup, year_run, seed, results_clim, other_clim, results):
    
    result_dict = {'variability_setup':variability_setup, 'year_run':year_run, 'seed':seed, 'results_clim':results_clim, 'other_clim':other_clim, 'results':{'train_rmse':results[0], 'train_r2':results[1], 'validation_rmse':results[2], 'validation_r2':results[3], 'test_rmse':results[4], 'test_r2':results[5], 'other_clim_test_rmse':results[6], 'other_clim_test_r2':results[7]}}
        
    return result_dict

def process_runs_results(results_path, year_runs, variability_setups, pretraining_climate, tuning_climates):
    
    #create combine results dict
    metric_dict = {'train_rmse':[], 'train_r2':[], 'validation_rmse':[], 'validation_r2':[], 'test_rmse':[], 'test_r2':[], 'other_clim_test_rmse':[], 'other_clim_test_r2':[],
      'avg_train_rmse':None, 'avg_train_r2':None, 'avg_validation_rmse':None, 'avg_validation_r2':None, 'avg_test_rmse':None, 'avg_test_r2':None, 'avg_other_clim_test_rmse':None, 'avg_other_clim_test_r2':None,
      'std_train_rmse':None, 'std_train_r2':None, 'std_validation_rmse':None, 'std_validation_r2':None, 'std_test_rmse':None, 'std_test_r2':None, 'std_other_clim_test_rmse':None, 'std_other_clim_test_r2':None, 'linear_model_train_rmse':None,  'linear_model_train_r2':None,  'linear_model_test_rmse':None,  'linear_model_test_r2':None}
    
    variability_dict = dict(zip(variability_setups, [copy.deepcopy(metric_dict) for i in range(len(variability_setups))]))  
    year_run_dict = dict(zip(year_runs, [copy.deepcopy(variability_dict) for i in range(len(year_runs))]))   
    #clim one has results for testing in the tuning climates, there has to be a disticiton between those results. This doesn't happen for clim3 and 7
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
                            
                            #clim 1 and 3,7 need different handling. The reason is that for the pretraining climate we need a seperation of results of other_clim because it can be for either 3 or 7
                            if clim == tuning_climates[0] or clim == tuning_climates[1]:                       
                                combined_results_dict[clim][year_run][variability_setup][metric] = combined_results_dict[clim][year_run][variability_setup][metric]+[rounded_metric_value]
                            else:
                                combined_results_dict[clim][other_clim][year_run][variability_setup][metric] = combined_results_dict[clim][other_clim][year_run][variability_setup][metric]+[rounded_metric_value]
    
    #then we calculate the averages and stds
    for clim in [pretraining_climate, *tuning_climates]:
        for year_run in year_runs:
            for variability_setup in variability_setups:
                for metric in ['train_rmse', 'train_r2', 'validation_rmse', 'validation_r2', 'test_rmse', 'test_r2', 'other_clim_test_rmse', 'other_clim_test_r2']:
                    
                    #again we need a different handling for clims 1 and 3,7
                    if clim == tuning_climates[0] or clim == tuning_climates[1]:
                        rounded_metric_avg_value = round(np.average(combined_results_dict[clim][year_run][variability_setup][metric]), 3)
                        rounded_metric_std_value = round(np.std(combined_results_dict[clim][year_run][variability_setup][metric]), 3)
                        combined_results_dict[clim][year_run][variability_setup]['avg_' + metric] = rounded_metric_avg_value
                        combined_results_dict[clim][year_run][variability_setup]['std_' + metric] = rounded_metric_std_value
                    else:
                        #we need this loop to calculate avgs and stds for both other_clims of clim 1
                        for other_clim in tuning_climates:
                            rounded_metric_avg_value = round(np.average(combined_results_dict[clim][other_clim][year_run][variability_setup][metric]), 3)
                            rounded_metric_std_value = round(np.std(combined_results_dict[clim][other_clim][year_run][variability_setup][metric]), 3)
                            combined_results_dict[clim][other_clim][year_run][variability_setup]['avg_' + metric] = rounded_metric_avg_value
                            combined_results_dict[clim][other_clim][year_run][variability_setup]['std_' + metric] = rounded_metric_std_value
    
    return combined_results_dict

def get_r2(df):
    return r2_score(df['target_var'], df['NRR'])

def plot_distribution_difference(save_path, pretraining_df, tuning_df, type):
    
    plt.figure(figsize=(10,8))
    plt.hist(pretraining_df, bins=35, alpha=0.5, label='pretraining data')
    plt.hist(tuning_df, bins=35, alpha=0.5, label='tuning data')
    
    plt.xlabel('Nitrogen response rate')
    plt.ylabel('Sample count')
    plt.xticks(np.arange(0, 40, 5))
    plt.legend(loc='upper right')
    plt.savefig(save_path + type + '_distributions_' + get_datetime() + '.png')
    plt.close('all')

def plot_training_validation_losses(save_path, num_epochs, avg_train_losses, avg_valid_losses):
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
    ax.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses, label='Validation Loss')
    
    # find epoch of lowest validation loss
    minposs_validation = np.where(avg_valid_losses == min(avg_valid_losses))[0][0]+1 #find the index where the loss is minimum
    ax.axvline(minposs_validation, linestyle='--', color='r',label='Min validation loss')
    ax.axhline(min(avg_valid_losses), linestyle='--', color='r')
    
    # find epoch of lowest training loss 
    minposs_train = np.where(avg_train_losses == min(avg_train_losses))[0][0]+1 #find the index where the loss is minimum
    ax.axvline(minposs_train, linestyle='--', color='g',label='Min training loss')
    ax.axhline(min(avg_train_losses), linestyle='--', color='g')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    
    #plt.xlim(0, num_epochs)
    #plt.ylim(0.2, 0.5)
    #plt.grid(True)
    
    #y_ticks = np.append(ax.get_yticks(), [min(avg_train_losses), min(avg_valid_losses)])
    #ax.set_yticks(y_ticks)
    #plt.xticks(np.arange(0, num_epochs + 1, 20))
    #x_ticks = np.append(ax.get_xticks(), [minposs_train, minposs_validation])
    #ax.set_xticks(x_ticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '_training_validation_losses_' + get_datetime() + '.png')
    plt.close('all')
    
def make_jointplot(save_path, df, set_name, clim):

    plt.clf()
    axis_min = 5
    axis_max = 25

    df = df.astype(np.float64)
    #print('df size is', df.shape[0])
    plot = sns.jointplot(x='target_var', y='NRR', data=df, joint_kws = dict(alpha=0.5))
    plot.ax_joint.plot([axis_min, axis_max], [axis_min, axis_max], 'black', linewidth=1)
    plot.ax_marg_x.set_xlim([axis_min, axis_max])
    plot.ax_marg_y.set_ylim([axis_min, axis_max])
    sns.regplot(x='target_var', y='NRR', data=df, ax=plot.ax_joint, scatter=False, color='red')
    plot.set_axis_labels('Simulated', 'Predicted')
    plt.savefig(save_path + 'densityplot_' + set_name + '_' + clim + '_' + get_datetime() + '.png')
    plt.close()
    
def plot_monthly_residuals(save_path, df, set_name, clim):
    
    palette = 'Set3'
    df['FertMonth'] = df['FertMonth'].apply(lambda x: {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}[x])
    
    order = ['Jan', 'Feb', 'Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    p=sns.boxplot(data=df, x='FertMonth', y="Residual", order=order, palette=palette, showfliers = False)
    
    for label in p.xaxis.get_ticklabels():
        p.set(ylim=(-1, 15))
        p.set(yticks=np.arange(0, 15, 2.5))
        p.set(xlabel='')
        p.set(ylabel='')
    
    plt.savefig(save_path + 'residuals_' + set_name + '_' + clim + '_' + get_datetime() + '.png')
    plt.close()    
    