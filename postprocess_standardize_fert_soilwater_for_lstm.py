import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, glob, pickle, math, sys
import warnings

#path_to_folder is the path of the folder containing the CSVs. There are supposed to be multiple CSVs as output of Spark
def combine_dataset(path_to_folder):
    if os.path.isdir(path_to_folder):
        df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(path_to_folder, "*.csv")))
    else:
        df = pd.read_csv(path_to_folder)
        
    return df

def standardize_values(train_df, validation_df, test_df, scaler):
    
    #to seperate the case where we use observations with an already made scaler
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(train_df)
    else:
        scaler = pickle.load(open(scaler, 'rb'))
        
    train_df[train_df.columns] = scaler.transform(train_df)
    validation_df[validation_df.columns] = scaler.transform(validation_df)
    test_df[test_df.columns] = scaler.transform(test_df)    
        
    return [train_df, validation_df, test_df, scaler]

def add_month_to_x_y_columns(df):
    
    #month to coordinates
    df['month_x'] = df['FertMonth'].apply(lambda month: math.cos(math.radians(month * 360/12)))
    df['month_y'] = df['FertMonth'].apply(lambda month: math.sin(math.radians(month * 360/12)))
    
    #round to zero
    df['month_x'] = df['month_x'].apply(lambda x: 0 if abs(x) < 0.000001 else x)
    df['month_y'] = df['month_y'].apply(lambda y: 0 if abs(y) < 0.000001 else y)
    
    return df

#shouldFilter is a flag that allows filtering based on a condition to happen. It is used to filter conditions for pretraining dataset of the transfer learning experiments.
#when postprocess is called from the sample_seperator_and_mixer.py it should be False because the incoming data are already filtered
def postprocess(clim, base_path, provided_scaler, train_csv, validation_csv, test_csv, pretraining_or_tuning, variability_setup):
    
        #to supress warning about highly fragmented dataframe
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        print('Postprocessing', clim)

        train_path =      base_path + '/' + train_csv
        validation_path = base_path + '/' + validation_csv
        test_path =       base_path + '/' + test_csv

        train_df = combine_dataset(train_path)
        train_df['FertRate_orig'] = train_df['FertRate']
        train_df['SoilWater_orig'] = train_df['SoilWater']
        validation_df = combine_dataset(validation_path)
        validation_df['FertRate_orig'] = validation_df['FertRate']
        validation_df['SoilWater_orig'] = validation_df['SoilWater']
        test_df = combine_dataset(test_path)
        test_df['FertRate_orig'] = test_df['FertRate']
        test_df['SoilWater_orig'] = test_df['SoilWater']

        initial_column_order = train_df.columns #we could take it also from validation or test set

        print('combined shapes')
        print(train_df.shape)
        print(validation_df.shape)
        print(test_df.shape)
        
        if pretraining_or_tuning=='tuning':
            train_df = train_df[(train_df['SoilFertility']==1) & (train_df['SoilWater']==67) & (train_df['Irrigation']==0) & (train_df['FertRate'].isin([20,40])) & (train_df['FertMonth'].isin([3,4,5,9,10,11]))]. \
            reset_index(drop=True)
            
            validation_df = validation_df[(validation_df['SoilFertility']==1) & (validation_df['SoilWater']==67) & (validation_df['Irrigation']==0) & (validation_df['FertRate'].isin([20,40])) & (validation_df['FertMonth'].isin([3,4,5,9,10,11]))]. \
            reset_index(drop=True)
            
            test_df = test_df[(test_df['SoilFertility']==1) & (test_df['SoilWater']==67) & (test_df['Irrigation']==0) & (test_df['FertRate'].isin([20,40])) & (test_df['FertMonth'].isin([3,4,5,9,10,11]))]. \
            reset_index(drop=True)
        else:
            #in this case we do the pretraining climate
            if variability_setup == 'little_weather_little_factorials' or variability_setup == 'more_weather_little_factorials':
                train_df = train_df[(train_df['SoilFertility']==1) & (train_df['SoilWater']==67) & (train_df['Irrigation']==0) & (train_df['FertRate'].isin([20,40]))]. \
                reset_index(drop=True)
                
                validation_df = validation_df[(validation_df['SoilFertility']==1) & (validation_df['SoilWater']==67) & (validation_df['Irrigation']==0) & (validation_df['FertRate'].isin([20,40]))]. \
                reset_index(drop=True)
                
                test_df = test_df[(test_df['SoilFertility']==1) & (test_df['SoilWater']==67) & (test_df['Irrigation']==0) & (test_df['FertRate'].isin([20,40]))]. \
                reset_index(drop=True)
            elif variability_setup == 'little_weather_more_factorials' or variability_setup == 'more_weather_more_factorials':
                train_df = train_df[train_df['Irrigation']==0]. \
                reset_index(drop=True)
                
                validation_df = validation_df[validation_df['Irrigation']==0]. \
                reset_index(drop=True)
                
                test_df = test_df[test_df['Irrigation']==0]. \
                reset_index(drop=True)
            else:
                raise Exception('Bad variability_setup in postprocessing of pretraining climate') 
                
            print('after filter shapes')
            print(train_df.shape)
            print(validation_df.shape)
            print(test_df.shape)
        
        
        #columns which will not be standardized
        cols_no_standardize = ['File', 'Weather', 'target_var',  
                               'SoilFertility',  'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year']

        train_df_file_target_var = train_df[cols_no_standardize]
        validation_df_file_target_var = validation_df[cols_no_standardize]
        test_df_file_target_var = test_df[cols_no_standardize]             
                     
        standardized_train_df, standardized_validation_df, standardized_test_df, scaler = standardize_values(train_df.drop(columns=cols_no_standardize), validation_df.drop(columns=cols_no_standardize), test_df.drop(columns=cols_no_standardize), provided_scaler)

        #save scaler
        #we only want to save it if it's created this time. If it's already provided then there is no point in saving the same thing again
        if provided_scaler is None:
            scaler_save_path = base_path + '/scaler'
            pickle.dump(scaler, open(scaler_save_path, 'wb'))

        print('standardized shapes')
        print(standardized_train_df.shape)
        print(standardized_validation_df.shape)
        print(standardized_test_df.shape) 

        # put non-standardized columns back to the df that has the standardized ones
        # this one hits numpy.core._exceptions.MemoryError: Unable to allocate 8.34 GiB for an array with shape (423, 2646130) and data type float64
        # standardized_train_df = standardized_train_df.join(train_df_file_target_var)
        # standardized_validation_df = standardized_validation_df.join(validation_df_file_target_var)
        # standardized_test_df = standardized_test_df.join(test_df_file_target_var)

        # put non-standardized columns back to the df that has the standardized ones
        for col in cols_no_standardize:
            standardized_train_df[col] = train_df_file_target_var[col]
            standardized_validation_df[col] = validation_df_file_target_var[col]
            standardized_test_df[col] = test_df_file_target_var[col]

        #rearrange columns
        standardized_train_df =      standardized_train_df[      initial_column_order]
        standardized_validation_df = standardized_validation_df[ initial_column_order]
        standardized_test_df =       standardized_test_df[       initial_column_order]

        print('standardized shapes with File and target_var')
        print(standardized_train_df.shape)
        print(standardized_validation_df.shape)
        print(standardized_test_df.shape)

        #add x y representation for month
        standardized_train_df = add_month_to_x_y_columns(standardized_train_df)
        standardized_validation_df = add_month_to_x_y_columns(standardized_validation_df)
        standardized_test_df = add_month_to_x_y_columns(standardized_test_df)
        
        #rearrange columns again
        columns = list(standardized_train_df.columns)
        columns = columns[:-11] + columns[-2:] + [columns[-11]] + [columns[-7]] + columns[-10:-7] + columns[-6:-2]
        standardized_train_df = standardized_train_df[columns]
        standardized_validation_df = standardized_validation_df[columns]
        standardized_test_df = standardized_test_df[columns]
        
        print('--------------------------------------------')
        print(clim, pretraining_or_tuning, variability_setup)
        print(standardized_train_df.shape[0], standardized_validation_df.shape[0], standardized_test_df.shape[0])
        for c in ['Year', 'SoilFertility', 'Irrigation', 'FertMonth', 'FertRate_orig', 'SoilWater_orig']:
            print(c + ':', sorted(standardized_train_df[c].unique()))
        print('--------------------------------------------')
        
        #save dfs
        standardized_train_df.to_csv(      base_path + '/' + train_csv[:-4] + '_standardized_fert_soilwater' + '.csv', index=False)
        standardized_validation_df.to_csv( base_path + '/' + validation_csv[:-4] + '_standardized_fert_soilwater' + '.csv', index=False)
        standardized_test_df.to_csv(       base_path + '/' + test_csv[:-4] + '_standardized_fert_soilwater' + '.csv', index=False)
              
        if provided_scaler is None:
            return scaler_save_path
        else:
            return None
