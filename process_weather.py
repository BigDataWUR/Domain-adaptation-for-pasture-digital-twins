import pandas as pd
import os
import matplotlib 
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('...WeatherDataAll.csv')

#keep only relevant data
df['Climate'] = df['Weather'].apply(lambda x: x[:5])
df = df[(df.Climate=='Clim4') | (df.Climate=='Clim7') | (df.Climate=='Clim6')]
df['year'] = df['Date'].apply(lambda x: int(x[-4:]))
df = df[(df.year>=1982) & (df.year<=2003)] 
df = df.reset_index(drop=True)
#get months
df['month'] = df['Date'].apply(lambda x: {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}[int(x[-7:-5])])
#rename climates
df['Climate'] = df['Climate'].apply(lambda x: {'Clim4':'Location 1', 'Clim6':'Location 2', 'Clim7':'Location 3'}[x])

order = ['Jan', 'Feb', 'Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#create precipitation and temperature plots
for y in ['Rain', 'MaxT']:
    ax=sns.boxplot(data=df, x='month', y=y, order=order, hue='Climate', showfliers=False)
    if y == 'Rain':
        ax.set_ylabel('Precipitation (mm)')
        ax.set_xlabel('Month')
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels) 
    
    plt.savefig('.../' + y + '.png')
    plt.clf()