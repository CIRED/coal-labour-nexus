# ===========================================================================================================================  
# Regional employment vulnerability to rapid coal transition in China and India, an integrated and downscaled assessment
# ===========================================================================================================================


#%%
# Importing libraries
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import date
from pyproj import CRS


# Plotting parameters
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Point
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import xycmap
from matplotlib.colors import PowerNorm
from matplotlib import ticker
import seaborn as sns
from IPython.display import Markdown as md

plt.rcParams['font.size'] = 9
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["legend.fancybox"] = False
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))


#%% Plotting functions
#    Stacked bars function
def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

#    Rectangle stripes function
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent],
                          width / self.num_stripes,
                          height,
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                          transform=trans)
            stripes.append(s)
        return stripes
    
# Functions used for bivariate plot 
def weighted_average(df, value, weight, country):
    return (df[df['Region'] == country][value] * df[df['Region'] == country][weight]).sum() / df[df['Region'] == country][weight].sum()

def interpol(x, xlim, ylim):
    y = ylim[0] + (ylim[1]-ylim[0])/(xlim[1]-xlim[0])*(x-xlim[0])
    return y


#%%
#Importing module results
T = range(2015, 2101)
T = np.array(T)

file_name = list(np.array([['output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NDC','NPI','NZ']] for y in ['','_PG0','_R55','_gem','_EW']]).flatten())

Result_data = []
for file in file_name:
    Result_data.append(pd.read_csv(file))#, dtype=str))

Result_data = pd.concat(Result_data, ignore_index=True)

Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].applymap(lambda x: str(x).replace('D', 'E'))
Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].apply(pd.to_numeric, errors='coerce')


col_data = pd.read_csv('plotting_data/Colors.csv', dtype=str)
z = [
    sns.color_palette(col_data['col'][k], n_colors=int(
        col_data['Num'][k]))[int(col_data['ind'][k])]
    for k in range(0, len(col_data['col'].values))
]
col_data['color'] = z


#%% #Importing Imaclim results

scenarios = list(np.array([[ x + y  for x in ['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']] for y in ['','_EW']]).flatten())

Imaclim_data = []
for scenario in scenarios:
    file_name ='input/IMACLIM_waysout_outputs_' + scenario +'.csv'
    Scenario_data= pd.read_csv(file_name)
    Scenario_data['Scenario'] = scenario
    Imaclim_data.append(Scenario_data)

Imaclim_data = pd.concat(Imaclim_data, ignore_index=True)
Imaclim_data.iloc[:, 5:] = Imaclim_data.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')


#%% # Importing additional data
# Map shapefile
Asia = gpd.read_file('plotting_data/Asia.shp')
Asia.loc[Asia.Region_Nam=="Orissa","Region_Nam"] = "Odisha"

# Historical employment data
Historical_data = pd.read_csv('data/Historical_labour.csv')

# Region positions for grid plots
provincesChina = {
    'Heilongjiang': (0, 5),
    'Xinjiang': (1, 0),
    'Qinghai': (1, 1),
    'Ningxia': (1, 2),
    'Inner Mongolia': (1, 3),
    'Liaoning': (1, 4),
    'Jilin': (1, 5),
    'Tibet': (2, 0),
    'Gansu': (2, 1),
    'Shaanxi': (2, 2),
    'Shanxi': (2, 3),
    'Hebei': (2, 4),
    'Sichuan': (3, 1),
    'Chongqing': (3, 2),
    'Hubei': (3, 3),
    'Henan': (3, 4),
    'Shandong': (3, 5),
    'Yunnan': (4, 1),
    'Guizhou': (4, 2),
    'Hunan': (4, 3),
    'Anhui': (4, 4),
    'Jiangsu': (4, 5),
    'Guangxi': (5, 2),
    'Jiangxi': (5, 3),
    'Zhejiang': (5, 4),
    'Shanghai': (5, 5),
    'Guangdong': (6, 3),
    'Fujian': (6, 4),
    'Hainan': (7, 4)
}

provincesIndia = {
    'Jammu & Kashmir': (0, 2),
    'Punjab': (1, 2),
    'Himachal Pradesh': (1, 3),
    'Uttarakhand': (1, 4),
    'Nagaland': (1, 7),
    'Rajasthan': (2, 1),
    'Haryana': (2, 2),
    'Uttar Pradesh': (2, 3),
    'Bihar': (2, 4),
    'Sikkim': (2, 5),
    'Assam': (2, 6),
    'Manipur': (2, 7),
    'Gujarat': (3, 0),
    'Madhya Pradesh': (3, 1),
    'Delhi': (3, 2),
    'Chhattisgarh': (3, 3),
    'Jharkhand': (3, 4),
    'Meghalaya': (3, 5),
    'Tripura': (3, 6),
    'Maharashtra': (4, 1),
    'Telangana': (4, 2),
    'Odisha': (4, 3),
    'West Bengal': (4, 4),
    'Goa': (5, 1),
    'Karnataka': (5, 2),
    'Andhra Pradesh': (5, 3),
    'Kerala': (6, 2),
    'Puducherry': (6, 3),
    'Tamil Nadu': (7, 3)
}

#Unit
exa2giga        =                 1e9 # G / E
tep2gj          =              41.855 # GJ/tep
mtoe2gj         =        1e6 * tep2gj # GJ/Mtep
mtoe2ej         =  mtoe2gj / exa2giga # EJ/Mtep


#%% 
Colors = {'NPI':sns.color_palette()[3],
          'NDC':sns.color_palette()[2],
          'NZ':sns.color_palette()[0],
          'NPI_gem':sns.color_palette('pastel')[3],
          'NDC_gem':sns.color_palette('pastel')[2],
          'NZ_gem':sns.color_palette('pastel')[0]}