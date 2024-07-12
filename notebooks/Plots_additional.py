# ===========================================================================================================================
# ===========================================================================================================================  
# Regional employment vulnerability to rapid coal transition in China and India, an integrated and downscaled assessment
#                                                    Additional graphs
# ===========================================================================================================================
# ===========================================================================================================================
"""

This file is a companion to `Plots.ipynb` aimed at plotting additional graphs 
not necessarily meant to be published but still useful for scenario analysis.

"""

#%%
# Importing libraries
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import date
from pyproj import CRS
import os

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

plt.rcParams['font.size'] = 9
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["legend.fancybox"] = False
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

# ===========================================================================================================================
#%% Plotting functions
from plotting_functions import *



# ===========================================================================================================================
#%%
#Importing module results
T = range(2015, 2101)
T = np.array(T)

file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NDC','NPI','NZ']] for y in ['','_PG0','_R55','_gem','_EW']]).flatten())

Result_data = []
for file in file_name:
    Result_data.append(pd.read_csv(file))#, dtype=str))

Result_data = pd.concat(Result_data, ignore_index=True)

Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].applymap(lambda x: str(x).replace('D', 'E'))
Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].apply(pd.to_numeric, errors='coerce')



#%% #Importing Imaclim results

scenarios = list(np.array([[ x + y  for x in ['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']] for y in ['']]).flatten())

Imaclim_data = []
for scenario in scenarios:
    file_name ='../coal.labour.nexus/input/IMACLIM_waysout_outputs_' + scenario +'.csv'
    Scenario_data= pd.read_csv(file_name)
    Scenario_data['Scenario'] = scenario
    Imaclim_data.append(Scenario_data)

Imaclim_data = pd.concat(Imaclim_data, ignore_index=True)
Imaclim_data.iloc[:, 5:] = Imaclim_data.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')

# ===========================================================================================================================
#%% # Importing additional data
# Map shapefile from naturalearthdata.com
Asia = gpd.read_file('shapefiles/Asia.shp')
Asia.loc[Asia.Region_Nam=="Orissa","Region_Nam"] = "Odisha"

# Historical employment data
Historical_data = pd.read_csv('data/Historical_labour.csv')

# Region positions for grid plots
provincesChina, provincesIndia = defining_province_grid()

# ===========================================================================================================================
#Unit
exa2giga        =                 1e9 # G / E
tep2gj          =              41.855 # GJ/tep
mtoe2gj         =        1e6 * tep2gj # GJ/Mtep
mtoe2ej         =  mtoe2gj / exa2giga # EJ/Mtep

Colors = defining_waysout_colour_scheme()
# ===========================================================================================================================
#%% 1) Different ways to look at unemployment

Sectors = ['Coal','Oil','Gas','ET','Elec','BTP','Services','Air','Mer','Ot','Agri','Indus']

cols = ['k','darkgrey','grey']+[sns.color_palette('pastel')[x] for x in [8,9,7,3,4,5,6,2,0] ]+[sns.color_palette()[1]]

Scenarios = ['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']
Scenarios_name = ['NPI','WO-NDCLTT','WO-15C']

Regions = ['China','India']

fig, axss = plt.subplots(6,3, figsize=(10, 15))

for ind_scenario, scenario in enumerate(Scenarios):
    for ind_region, region in enumerate(Regions):
        axs = axss[len(Scenarios)*ind_region+ind_scenario]

        # Employment from Imaclim
        
        for ind_ILO in [0,1]:
            ax = axs[ind_ILO]

            data = []
            for sector in Sectors:
                data.append(Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==['CHN','IND'][ind_region]) & (Imaclim_data['Variables']=='Employment|'+['','ILO|'][ind_ILO]+sector)].values[0][5:])

            data.append(
                Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==['CHN','IND'][ind_region]) & (Imaclim_data['Variables']=='Z')].values[0][5:]*
                Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==['CHN','IND'][ind_region]) & (Imaclim_data['Variables']=='Labour|Force')].values[0][5:]*1e-6
                )
            
            
            data = np.array(data)
            data_shape = np.shape(data)
            cumulated_data = get_cumulated_array(data, min=0)
            cumulated_data_neg = get_cumulated_array(data, max=0)

            # Re-merge negative and positive data.
            row_mask = (data < 0)
            cumulated_data[row_mask] = cumulated_data_neg[row_mask]
            data_stack = cumulated_data

            alines = []
            for i in np.arange(0, data_shape[0]):
                alines.append([
                    ax.bar(T,
                            data[i],
                            bottom=data_stack[i],
                            width=1,
                            alpha=0.6,
                            color=cols[i])
                ])

            ax.set_ylim([0,1000])


        # Employment from module
        ax = axs[2]
        data = []
        for variable in ['Employment|Coal|Downscaled','Employment|Non Coal|Downscaled','Unemployment|Downscaled']:
            data.append(Result_data[(Result_data['Scenario']==['NPI','NDC','NZ'][ind_scenario]) & (Result_data['Downscaled Region']==region) & (Result_data['Variable']==variable)].values[0][6:])
        data = np.array(data)*1e-6
        data_shape = np.shape(data)
        cumulated_data = get_cumulated_array(data, min=0)
        cumulated_data_neg = get_cumulated_array(data, max=0)

        # Re-merge negative and positive data.
        row_mask = (data < 0)
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data

        alines = []
        for i in np.arange(0, data_shape[0]):
            alines.append([
                ax.bar(T,
                        data[i],
                        bottom=data_stack[i],
                        width=1,
                        alpha=0.6,
                        color = ['k','grey',sns.color_palette()[1]][i])
            ])

        
    
        if (ind_scenario == 0) & (ind_region == 0):
            [axs[x].set_title(['Imaclim','ILO','Module'][x]) for x, ax in enumerate(axs)]
        
        axs[0].set_ylabel(['NPI','NDC','NZ'][ind_scenario]+' '+['China','India'][ind_region])
        [ax.set_ylim([0,[1100,1200][ind_region]]) for ax in axs]


#%% 2) Carbon price
Regions = ['World','CHN','IND']
variable = 'Price|Carbon'
Scenarios = ['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']

fig, axs = plt.subplots(1,3, figsize=(10, 10))

for ind_region, region in enumerate(Regions):
    ax = axs[ind_region]
    for ind_scenario, scenario in enumerate(Scenarios):
        y = Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==variable)].values[0][5:]
        ax.plot(T, y, label=Scenarios[ind_scenario], color = Colors[scenario])

#%% 3) Mapping productivity
Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)

zlim = [0, 150]
tickz = np.array([5]+list(np.arange(25,176,50)))

Ts = [2020, 2035, 2050]
Scenarios = ['NPI','NPI_gem']
Scenarios_names = ['NPI','NPI GEM']
fig, axs = plt.subplots(1,len(Scenarios),
                        figsize=(6.7, 5))


t = 2015

Data_Chn = []
Data_Ind = []
for s_index, scenario in enumerate(Scenarios):
    ax = axs[s_index]
    Unemployment = []
    
    if s_index ==0:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cax.set_ylabel('TJ/person')

    for region in Regions:

        Q = Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Resource|Extraction|Coal|Downscaled') &
                        (Result_data['Scenario'] == scenario)][str(t)].values[0]
        E = Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Employment|Coal|Downscaled') &
                        (Result_data['Scenario'] == scenario)][str(t)].values[0]
        

        if (region in ['Jharkhand','Shanxi'])&(scenario == 'NPI'):
            Qt = round(Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Employment|Coal|Downscaled') &
                        (Result_data['Scenario'] == scenario)][str(2021)].values[0]/1000,1)
            print(f'There are {Qt} thousands coal workers in {region} in {scenario} in 2021')

        if E != 0:
            Unemployment.append(float(Q / E*1e6))
        else:
            Unemployment.append(np.nan)

        if Result_data[(Result_data['Downscaled Region'] == region
                        )]['Region'].values[0] == 'CHN':
            Data_Chn.append(Unemployment[-1])
        else:
            Data_Ind.append(Unemployment[-1])
    
    Unemployment = pd.DataFrame(data=np.array([list(Regions), Unemployment]).transpose(),
                        columns=['Region_Nam', 'Unemployment'])

    Asia_Data = Asia.merge(Unemployment, on='Region_Nam')

    Asia_Data['Unemployment'] = pd.to_numeric(Asia_Data['Unemployment'], errors='coerce')



    cbar = Asia_Data.plot(column='Unemployment',
                        cmap='OrRd',
                        legend=True,
                        ax=ax,
                        edgecolor='black',
                        missing_kwds={
                            "color": "lightgrey",
                            "label": "Missing values",
                        },
                        vmin=zlim[0],
                        vmax=zlim[1],
                        linewidth=0.75,
                        cax=cax,
                        rasterized=True)

    Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color='whitesmoke', edgecolor='black', linestyle='--',linewidth=0.5)
    

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([65,140])
    ax.set_ylim([7,55])

    ax.set_title(Scenarios_names[s_index])
    
# %% 4) Macroeconomic cost

Scenarios = ['WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']
var = 'GDP|PPP'
Regions = ['CHN','IND']

fig, axs = plt.subplots(1,2, figsize=(10, 10))
for ind_reg, region in enumerate(Regions):
    ax = axs[ind_reg]
    base = Imaclim_data[(Imaclim_data['Scenario']=='WO-NPi-ElecIndus') & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]
    for ind_scenario, scenario in enumerate(Scenarios):
        y = (Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]-base)/base*100

        y2 = [np.mean(y[i-4:i+4]) for i in range(4,len(y)-4)]
        ax.plot(T, y, label=Scenarios[ind_scenario], color = Colors[scenario], linewidth= 0.5)
        ax.plot(T[4:-4], y2, label=Scenarios[ind_scenario], color = Colors[scenario])



        print(f'In 2050 the GDP loss in {region} is {y2[2050-2015-4]:.0f}% for {Scenarios[ind_scenario]}')
    ax.set_title(region)
    ax.set_ylabel('GDP loss [%]')
    ax.set_ylim([-30,15])


# %%
