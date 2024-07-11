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


#%% Plotting functions
from plotting_functions import *


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


#%% # Importing additional data
# Map shapefile
Asia = gpd.read_file('shapefiles/Asia.shp')
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


#%%
# ===========================================================================================================================
# ===========================================================================================================================
#                                              Plots for the core of the article
# ===========================================================================================================================
# ===========================================================================================================================


#%% 1) Employment trajectories
# ===========================================================================================================================

Countries = ['India', 'China']
Regions = [
    Result_data[(Result_data['Region'] == ['IND', 'CHN'][x])&(~Result_data['Downscaled Region'].isin(['China', 'India']))]['Downscaled Region'].unique()
    for x in [0, 1]
]
Scen_type = ['NPI','NDC','NZ']
Alt_type = [
    '','_PG0']
Scenarios = [x + y for x in Scen_type for y in Alt_type]

Ralpha = [1, 0.75] * 3
Rlinestyle = ['-','--']*3
Rlinewidth = [1, 0.75]* 3
Rmarker = ['', '', '', '', ''] * 3

fig, axs = plt.subplots(1, 2, figsize=(27 / 3, 13 / 3))
for c_index in [0, 1]:
    country = Countries[c_index]

    Variable = "Employment|Coal|Downscaled"

    ax = axs[c_index]

    ax.axhline(y=0, color='k', linewidth=0.8)

    ax.plot(Historical_data['t'],Historical_data[country]/1e6,color='k')


    Scen_y = []
    for j,scenario in enumerate(Scenarios):
        COAL_emp = Result_data[(Result_data['Downscaled Region'] == country)
                               & (Result_data['Variable'] == Variable) &
                               (Result_data['Scenario'] == Scenarios[j])]

        y = np.zeros(86)
        for region in Regions[c_index]:
            y = y + np.array(Result_data[
                (Result_data['Downscaled Region'] == region)
                & (Result_data['Variable'] == Variable)
                &
                (Result_data['Scenario'] == Scenarios[j])].values[0][6:]) / 1e6

        Scen_y.append(y)

        if Scenarios[j] in  ['NPI','NDC','NZ']:
            if COAL_emp.values[0][6:][
                    T < 2070][-1] < COAL_emp.values[0][6:][5] / 2:
                
                ax.scatter(T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] /2][0],
                        -0.125,
                        color=Colors[scenario.split('_')[0]],
                        marker='o',
                        s=20)
                
                if COAL_emp.values[0][6:][
                    T < 2070][-1] < COAL_emp.values[0][6:][5] *0.05:
                    ax.scatter(T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] *0.05][0],
                            -0.125,
                            color=Colors[scenario.split('_')[0]],
                            marker='^',
                            s=20)
                    
                if scenario == 'NZ':
                    print(f'In {country}, coal employment must be halved by {T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] /2][0]} to be 1.5°C-aligned')

    Scen_y = pd.DataFrame(Scen_y, index=Scenarios)

    for Scen_type_ind,scenario in enumerate(Scenarios):
        ax.plot(T[T < 2070],
                Scen_y.loc[scenario].values[T < 2070],
                color=Colors[scenario.split('_')[0]],
                linestyle=Rlinestyle[Scen_type_ind],
                linewidth=Rlinewidth[Scen_type_ind],
                alpha=Ralpha[Scen_type_ind],)

    # Formatting axes
    ax.set_title(country)
    ax.set_ylabel('Million workers')
    ax.set_ylim([-0.25, 5])
    ax.axvline(x=2020, color='k', linestyle='--', linewidth=0.8)

alines = []

for Scen_type_ind in [0,2,4,1,3,5]:
    scenario = Scenarios[Scen_type_ind]
    alines.append(axs[0].plot([], [],
                              color=Colors[scenario.split('_')[0]],
                              label=['NPi','NPi no growth','NDC','NDC no growth','1.5°C no growth','1.5°C'][Scen_type_ind],
                              linestyle=Rlinestyle[Scen_type_ind],
                linewidth=Rlinewidth[Scen_type_ind],
                alpha=Ralpha[Scen_type_ind] )[0])

alines.append(axs[0].plot([], [],
                        color='k',
                        label='Historical data')[0])
alines.append(axs[0].scatter([], [],
                        color='k',
                        marker='o',
                        s=20,
                        alpha = 0.5,
                        label='-50% from 2020'))
alines.append(axs[0].scatter([], [],
                        color='k',
                        marker='^',
                        s=20,
                        alpha = 0.5,
                        label='-95% from 2020'))


labels = [l.get_label() for l in alines]
handles = [l for l in alines]

fig.legend(handles=alines,
           labels=labels,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.1),
           frameon=False)
# %% 2) Exposure and vulnerability of regions to coal transition betweeen 2020 and 2050
# ===========================================================================================================================



# Defining the colormap

b_xlim = [0.3,0.85]
b_ylim = [np.log(6e-4),np.log(0.06)]


n = (10, 10)  # x, y

corner_colors = ("#e8e8e8",   "#C85a5a", "#64acbe", "#574249")
cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)

Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China','India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)


# Defining the scenarios
Scenarios =  ['NPI','NDC','NZ']
Scenarios_names = ['NPI','NDC','1.5°C']

t0 = 2019
t1 = 2050

# Creating the figure
fig1, axs1 = plt.subplots(2,2, figsize=(18/1.6, 15.3/1.6))
axs = [axs1]
axs = np.array(axs).flatten()


# Initiating the data lists
Data_Chn=[]
Data_Ind=[]
key_data = {}

# Iterating over scenarios
for s_index in [0,2,1]:
    ax = axs[s_index]
    scenario = Scenarios[s_index]

    
    Asia_Data, key_data = plot_bivariate(scenario, ax, Regions, Result_data, cmap, b_xlim, b_ylim, T, t0, t1, Asia, s_index, key_data, Scenarios_names)
    """
    # Iterating over regions
    for region in list(Regions):

        L = Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Employment|Coal|Downscaled') &
                        (Result_data['Scenario'] == scenario)].values[0][6:]
        LF0 = float(
            Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Labour Force|Downscaled')
                        & (Result_data['Scenario'] == scenario)].values[0][6])
        Ls.append(L[0])
        y = float((L[T == 2020] - L[T == 2035]) / LF0)
        if y == 0:
            y = np.nan
        else:
            y = np.log(y)
        share_destruction.append(y)
        destruction.append(float((L[T == 2020] - L[T == 2035]) ))


        
        D = np.array(
            Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
                        & (Result_data['Scenario'] == scenario) &
                        (Result_data['Downscaled Region']
                         == region)].values.flat[6:][(T < t1) & (T > t0)])
        U = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])
        I = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])

        if sum((U + D + I)) != 0:
            share_n_finding.append(1-sum(D+I) / sum((U + D + I)))
        else:
            share_n_finding.append(np.nan)

        cntry.append(Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0])
        if Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0] == 'CHN':	
            Dc.append(share_n_finding[-1])
        else:
            DI.append(share_n_finding[-1])
    
       

    Regions = list(Regions)
    share_n_finding = pd.DataFrame(data=np.array([list(Regions), share_n_finding,cntry,share_destruction,destruction,Ls]).transpose(),
                           columns=['Region_Nam', 'share_n_finding','Region','share_destruction','destruction','Workforce'])

    share_n_finding['share_n_finding'] = pd.to_numeric(share_n_finding['share_n_finding'], errors='coerce')
    share_n_finding['share_destruction'] = pd.to_numeric(share_n_finding['share_destruction'], errors='coerce')
    share_n_finding['destruction'] = pd.to_numeric(share_n_finding['destruction'], errors='coerce')
    share_n_finding['Workforce'] = pd.to_numeric(share_n_finding['Workforce'], errors='coerce')
    
    Data_Chn.append(share_n_finding)
    Asia_Data = Asia.merge(share_n_finding, on='Region_Nam')

    Asia_Data['share_n_finding'] = pd.to_numeric(Asia_Data['share_n_finding'], errors='coerce')
    Asia_Data['share_destruction'] = pd.to_numeric(Asia_Data['share_destruction'], errors='coerce')
    Asia_Data['destruction'] = pd.to_numeric(Asia_Data['destruction'], errors='coerce')


    cmapi = xycmap.bivariate_color(sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim)
    cmapi = pd.DataFrame(data=np.array([cmapi,Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['Region_Nam'].values]).transpose(),
                           columns=['colors', 'Region_Nam'])
    
    Asia_Data_with_colors = Asia_Data.merge(cmapi, on='Region_Nam', how='left').fillna('lightgrey')
    Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
        edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)

    Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color="whitesmoke", edgecolor='black', linestyle='--',linewidth=0.5)
    
    for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
        key_data[region+str(s_index)] = {'Downscaled Region':region,
                                         'Scenario':scenario,
                                        'coordinates':[float(share_n_finding[share_n_finding['Region_Nam']==region]['share_n_finding'].values[0]),
                                    float(share_n_finding[share_n_finding['Region_Nam']==region]['share_destruction'].values[0])],
                                    'destruction':float(share_n_finding[share_n_finding['Region_Nam']==region]['destruction'].values[0])}

    for region in ['China','India']:  
        key_data[region+str(s_index)] = {'Downscaled Region':'Average \n'+region,
                                         'Scenario':scenario,
                                        'coordinates':[weighted_average(share_n_finding, 'share_n_finding','Workforce', ['IND' if region =='India' else 'CHN'][0]),
                                                       weighted_average(share_n_finding, 'share_destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])],
                                    'destruction':weighted_average(share_n_finding, 'destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])}
        

    
    ax.set_title(Scenarios_names[s_index])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([65,140])
    ax.set_ylim([7,55])
    """

axs[-1].set_axis_off()



# ====================================
# Legend

# Creating legend axis
cax = fig1.add_axes([0.55, 0.18, 0.29, 0.29])
cax = xycmap.bivariate_legend(ax=cax, sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim) #xlims=(0,0.5),ylims=(0,0.06)
cax.set_xlabel('Share of laid-off workers \n going into unemployment',fontsize=11)
cax.set_ylabel('Decrease in relative coal jobs',fontsize=11)

# If the number of color box is more than 5, not all ticks are shown
if min(n)>5:    
    cax.set_xticks([interpol(x,b_xlim,cax.get_xlim()) for x in [0.5,0.7,0.9]])
    cax.set_yticks([interpol(np.log(x),b_ylim,cax.get_ylim()) for x in [1e-3,5e-3,0.01,0.05]])
    cax.set_xticklabels(["50%","70%","90%"],fontsize=11)
    cax.set_yticklabels(["0.1%","0.5%","1%","5%"],fontsize=11)


# Plotting evolution of results across main scenarios for the main regions
for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [0,1,2]:
        xs.append(interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)



# Scattering the results for the main regions
for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [0,1,2]:
        xs.append(interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
        cax.scatter(xs[-1],ys[-1],color=Colors[key_data[region+str(s_index)]['Scenario']],marker='o',s=key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)
    cax.annotate(region,(xs[0],ys[0]-0.1),ha='center',va='top',fontsize=8)


# Legend
alines = []
alines.append(cax.scatter([], [], color=Colors['NPI'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NDC'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NZ'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color='white', marker='o',s=0))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=2e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=4e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=6e5/1200))
labels = ['NPi','NDC LTS', '1.5°C','Job losses \n [people]','200k','400k','600k']
cax.legend(handles=alines,
           labels=labels,
           loc='center right',
           bbox_to_anchor=(1.4, 0.5),
           frameon=False,
           fontsize=8)

fig1.subplots_adjust(wspace=0.07,hspace=-0.04)

Regions = ['Shanxi','Jharkhand']
for region in Regions:
    prod = Result_data[(Result_data["Downscaled Region"] == region) & (Result_data["Variable"] == "Resource|Extraction|Coal|Downscaled") & (Result_data["Scenario"] == "NPI_gem")]['2021'].values[0]
    emp = Result_data[(Result_data["Downscaled Region"] == region) & (Result_data["Variable"] == "Employment|Coal|Downscaled") & (Result_data["Scenario"] == "NPI_gem")]['2021'].values[0]
 
    print(f'Production in {region} in 2021 is {prod} EJ')     
    print(f'Employment in {region} in 2021 is {round(emp)} people')



# %% 3) Mobility of laid-off coal workers between 2020-2030 and 2020-2050.
# ===========================================================================================================================
# BARCHART DESTINATION 1 Period
Provinces = [provincesChina, provincesIndia]
Countries = ['China', 'India']
nls = [6, 8]

Scenarioss = [ ['NPI','NDC','NZ'],['NPI','NDC','NZ']]

Scenarioss_name = [[ 'NPI', 'NDC\nLTT', '1.5°C'],[ 'NPI', 'NDC\nLTT', '1.5°C']]

t0 = 2020
t1 = 2050

Xs = [
    list(range(len(Scenarioss[0]))),list(range(len(Scenarioss[0])))
]
data_save = []
fig, axs = plt.subplots(2,
                        2,
                        figsize=(17.21 / 2.54, 13.09 / 2.54),
                        )
for c_index in [0, 1]:
    x = 0
    provinces = Provinces[c_index]
    region = ['China', 'India'][c_index]
    
    for stype_index in [0,1]:
        ax = axs[c_index][stype_index]
        t0 = 2020
        t1 = [2030,2050][stype_index]
        if stype_index == 0:
            ax.set_ylabel(region+'\nMillion workers')
        Scenarios = Scenarioss[stype_index]
        Sc = Scenarios
        X = Xs[stype_index]
        for s_index in range(len(Scenarios)):
            Scenario = Scenarios[s_index]
            Retirement = 0
            Direct = 0
            Indirect = 0
            Unemployed = 0
            Hire = 0
            for province, pos in provinces.items():

                Retirement += sum(Result_data[
                    (Result_data['Variable'] == 'Coal Worker Destination|Retire')
                    & (Result_data['Scenario'] == Scenario) &
                    (Result_data['Downscaled Region']
                     == province)].values.flat[6:][(T < t1) & (T > t0)])
                Direct += sum(Result_data[
                    (Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
                    & (Result_data['Scenario'] == Scenario) &
                    (Result_data['Downscaled Region']
                     == province)].values.flat[6:][(T < t1) & (T > t0)])
                Indirect += sum(
                    Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                & (Result_data['Scenario'] == Scenario) &
                                (Result_data['Downscaled Region']
                                 == province)].values.flat[6:][(T < t1)
                                                               & (T > t0)])
                Unemployed += sum(
                    Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                                & (Result_data['Scenario'] == Scenario) &
                                (Result_data['Downscaled Region']
                                 == province)].values.flat[6:][(T < t1)
                                                               & (T > t0)])
                Hire += sum(-Result_data[
                    (Result_data['Variable'] == 'Coal Worker Destination|Hire')
                    & (Result_data['Scenario'] == Scenario)
                    & (Result_data['Downscaled Region'] == province)].values.flat[6:][
                        (T < t1) & (T > t0)])

            data = np.array([Retirement, Direct, Indirect, Unemployed, Hire
                             ]) / 1e6
            data_save.append(data)
            data_shape = np.shape(data)
            cumulated_data = get_cumulated_array(data, min=0)
            cumulated_data_neg = get_cumulated_array(data, max=0)

            # Re-merge negative and positive data.
            row_mask = (data < 0)
            cumulated_data[row_mask] = cumulated_data_neg[row_mask]
            data_stack = cumulated_data

            alines = []
            clrs = [sns.color_palette()[x] for x in [1, 2, 4, 3, 6]]
            alines = []
            labelz = [
                'Retirement', "Instantaneous \nmatches", "Delayed \nmatches",
                'Unemployed', 'Hire'
            ]
            for i in np.arange(0, data_shape[0]):
                alines.append([
                    ax.bar(X[s_index],
                           data[i],
                           bottom=data_stack[i],
                           width=0.8,
                           alpha=0.6,
                           color=clrs[i],
                           label=labelz[i])
                ])

            x += 1


            if (Scenario == 'NPI')&(stype_index == 1):
                print(f'In {region}, {round(data[3]*1000)} thousand workers will not find new employment by 2050')

        alines = [x[0] for x in alines]
        labs = [l.get_label() for l in alines]
        ax.axhline(y=0, color='k', linewidth=0.9)

        
        if c_index == 0:
            ax.set_title(str(t0) + '-' + str(t1))
            ax.set_xticks([])
        else:
            ax.set_xticks(X)
            ax.set_xticklabels(
            Scenarioss_name[stype_index],
            rotation=90,
            )

        ax.set_ylim([-0.5, 3.8])

fig.legend(handles=alines,
           labels=labs,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.1),
           frameon=False)

fig.subplots_adjust(hspace=0.02, wspace=0.1)


#%%
# ===========================================================================================================================
# ===========================================================================================================================
#                                              Plots for the core of the annex
# ===========================================================================================================================
# ===========================================================================================================================

#%% 1) Sensitivity analyses
# ===========================================================================================================================
# ===========================================================================================================================
#%% 1.1) Calibration source
# ===========================================================================================================================


b_xlim = [0.3,0.85]
b_ylim = [np.log(11e-5),np.log(0.06)]


n = (10, 10)  # x, y
corner_colors = ("#e8e8e8",   "#C85a5a", "#64acbe", "#574249")
cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)

Result_data[Result_data['Downscaled Region']=='Odisha']=='Odisha'

Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China','India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)



Scenarios =  ['NPI','NDC','NZ','NPI_gem','NDC_gem','NZ_gem']

Scenarios_names = ['NPI','NDC','1.5°C','NPI GEM','NDC GEM','1.5 GEM']

fig1, axs1 = plt.subplots(3,2, figsize=(18/1.6, 15.3/1.6))

axs = [axs1]

axs = np.array(axs).flatten()
t0 = 2019
t1 = 2050
Data_Chn=[]
Data_Ind=[]


key_data = {}
s0 = 0
for s_index in [0,3,1,4,2,5]:
    ax = axs[s0]
    s0+=1
    scenario = Scenarios[s_index]
    share_n_finding = []
    cntry = []
    Dc=[]
    DI=[]
    share_destruction =[]
    destruction = []
    Ls = []
    
    for region in list(Regions):

        L = Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Employment|Coal|Downscaled') &
                        (Result_data['Scenario'] == scenario)].values[0][6:]
        LF0 = float(
            Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Labour Force|Downscaled')
                        & (Result_data['Scenario'] == scenario)].values[0][6])
        Ls.append(L[0])
        y = float((L[T == 2020] - L[T == 2035]) / LF0)
        if y == 0:
            y = np.nan
        else:
            y = np.log(y)
        share_destruction.append(y)
        destruction.append(float((L[T == 2020] - L[T == 2035]) ))


        
        D = np.array(
            Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
                        & (Result_data['Scenario'] == scenario) &
                        (Result_data['Downscaled Region']
                         == region)].values.flat[6:][(T < t1) & (T > t0)])
        U = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])
        I = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])

        if sum((U + D + I)) != 0:
            share_n_finding.append(1-sum(D+I) / sum((U + D + I)))
        else:
            share_n_finding.append(np.nan)

        cntry.append(Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0])
        if Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0] == 'CHN':	
            Dc.append(share_n_finding[-1])
        else:
            DI.append(share_n_finding[-1])
    
       

    
    Regions = list(Regions)
    share_n_finding = pd.DataFrame(data=np.array([list(Regions), share_n_finding,cntry,share_destruction,destruction,Ls]).transpose(),
                           columns=['Region_Nam', 'share_n_finding','Region','share_destruction','destruction','Workforce'])

    share_n_finding['share_n_finding'] = pd.to_numeric(share_n_finding['share_n_finding'], errors='coerce')
    share_n_finding['share_destruction'] = pd.to_numeric(share_n_finding['share_destruction'], errors='coerce')
    share_n_finding['destruction'] = pd.to_numeric(share_n_finding['destruction'], errors='coerce')
    share_n_finding['Workforce'] = pd.to_numeric(share_n_finding['Workforce'], errors='coerce')
    
    Data_Chn.append(share_n_finding)
    Asia_Data = Asia.merge(share_n_finding, on='Region_Nam')

    Asia_Data['share_n_finding'] = pd.to_numeric(Asia_Data['share_n_finding'], errors='coerce')
    Asia_Data['share_destruction'] = pd.to_numeric(Asia_Data['share_destruction'], errors='coerce')
    Asia_Data['destruction'] = pd.to_numeric(Asia_Data['destruction'], errors='coerce')

    norm = ([PowerNorm(gamma=1, vmin=0, vmax=1)] * 3 +
            [PowerNorm(gamma=1, vmin=0, vmax=1)] * 3)[s_index]


    cmapi = xycmap.bivariate_color(sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim)
    cmapi = pd.DataFrame(data=np.array([cmapi,Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['Region_Nam'].values]).transpose(),
                           columns=['colors', 'Region_Nam'])
    
    Asia_Data_with_colors = Asia_Data.merge(cmapi, on='Region_Nam', how='left').fillna('lightgrey')
    Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
    edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)

    Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color='whitesmoke', edgecolor='black', linestyle='--',linewidth=0.5)

    
    for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
        key_data[region+str(s_index)] = {'Downscaled Region':region,
                                         'Scenario':scenario,
                                        'coordinates':[float(share_n_finding[share_n_finding['Region_Nam']==region]['share_n_finding'].values[0]),
                                    float(share_n_finding[share_n_finding['Region_Nam']==region]['share_destruction'].values[0])],
                                    'destruction':float(share_n_finding[share_n_finding['Region_Nam']==region]['destruction'].values[0])}

    for region in ['China','India']:  
        key_data[region+str(s_index)] = {'Downscaled Region':'Average \n'+region,
                                         'Scenario':scenario,
                                        'coordinates':[weighted_average(share_n_finding, 'share_n_finding','Workforce', ['IND' if region =='India' else 'CHN'][0]),
                                                       weighted_average(share_n_finding, 'share_destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])],
                                    'destruction':weighted_average(share_n_finding, 'destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])}
        

    
    ax.set_title(Scenarios_names[s_index])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([65,140])
    ax.set_ylim([7,55])


Cscenario = ['#F18872','#A7C682','#97CEE4','red','green','blue']

cax = fig1.add_axes([0.85, 0.3+0.075, 0.25, 0.25])
cax = xycmap.bivariate_legend(ax=cax, sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim) #xlims=(0,0.5),ylims=(0,0.06)
cax.set_xlabel('Share of workers not \n finding new employment',fontsize=11)
cax.set_ylabel('Decrease in relative coal jobs',fontsize=11)
if min(n)>5:    
    cax.set_xticks([interpol(x,b_xlim,cax.get_xlim()) for x in [0.5,0.7,0.9]])
    cax.set_yticks([interpol(np.log(x),b_ylim,cax.get_ylim()) for x in [1e-3,5e-3,0.01,0.05]])
    cax.set_xticklabels(["50%","70%","90%"],fontsize=11)
    cax.set_yticklabels(["0.1%","0.5%","1%","5%"],fontsize=11)
    
for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [0,1,2]:
        xs.append(interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)
for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [3,4,5]:
        xs.append(interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)

for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [0,1,2,3,4,5]:
        xs.append(interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
        cax.scatter(xs[-1],ys[-1],color=Colors[key_data[region+str(s_index)]['Scenario']],marker='o',s=key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)
    cax.annotate(region,(xs[0],ys[0]-0.1),ha='center',va='top',fontsize=8)


alines = []
alines.append(cax.scatter([], [], color=Colors['NPI'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NDC'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NZ'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NPI_gem'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NDC_gem'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Colors['NZ_gem'], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color='white', marker='o',s=0))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=2e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=4e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=6e5/1200))
labels = ['NPi','NDC LTS', '1.5°C','NPi GEM','NDC GEM','1.5°C GEM','Job losses \n [people]','200k','400k','600k']
cax.legend(handles=alines,
           labels=labels,
           loc='center right',
           bbox_to_anchor=(1.5, 0.5),
           frameon=False,
           fontsize=8)

fig1.subplots_adjust(wspace=-0.45,hspace=0.2)

#%% 1.2) Structural change
# ===========================================================================================================================

# See dedicated file
# Here we map the evolution of unemployment across regions

# ===========================================================================================================================

#  Mapping Unemployment
Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)

zlim = [0, 35]
tickz = np.array([5]+list(np.arange(25,176,50)))

Ts = [2020, 2035, 2050]
Scenarios = ['NPI','NDC','NZ']
Scenarios_names = ['NPI','NDC','NZ']
fig, axs = plt.subplots(len(Scenarios),
                        len(Ts),
                        figsize=(6.7, 5))


Data_Chn = []
Data_Ind = []
for s_index, scenario in enumerate(Scenarios):
    for ind_t,t in enumerate(Ts):    
        ax = axs[s_index][ind_t]
        Unemployment = []
        
        for region in Regions:

            U = Result_data[(Result_data['Downscaled Region'] == region)
                            & (Result_data['Variable'] == 'Unemployment|Downscaled') &
                            (Result_data['Scenario'] == scenario)].values[0][4+t-2015]
            LF = Result_data[(Result_data['Downscaled Region'] == region)
                            & (Result_data['Variable'] == 'Labour Force|Downscaled') &
                            (Result_data['Scenario'] == scenario)].values[0][4+t-2015]
            u = 100*U / LF

            if u > 0:
                Unemployment.append(float(u))
            else:
                Unemployment.append(np.nan)

            if Result_data[(Result_data['Downscaled Region'] == region
                            )]['Region'].values[0] == 'CHN':
                Data_Chn.append(Unemployment[-1])
            else:
                Data_Ind.append(Unemployment[-1])
        Regions[Regions == 'Odisha'] = 'Odisha'
        Unemployment = pd.DataFrame(data=np.array([list(Regions), Unemployment]).transpose(),
                            columns=['Region_Nam', 'Unemployment'])

        Asia_Data = Asia.merge(Unemployment, on='Region_Nam')

        Asia_Data['Unemployment'] = pd.to_numeric(Asia_Data['Unemployment'], errors='coerce')

        if (ind_t == 0) & (s_index == 0):
            cax = ax.inset_axes([-0.5, -2.5, 0.09, 3.5])
            cax.set_ylabel('Unemployment rate [%]')

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

        if ind_t == 0:
            ax.set_ylabel(scenario)
        if s_index == 0:
            ax.set_title(str(t))


axn = axs[1,2].inset_axes([1.3, -1, 3, 3])
us = {}
for s_i, scenario in enumerate(Scenarios):
    
    for c_i, country in enumerate(['China','India']):
        U = np.array(Result_data[(Result_data['Downscaled Region'] == country)
                            & (Result_data['Variable'] == 'Unemployment|Downscaled') &
                            (Result_data['Scenario'] == scenario)].values[0][6:])
        LF = np.array(Result_data[(Result_data['Downscaled Region'] == country)
                            & (Result_data['Variable'] == 'Labour Force|Downscaled') &
                            (Result_data['Scenario'] == scenario)].values[0][6:])
        u = 100*U / LF
        us[(country,scenario)] = u

        if s_i !=0:
            u2050 = us[(country,scenario)][2050-2015]-us[(country,'NPI')][2050-2015]
            print(f'In 2050 in scenario {scenario} the unemployment rate in {country} has increased by is {u2050:.1f} points')

        axn.plot(T,u,label=country,color=Colors[scenario],linestyle=['-','--'][c_i])
axn.set_ylabel('Unemployment rate [%]')
axn.set_ylim([0,14])
# %%
