# ===========================================================================================================================
# ===========================================================================================================================  
# Regional employment vulnerability to rapid coal transition in China and India, an integrated and downscaled assessment
# ===========================================================================================================================
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

    
    Asia_Data, key_data, cmap_zip = plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names)
    

cmap, b_xlim, b_ylim, n = cmap_zip
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
data_save = {}
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

            data_save, alines = destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)

            x += 1


            if (Scenario == 'NPI')&(stype_index == 1):
                u = round(data_save[(region, Scenario,2050)][3]*1000)
                print(f'In {region}, {u} thousand workers will not find new employment by 2050')

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


## Aditional informations ================
ds = pd.DataFrame(data_save).T
ds.columns = ['R', 'D', 'I', 'U', 'H']
# Leaving into retirement
scenario = 'NDC'
t1 = 2050
chn = ds.loc[('China',scenario,t1),'R']/sum(ds.loc[('China',scenario,t1),ds.columns!='H'])*100
ind = ds.loc[('India',scenario,t1),'R']/sum(ds.loc[('India',scenario,t1),ds.columns!='H'])*100
print(f'- In all cases, we find that a significant share of workers is able to leave into retirement rather than be laid off.\n By {t1} that is the case of around {chn:.1f}% of workers in China and {ind:.1f}% of workers in India the NDC-LTT.')

# Laid-off before 2030
scenario = 'NZ'
t1 = 2030
chn = sum(ds.loc[('China',scenario,t1),~ds.columns.isin(['H','R'])])/sum(ds.loc[('China',scenario,t1),ds.columns!='H'])*100
ind = sum(ds.loc[('India',scenario,t1),~ds.columns.isin(['H','R'])])/sum(ds.loc[('India',scenario,t1),ds.columns!='H'])*100
print(f'- This is particularly true in the short run with {chn:.1f}% of exiting Chinese workers and {ind:.1f}% of \n exiting Indian workers being laid-off before {t1} under the 1.5°C scenario.')

# Unemployed by 2050
scenario = 'NPI'
t1 = 2050
chn0 = ds.loc[('China',scenario,t1),'U']/sum(ds.loc[('China',scenario,t1),ds.columns!='H'])*100
ind0 = ds.loc[('India',scenario,t1),'U']/sum(ds.loc[('India',scenario,t1),ds.columns!='H'])*100
scenario = 'NZ'
chn1 = ds.loc[('China',scenario,t1),'U']/sum(ds.loc[('China',scenario,t1),ds.columns!='H'])*100
ind1 = ds.loc[('India',scenario,t1),'U']/sum(ds.loc[('India',scenario,t1),ds.columns!='H'])*100

print(f'- In the long run, the 1.5°C scenario leads to a significant share of workers being unemployed by 2050, \n with {chn1:.1f}% of Chinese workers and {ind1:.1f}% of Indian workers not finding new employment \n against {chn0:.1f}% and {ind0:.1f}% respectively in the NPI scenario.')


# Share of lay-offs not finding new employment
scenario = 'NPI'
t1 = 2050
chn0 = 100-(ds.loc[('China',scenario,t1),'D'])/sum(ds.loc[('China',scenario,t1),['D','I','U']])*100
ind0 = 100-(ds.loc[('India',scenario,t1),'D'])/sum(ds.loc[('India',scenario,t1),['D','I','U']])*100
scenario = 'NZ'
chn1 = 100-(ds.loc[('China',scenario,t1),'D'])/sum(ds.loc[('China',scenario,t1),['D','I','U']])*100
ind1 = 100-(ds.loc[('India',scenario,t1),'D'])/sum(ds.loc[('India',scenario,t1),['D','I','U']])*100

print(f'- In the long run, the 1.5°C scenario leads to a significant share of laid-off workers not finding employment by 2050, \n with {chn1:.1f}% of Chinese workers and {ind1:.1f}% of Indian workers not finding new employment \n against {chn0:.1f}% and {ind0:.1f}% respectively in the NPI scenario.')




#%%
# ===========================================================================================================================
# ===========================================================================================================================
#                                              Plots for the annex
# ===========================================================================================================================
# ===========================================================================================================================

#%% 1) Sensitivity analyses
# ===========================================================================================================================
# ===========================================================================================================================
#%% 1.1) Calibration source
# ===========================================================================================================================


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
    Asia_Data, key_data, cmap_zip = plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names)
    

cmap, b_xlim, b_ylim, n = cmap_zip

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
#%% 1.3) Sensitivity to retirement age
# ===========================================================================================================================

Provinces = [provincesChina, provincesIndia]
Countries = ['China', 'India']
nls = [6, 8]

Scenarioss = [['NPI',  'NPI_R55'],
              ['NDC', 'NDC_R55'],
              ['NZ','NZ_R55']]

Scenarioss_name = [['NPI','Retirement \n55'],
                   ['NDC', 'Retirement \n55'],
                   ['1.5°C','Retirement \n55']]

t0 = 2020
t1 = 2050

Xs = [[x for x in range(len(Scenarioss[0]))],[x for x in range(len(Scenarioss[1]))],
    [x for x in range(len(Scenarioss[2]))]]

data_save = {}
fig, axs = plt.subplots(2,
                        3,
                        figsize=(29.21 / 2.54, 13.09 / 2.54),
      )

for c_index in [0, 1]:
    x = 0
    provinces = Provinces[c_index]
    region = ['China', 'India'][c_index]
    for stype_index in range(3):
        ax = axs[c_index][stype_index]
        if stype_index == 0:
            ax.set_ylabel(region + '\nMillion workers')
        Scenarios = Scenarioss[stype_index]
        Sc = Scenarios
        X = Xs[stype_index]
        for s_index in range(len(Scenarios)):
            Scenario = Scenarios[s_index]
            data_save, alines = destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
            x += 1

        alines = [x[0] for x in alines]
        labs = [l.get_label() for l in alines]
        ax.axhline(y=0, color='k', linewidth=0.9)
        if c_index == 1:
            ax.set_xticks(X)
            ax.set_xticklabels(
                Scenarioss_name[stype_index],
                rotation=90,
            )
        else:
            ax.set_xticks([])
            ax.set_title(['NPI', 'NDC',
                          '1.5°C'][stype_index])
        if stype_index != 0:
            ax.set_yticks([])
        ax.set_ylim([-0.6, 2.38])
        ax.set_ylim([-0.6, 3.8])

fig.legend(handles=alines,
           labels=labs,

           loc='center left',
           ncol=1,
           bbox_to_anchor=(0.9, 0.5),
           frameon=False)

fig.subplots_adjust(hspace=0.02, wspace=0.02)


ds = pd.DataFrame(data_save).T
ds.columns = ['R', 'D', 'I', 'U', 'H']
ch60 = ds.loc[('China','NZ',t1),'U']
ch55= ds.loc[('China','NZ_R55',t1),'U']

print(f'Sensitivity analysis shown in the annex where retirement age is moved from 60 to 55 show that such a policy would reduce the number of workers leaving into unemployment from {ch60} to {ch55}.')
#%% 2) Decomposing coal demand
# ===========================================================================================================================

time_gap = 10

Ts = range(2015,2095,time_gap)
Regions = ['CHN','IND']
Scenarios =['WO-NPi-ElecIndus', 'WO-NDCLTT-ElecIndus', 'WO-15C-ElecIndus']
Outputs_name = ['NPi','NDC LTT','1.5°C']

Variabless = ['Import|Coal','Final Energy|Industry|Solids|Fossil','Secondary Energy|Electricity|Coal','Power Plants|Coal','Refineries|Coal',
             'Final Energy|Commercial|Solids', 'Final Energy|Residential|Solids|Fossil','Export|Coal']

Variabless_names = ['Import','Industry','Electricity','Power plants losses','Refineries','Commercial','Residential','Export']


Variable2 = ['Primary Energy|Coal','Output|Coal']
Variable2_name = ['Consumption','Production']


Colorss = [sns.color_palette()[x] for x in [0,7,1]]+[sns.color_palette("pastel")[1]]+[sns.color_palette()[x] for x in [2,3,4]]+[sns.color_palette("pastel")[0]]

Regions = ['World', 'CHN', 'IND','USA','AFR','EUR']

fig, axs = plt.subplots(len(Regions), len(Scenarios), figsize=(20, 12))

for ind_region, region in enumerate(Regions):
    for ind_output, scenario in enumerate(Scenarios):
        
        ax = axs[ind_region][ind_output]
        data = []

        if Regions[ind_region] == 'World':
            Variables = Variabless[1:-1]
            Colors = Colorss[1:-1]
        else:
            Variables = Variabless
            Colors = Colorss

        labels = Variabless_names

        for ind_var in range(len(Variables)):
            variable = Variables[ind_var]

            if variable == "Power Plants|Coal":

                y = -np.array(Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==variable)].values[0][5:]) - np.array(
                    Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']=='Secondary Energy|Electricity|Coal')].values[0][5:])
                
                
            elif variable == "Refineries|Coal":
                y = -np.array(Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==variable)].values[0][5:])
            else:
                y = np.array(Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==variable)].values[0][5:])

            y=[sum([y[x+xs] for x in range(0,time_gap)])/time_gap for xs in range(0,len(y)-6,time_gap)]
            data.append(y)
            
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
                ax.bar(Ts,
                       data[i],
                       bottom=data_stack[i],
                       color=Colors[i],
                       width=time_gap*0.9,
                       alpha=0.8,
                       label=labels[i])
            ])
        for ind_var in range(len(Variable2)):
            variable = Variable2[ind_var]

            y = [1,mtoe2ej][ind_var]*np.array(Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==variable)].values[0][5:])

            y=[sum([y[x+xs] for x in range(0,time_gap)])/time_gap for xs in range(0,len(y)-6,time_gap)]
            alines.append(ax.plot(Ts, y, color='k', linestyle=[':','-'][ind_var], linewidth=1,label=Variable2_name[ind_var]))
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)

        if ind_region == 0:
            ax.set_title(Outputs_name[ind_output])
        if ind_output == 0:
            ax.set_ylabel(Regions[ind_region]+'\n [EJ/yr]')

        ax.set_ylim([[0,220],[-20,130],[-30,53],[0,60],[-7,15],[-7,16]][ind_region])
        ax.set_xlim([2010,2090])

handles = [aline[0] for aline in alines]
labelz = [aline[0].get_label() for aline in alines]
fig.legend(handles=handles,
           labels=labelz,
           loc='lower center',
           ncol=4,
           bbox_to_anchor=(0.5, +0.01),
           frameon=False)
fig.suptitle('Use of primary energy from coal', y=0.92, fontsize=16)


#%%
# ===========================================================================================================================
#                                              Comparing to AR6 database
# ===========================================================================================================================
"""

Here we benchmark the WO scenarios with scenarios from the AR6 database
This requires pulling scenarios from the IIASA database using pyam

"""
#%%  Importing AR6 database
import pyam
chn_name = 'Countries of centrally-planned Asia; primarily China'
ind_name = 'Countries of South Asia; primarily India'
regions_ar6 = ['World', chn_name, ind_name]

variables_ar6 = ['Primary Energy|Coal','Trade|Primary Energy|Coal|Volume',
             'Final Energy|Industry|Solids|Coal','Emissions|CO2|Energy and Industrial Processes',
             'Carbon Sequestration|CCS|Biomass','Secondary Energy|Electricity|Coal',
             'Primary Energy|Oil','Primary Energy|Gas','Unemployment|Rate','Employment|Industry|Mining']
df = pyam.read_iiasa( 'ar6-public', region=regions_ar6, variable=variables_ar6)

df.add('Primary Energy|Coal','Trade|Primary Energy|Coal|Volume',"Output|Coal",
        append=True,)



color_map = {
    "C1": "AR6-C1",
    "C2": "AR6-C2",
    "C3": "AR6-C3",
    "C4": "AR6-C4",
    "C5": "AR6-C5",
    "C6": "AR6-C6",
    "C7": "AR6-C7",
    "C8": "AR6-C8",
}

cols = {
    "C1": "#97CEE4",
    "C2": "#778663",
    "C3": "#6F7899",
    "C4": "#A7C682",
    "C5": "#8CA7D0",
    "C6": "#FAC182",
    "C7": "#F18872",
    "C8": "#BD7161",
}

pyam.run_control().update({"color": {"Category": color_map}})
categories = ['C1','C3','C6']


# %% Importing additional data
VdV = pd.read_csv('data\\global_ite2_allmodels.csv')
VdV.dropna(how = 'all',inplace=True, axis=0)
VdV.dropna(how = 'all',inplace=True, axis=1)

Garg = pd.read_csv('data\\Garg_et_al_2024.csv')
Q_WEB = pd.read_csv('data\\CoalProductionEJ_WEB.csv')
SEI = pd.read_csv('data\\SEI.csv')




#%% 3.1) Emissions
var = 'Emissions|CO2|Energy and Industrial Processes'
fig, axs = plt.subplots(len(regions_ar6),len(categories))


for i_r, reg in enumerate(regions_ar6):
    for i_c, cat in enumerate(categories):
        ax = axs[i_r, i_c]
        ax.axvline(2020, color='k', linestyle='--')
        
        ax.axhline(0, color='k', linestyle='-',linewidth=0.75)
        (
            df.filter(variable=var, region=reg, Category=cat).plot.line(
                color="Category", ax=ax,
                alpha=0, fill_between=True 
            )
        )

        tq = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().columns
        q25 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[0]
        q75 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[1]
        q50 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.5]).timeseries()

        filter=(~np.isnan(q75))&(~np.isnan(q25))
        tq = tq[filter]
        q25 = q25[filter]
        q75 = q75[filter]

        ax.fill_between(tq, q25, q75, color=cols[cat], alpha=0.5)
        ax.plot(q50.columns, q50.values[0], color=cols[cat])


        n= len(df.filter(variable=var, region=reg, Category=cat)['scenario'])

        #Plotting Van den Ven range
        if i_c in [1,2]:
            if i_r == 0:
                Regionss = ['World','WORLD']
            elif i_r == 1:
                Regionss = ['China','CHN','CHI']
            elif i_r == 2:
                Regionss = ['India','IND']    

            scenario = ['NDC_LTT','CP_EI'][i_c-1]
            ys = VdV[(VdV.Region.isin(Regionss))&(VdV.Scenario==scenario)&
                    (VdV.Variable=='Emissions|CO2|Energy and Industrial Processes')]
            for y in ys.values:
                ax.plot([float(x) for x in ys.columns[5:]], [float(x) for x in y[5:]], color='k', alpha=0.6, linewidth=0.5)
                

        if (i_c == 1)&(i_r == 2):
            for ig, sc in enumerate(Garg.columns[1:]):
                ax.plot(Garg['Year'], Garg[sc], color=([sns.color_palette()[1]]*3+[sns.color_palette('pastel')[1]]*4)[ig],  linewidth=0.5)

        # Plotting Imaclim results
        
        Output = ['WO-15C-ElecIndus','WO-NDCLTT-ElecIndus','WO-NPi-ElecIndus'][i_c]
        Region = ['World','CHN','IND'][i_r]

        y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output)&(Imaclim_data['Variables'] == var)].values[0][5:]
        
        ax.plot(T, y, color='k',linewidth=0.75)
        
        ax.text(2098,0, f'n={n}', fontsize=8, ha='right')

        ax.set_title("")
        ax.set_ylim([-[1e4,0.5e4,5e2][i_r], [7e4,2e4,5e3][i_r]])
        ax.set_xlim([2010,2105])
        ax.legend().remove()
        if i_r ==0:
            ax.set_title(['1.5°C','NDC LTT','NPI'][i_c]+' - '+cat)

        if i_r == 2: 
            ax.set_xlabel('Year')
            ax.axvline(2025, color='k', linestyle=':',linewidth=0.75)
            ax.axvline(2070, color='k', linestyle=':',linewidth=0.75)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if i_r == 1:
            ax.axvline(2060, color='k', linestyle=':',linewidth=0.75)
        
        if i_c == 0:
            ax.set_ylabel(["World","China","India"][i_r]+' \n Mt CO2/yr')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
            

fig.suptitle(var)

#%% 3.2) Coal production

var = 'Output|Coal'
fig, axs = plt.subplots(len(regions_ar6),len(categories))


cax= [[0,0,0],[0,0,0],[0,0,0]]
for i_r, reg in enumerate(regions_ar6):
    for i_c, cat in enumerate(categories):
        ax = axs[i_r, i_c]
        cax[i_r][i_c] = ax.inset_axes([0, -0.12, 1, 0.1])


for i_r, reg in enumerate(regions_ar6):
    for i_c, cat in enumerate(categories):
        ax = axs[i_r, i_c]
        ax.axhline(y=0, color='k', linewidth=0.75)
        ax2 = cax[i_r][i_c]
        ax.axvline(2020, color='k', linestyle='--')
        (
            df.filter(variable=var, region=reg, Category=cat).plot.line(
                color="Category", ax=ax,
                alpha=0, fill_between=True 
            )
        )

        tq = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().columns
        q25 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[0]
        q75 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[1]
        q50 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.5]).timeseries()

        filter=(~np.isnan(q75))&(~np.isnan(q25))
        tq = tq[filter]
        q25 = q25[filter]
        q75 = q75[filter]


        #Max
        t9_l = tq[q25==max(q25)]
        t9_m = q50.columns[(q50.values[0]==max(q50.values[0]))]
        t9_h = tq[q75==max(q75)]

        t9_l = t9_l[0] if len(t9_l)>0 else 2106
        t9_m = t9_m[0] if len(t9_m)>0 else 2106
        t9_h = t9_h[0] if len(t9_h)>0 else 2106

        ax2.bxp(
            [{'med': t9_m,'q1': t9_l,'q3': t9_h,
                'whislo': t9_l,'whishi': t9_h }],
            vert=False,patch_artist=True,boxprops=dict(facecolor=cols[cat], linewidth=0),
            medianprops=dict(color='k'),showfliers = False, showcaps = False, positions = [0.3], widths = [0.075]
        )



        #Thresholds
        for ind_box, q in enumerate([0.5,0.05]):
            t9_l = tq[(q25<=q*q25[tq==2020])&(tq>2020)]
            t9_m = q50.columns[(q50.values[0]<=q*q50.values[0][q50.columns==2020])&(q50.columns>2020)]
            t9_h = tq[(q75<=q*q75[tq==2020])&(tq>2020)]

            t9_l = t9_l[0] if len(t9_l)>0 else 2106
            t9_m = t9_m[0] if len(t9_m)>0 else 2106
            t9_h = t9_h[0] if len(t9_h)>0 else 2106

            ax2.bxp(
                [{'med': t9_m,'q1': t9_l,'q3': t9_h,
                    'whislo': t9_l,'whishi': t9_h }],
                vert=False,patch_artist=True,boxprops=dict(facecolor=cols[cat], linewidth=0),
                medianprops=dict(color='k'),showfliers = False, showcaps = False, positions = [[0.2,0.1][ind_box]], widths = [0.075]
            )



        ax.fill_between(tq, q25, q75, color=cols[cat], alpha=0.5)
        ax.plot(q50.columns, q50.values[0], color=cols[cat])

        n= len(df.filter(variable=var, region=reg, Category=cat)['scenario'])

        # Plotting Imaclim results
        Output = ['WO-15C-ElecIndus','WO-NDCLTT-ElecIndus','WO-NPi-ElecIndus'][i_c]
        Region = ['World','CHN','IND'][i_r]

        y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output)&(Imaclim_data['Variables'] == var)].values[0][5:]*mtoe2ej
        
        
        ax.plot(T, y, color='k',linewidth=0.75)
        ax.text(2098,0, f'n={n}', fontsize=8, ha='right')


        # Max
        t_m = T[y==max(y)]
        t_m = t_m[0] if len(t_m)>0 else 2106
        ax2.scatter(t_m, 0.3, color='k', marker='x', s=6, zorder=5)
        [cax[i_r][ni_c].scatter(t_m, 0.3, color='k', alpha=0.3, marker='x', s=6, zorder=5) for ni_c in range(3) if ni_c != i_c]

        # Threshold
        for ind_box, q in enumerate([0.5,0.05]):
            t_m = T[y<=q*y[T==2020]]
            t_m = t_m[0] if len(t_m)>0 else 2106
            ax2.scatter(t_m, [0.2,0.1][ind_box], color='k', marker=['o','^'][ind_box], s=6, zorder=5)
            [cax[i_r][ni_c].scatter(t_m, [0.2,0.1][ind_box], color='k', alpha=0.3, marker=['o','^'][ind_box], s=6, zorder=5) for ni_c in range(3) if ni_c != i_c]

        ax.set_title("")
        ax.set_ylim([-5, [375,130,50][i_r]])
        ax.set_xlim([2010,2105])
        ax2.set_ylim([0, 0.4])
        ax2.set_xlim([2010,2105])
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax.legend().remove()
        if i_r ==0:
            ax.set_title(['1.5°C','NDC LTT','NPI'][i_c]+' - '+cat)

        if i_r == 2:
            ax2.set_xticks(ax.get_xticks())
            ax.set_xticks([])
            ax2.set_xlim([2010,2105])
            ax2.set_ylim([0, 0.4])
            ax.set_xlabel('')
            ax2.set_xlabel('Year')
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        
        if i_c == 0:
            ax.set_ylabel(["World","China","India"][i_r]+' \n EJ/yr')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # ax2.axis('off')
for i_r, reg in enumerate(regions_ar6):
    for i_c, cat in enumerate(categories):
        ax = axs[i_r, i_c]
        ax.plot(Q_WEB['t'], [float(x) for x in Q_WEB[['World','China','India'][i_r]]], color='gray', alpha=1, linestyle='-', linewidth=1.2)

        if i_r != 1:
            ax.plot(SEI['t'][1:], [float(x) for x in SEI[['GPP','CHN_GPP','IND_GPP'][i_r]][1:]], color='red', alpha=1, linestyle='-', linewidth=1.2)

fig.suptitle(var)


#%% 3.3) Electricity from coal

var = 'Secondary Energy|Electricity|Coal'
fig, axs = plt.subplots(len(regions_ar6),len(categories))


cax= [[0,0,0],[0,0,0],[0,0,0]]
for i_r, reg in enumerate(regions_ar6):
    for i_c, cat in enumerate(categories):
        ax = axs[i_r, i_c]
        cax[i_r][i_c] = ax.inset_axes([0, -0.12, 1, 0.1])


for i_r, reg in enumerate(regions_ar6):
    for i_c, cat in enumerate(categories):
        ax = axs[i_r, i_c]
        ax.axhline(y=0, color='k', linewidth=0.75)
        ax2 = cax[i_r][i_c]
        ax.axvline(2020, color='k', linestyle='--')
        (
            df.filter(variable=var, region=reg, Category=cat).plot.line(
                color="Category", ax=ax,
                alpha=0, fill_between=True 
            )
        )

        tq = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().columns
        q25 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[0]
        q75 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[1]
        q50 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.5]).timeseries()

        filter=(~np.isnan(q75))&(~np.isnan(q25))
        tq = tq[filter]
        q25 = q25[filter]
        q75 = q75[filter]


        #Max
        t9_l = tq[q25==max(q25)]
        t9_m = q50.columns[(q50.values[0]==max(q50.values[0]))]
        t9_h = tq[q75==max(q75)]

        t9_l = t9_l[0] if len(t9_l)>0 else 2106
        t9_m = t9_m[0] if len(t9_m)>0 else 2106
        t9_h = t9_h[0] if len(t9_h)>0 else 2106

        ax2.bxp(
            [{'med': t9_m,'q1': t9_l,'q3': t9_h,
                'whislo': t9_l,'whishi': t9_h }],
            vert=False,patch_artist=True,boxprops=dict(facecolor=cols[cat], linewidth=0),
            medianprops=dict(color='k'),showfliers = False, showcaps = False, positions = [0.3], widths = [0.075]
        )



        #Thresholds
        for ind_box, q in enumerate([0.5,0.05]):
            t9_l = tq[(q25<=q*q25[tq==2020])&(tq>2020)]
            t9_m = q50.columns[(q50.values[0]<=q*q50.values[0][q50.columns==2020])&(q50.columns>2020)]
            t9_h = tq[(q75<=q*q75[tq==2020])&(tq>2020)]

            t9_l = t9_l[0] if len(t9_l)>0 else 2106
            t9_m = t9_m[0] if len(t9_m)>0 else 2106
            t9_h = t9_h[0] if len(t9_h)>0 else 2106

            ax2.bxp(
                [{'med': t9_m,'q1': t9_l,'q3': t9_h,
                    'whislo': t9_l,'whishi': t9_h }],
                vert=False,patch_artist=True,boxprops=dict(facecolor=cols[cat], linewidth=0),
                medianprops=dict(color='k'),showfliers = False, showcaps = False, positions = [[0.2,0.1][ind_box]], widths = [0.075]
            )



        ax.fill_between(tq, q25, q75, color=cols[cat], alpha=0.5)
        ax.plot(q50.columns, q50.values[0], color=cols[cat])

        n= len(df.filter(variable=var, region=reg, Category=cat)['scenario'])

        # Plotting Imaclim results
        Output = ['WO-15C-ElecIndus','WO-NDCLTT-ElecIndus','WO-NPi-ElecIndus'][i_c]
        Region = ['World','CHN','IND'][i_r]

        y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output)&(Imaclim_data['Variables'] == var)].values[0][5:]
        
        
        ax.plot(T, y, color='k',linewidth=0.75)
        ax.text(2098,0, f'n={n}', fontsize=8, ha='right')


        # Max
        t_m = T[y==max(y)]
        t_m = t_m[0] if len(t_m)>0 else 2106
        ax2.scatter(t_m, 0.3, color='k', marker='x', s=6, zorder=5)
        [cax[i_r][ni_c].scatter(t_m, 0.3, color='k', alpha=0.3, marker='x', s=6, zorder=5) for ni_c in range(3) if ni_c != i_c]

        # Threshold
        for ind_box, q in enumerate([0.5,0.05]):
            t_m = T[y<=q*y[T==2020]]
            t_m = t_m[0] if len(t_m)>0 else 2106
            ax2.scatter(t_m, [0.2,0.1][ind_box], color='k', marker=['o','^'][ind_box], s=6, zorder=5)
            [cax[i_r][ni_c].scatter(t_m, [0.2,0.1][ind_box], color='k', alpha=0.3, marker=['o','^'][ind_box], s=6, zorder=5) for ni_c in range(3) if ni_c != i_c]

            if (ind_box==1)&(i_c==1):
                print(f'Coal generation has decreased by 95% in {Region} in scenario {Output} by {t_m}')

        ax.set_title("")
        ax.set_ylim([-1, [50,20,10][i_r]])
        ax.set_xlim([2010,2105])
        ax2.set_ylim([0, 0.4])
        ax2.set_xlim([2010,2105])
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax.legend().remove()
        if i_r ==0:
            ax.set_title(['1.5°C','NDC LTT','NPI'][i_c]+' - '+cat)

        if i_r == 2:
            ax2.set_xticks(ax.get_xticks())
            ax.set_xticks([])
            ax2.set_xlim([2010,2105])
            ax2.set_ylim([0, 0.4])
            ax.set_xlabel('')
            ax2.set_xlabel('Year')
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        
        if i_c == 0:
            ax.set_ylabel(["World","China","India"][i_r]+' \n EJ/yr')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])


fig.suptitle(var)

# %%
