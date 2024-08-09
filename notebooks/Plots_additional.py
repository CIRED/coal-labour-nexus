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
import os
import matplotlib.pyplot as plt
import matplotlib
import xycmap
from matplotlib import ticker
import seaborn as sns

# Setting plotting parameters
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
#%% Importing module results
T = range(2015, 2101)
T = np.array(T)

file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NDC','NPI','NZ']] for y in ['','_PG0','_R55','_gem','_EW']]).flatten())

Result_data = []
for file in file_name:
    Result_data.append(pd.read_csv(file))#, dtype=str))

Result_data = pd.concat(Result_data, ignore_index=True)

Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].applymap(lambda x: str(x).replace('D', 'E'))
Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].apply(pd.to_numeric, errors='coerce')



#%% Importing Imaclim results

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


# %% 5) Comparing power generation technology costs with AR6
import pyam

# Check available variables 
conn = pyam.iiasa.Connection()
conn.connect('ar6-public')
available_variables = conn.variables()

cost_type_ar6 = ['Capital Cost|Electricity|','OM Cost|Fixed|Electricity|']
technologies_ar6 = ['Solar|PV','Nuclear','Gas|w/o CCS','Wind|Offshore','Wind|Onshore']

variables_ar6 = [x+y for y in technologies_ar6 for x in cost_type_ar6]
variables_ar6 += ["Price|Primary Energy|Coal","Price|Primary Energy|Gas"]


chn_name ='Countries of centrally-planned Asia; primarily China'
ind_name ='Countries of South Asia; primarily India'
regions_ar6 = ['World',ind_name]

df = pyam.read_iiasa( 'ar6-public', region=regions_ar6, variable=variables_ar6)

Policies = ['P1a','P1b']

fig, axs = plt.subplots(2,int(round(len(variables_ar6)/2)), figsize=(15, 10))
axs = axs.flatten()
for ind_var, var in enumerate(variables_ar6):
    ax = axs[ind_var]

    for i_p, pol in enumerate(Policies):
        df.filter(variable=var, region='World', Policy_category=pol).plot.line(
                color=sns.color_palette()[[4,2][i_p]], ax=ax,
                alpha=0.5, linewidth=0.2, legend=False
            )
        
    
        q50 = df.filter(variable=var, region='World', Policy_category=pol).compute.quantiles([0.5]).timeseries()

        ax.plot(q50.columns, q50.values[0], color = sns.color_palette()[[4,2][i_p]],linewidth=[1,1.5][i_p])

        
        y = Imaclim_data[(Imaclim_data['Scenario']==['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus'][i_p]) & (Imaclim_data['Region']=='World') & (Imaclim_data['Variables']==var)].values[0][5:]
        ax.plot(T,y, color=['red','green'][i_p])

    if "Wind" in var:
        var= var.replace('|Onshore','')
        var= var.replace('|Offshore','')
    ax.set_title(var,fontsize=7)
    ax.set_ylim([0,[6e3,200,2e4,5e2,25e2,1e2,1e5,200,5e4,1e2,25,25][ind_var]])
 

fig.subplots_adjust(hspace=0.2, wspace=0.5)

#%% 7) What replaces coal for electricity generation in China and India under the NDC scenario


countries = ['CHN','IND']
variables = [ 'Secondary Energy|Electricity|Biomass',
 'Secondary Energy|Electricity|Coal',
 'Secondary Energy|Electricity|Gas',
 'Secondary Energy|Electricity|Geothermal',
 'Secondary Energy|Electricity|Hydro',
 'Secondary Energy|Electricity|Nuclear',
 'Secondary Energy|Electricity|Ocean',
 'Secondary Energy|Electricity|Oil',
 'Secondary Energy|Electricity|Other',
 'Secondary Energy|Electricity|Solar',
 'Secondary Energy|Electricity|Wind',]

colors = sns.color_palette()+sns.color_palette('pastel')

fix, axs = plt.subplots(1,2,figsize=(10,15))

for ind_country, country in enumerate(countries):
    ax = axs[ind_country]
    for ind_var, variable in enumerate(variables):
        y = Imaclim_data[(Imaclim_data['Scenario']=='WO-NDCLTT-ElecIndus') & (Imaclim_data['Region']==country) & (Imaclim_data['Variables']==variable)].values[0][5:]
        ax.plot(T,y,label=variable,color=colors[ind_var])
    
    ax.set_label("EJ/yr")

ax.legend()

#%% 8) Destination graph for Shanxi and Jharkhand only

Provinces = [{'Shanxi':'a'}, {'Jharkhand':'b'}]

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
    region = ['Shanxi', 'Jharkhand'][c_index]
    
    for stype_index in [0,1]:
        ax = axs[c_index][stype_index]
        t0 = 2020
        t1 = [2030,2050][stype_index]
        if stype_index == 0:
            ax.set_ylabel(region+'\nMillion workers')
        Scenarios = Scenarioss[stype_index]
        Sc = Scenarios
        X = Xs[stype_index]
        for s_index, Scenario in enumerate(Scenarios):
            
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

        ax.set_ylim([-0.5, 1.1])

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


#%% 9) Comparing coal and gas trajectories in India
country = "IND"
Scenarios = Imaclim_data.Scenario.unique()
Variables = [
    'Secondary Energy|Electricity|',
    'Capacity|Electricity|',
    'Price|Primary Energy|',
    'Capital Cost|Electricity|',
    'OM Cost|Fixed|Electricity|',
    ]

suffix = ['']*3+['|w/o CCS']*2
Fuels = {'Gas':'grey','Coal':'black'}

fig, axs = plt.subplots(len(Variables),len(Scenarios))

for ind_scenario, scenario in enumerate(Scenarios):
    for ind_variable, variable in enumerate(Variables):
        ax = axs[ind_variable,ind_scenario]

        for fuel, fuel_color in Fuels.items():
            y = [
                float(x) for x in Imaclim_data[
                    (Imaclim_data.Variables == variable+fuel+suffix[ind_variable])&
                    (Imaclim_data.Scenario== scenario)&
                    (Imaclim_data.Region==country)].values[0][5:]
            ]
            ax.plot(T,y,color=fuel_color)

        if ind_scenario == 0:
            ax.set_ylabel(variable+'\n'+Imaclim_data[(Imaclim_data.Variables == variable+fuel+suffix[ind_variable])]['Unit'].values[0])
        if ind_variable == 0:
            ax.set_title(scenario)



# %% 10) Influence of phase-out rate on unemployment
'''
Share of workers leaving into unemployment against phase out rate between 2025 and 2045. 
The phase out rate is defined as the production difference in coal production between two subsequent years divided by the 2015 level. 
Phase out rate is calculated based on the 10 year rolling average of production to account for long-term phase-down trajectories.
'''
Colors=defining_waysout_colour_scheme()
    
Countries = ['CHN','IND']
exclude_downscaled_regions = ['China','India']
destination_variables = [x for x in Result_data.Variable.unique() if 'Coal Worker Destination' in x]


Scenarios = {
    'NPI':'WO-NPi-ElecIndus',
    'NDC':'WO-NDCLTT-ElecIndus',
    'NZ':'WO-15C-ElecIndus',
}

t0 = 2025-2015
t1 = 2045-2015

def calc_phase_down_rate(Imaclim_data,rolling_window):
    Phase_out_rate = Imaclim_data[
        (Imaclim_data.Variables=="Output|Coal") 
        &(Imaclim_data.Region==country)
        ].set_index('Scenario').drop(['Model','Variables','Unit','Region'],axis=1)
    #Calculating rolling average
    Phase_out_rate = Phase_out_rate.apply(pd.to_numeric, errors='coerce').rolling(window=rolling_window, axis=1, min_periods=1, center=True).mean()
    #Taking the year-on-year difference
    Phase_out_rate = Phase_out_rate.diff(axis=1)
    #Normalizing by reference year production
    Phase_out_rate = Phase_out_rate/Imaclim_data[
        (Imaclim_data.Variables=="Output|Coal") 
        &(Imaclim_data.Region==country)
        ]['2015'].values[0]
    return Phase_out_rate
    


fix, axs = plt.subplots(1,len(Countries),figsize=(10,5.5))
for ind_country, country in enumerate(Countries):
    ax = axs[ind_country]
    Phase_out_rate = calc_phase_down_rate(Imaclim_data,7)

    Destination = Result_data[(Result_data.Variable.isin(destination_variables))&(Result_data.Region==country)&(~Result_data['Downscaled Region'].isin(exclude_downscaled_regions))]
    Share_U = Destination[Destination.Variable=='Coal Worker Destination|Unemployment'].drop(['Model','Region','Downscaled Region','Variable','Unit'],axis=1).groupby('Scenario').sum()/Destination.drop(['Model','Region','Downscaled Region','Variable','Unit'],axis=1).groupby('Scenario').sum()

    for down_scen, im_scen in Scenarios.items():
        ax.scatter(Phase_out_rate.loc[im_scen,:].iloc[t0:t1],Share_U.loc[down_scen,:].iloc[t0:t1],color=Colors[down_scen])
        ax.axhline(y=np.mean(Share_U.loc[down_scen,:].iloc[t0:t1]),linewidth=0.75,linestyle="--",color=Colors[down_scen])
        ax.axvline(x=np.mean(Phase_out_rate.loc[im_scen,:].iloc[t0:t1]),linewidth=0.75,linestyle="--",color=Colors[down_scen])

    ax.set_xlim([-0.1,0.1])
    ax.set_ylim([-0.1,1])
    ax.axhline(y=0,color='k',linewidth=0.75)
    ax.axvline(x=0,color='k',linewidth=0.75)
    ax.set_title(['China','India'][ind_country])
    ax.set_xlabel('Phase down rate [-]')
    ax.set_ylabel('Share of workers going into unemployment')

# %% 11) Plotting energy mix


Fuels =  ['Coal','Oil','Gas','Nuclear','Hydro','Biomass','Solar','Wind']

Primary_energy_variables = ['Primary Energy|'+x for x in Fuels]
Elec_variables = ['Secondary Energy|Electricity|'+x for x in Fuels]

Countries = ["CHN",'IND']

Scenarios = [
            'WO-NPi-ElecIndus',
            'WO-NDCLTT-ElecIndus',
            'WO-15C-ElecIndus',
            ]


def Energy_colors():
    Energy_colors = {
        'Biomass':'green',
        'Coal':'black',
        'Gas':'lightgrey',
        'Hydro':'navy',
        'Nuclear':'purple',
        'Solar':'gold',
        'Wind':'blue',
        'Geothermal':'pink',
        'Non-Biomass Renewables':'pink',
        'Oil':'dimgrey',
        'Other':'pink',
        'Ocean':'pink'
    }
    return Energy_colors

def get_data_stack(data):
    data = np.array(data)
    data_shape = np.shape(data)
    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)

    # Re-merge negative and positive data.
    row_mask = (data < 0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data
    return data_shape, data_stack

Variables = Primary_energy_variables
def plot_energy_mix(Imaclim_data,Countries,Scenarios,Energy_colors,Variables):
    

    fig, axs = plt.subplots(len(Countries),len(Scenarios))
    for ind_scenario, scenario in enumerate(Scenarios):
        for ind_country, country in enumerate(Countries):
            ax = axs[ind_country,ind_scenario]

            data = []

            for variable in Variables:
                data.append(
                    [float(x) for x in Imaclim_data[(Imaclim_data.Variables==variable)&
                                (Imaclim_data.Scenario==scenario)&
                                (Imaclim_data.Region==country)].values[0][5:]]
                )   

        
            data_shape, data_stack = get_data_stack(data)
            alines = []
            
            for i in np.arange(0, data_shape[0]):
                alines.append([
                    ax.bar(T,
                            data[i],
                            bottom=data_stack[i],
                            width=1,
                            alpha=0.6,
                            label = Variables[i],
                            color = Energy_colors[Variables[i].split('|')[-1]]
                            )
                ])

    for ind_country,_ in enumerate(Countries):
        y_max = max([ax.get_ylim()[1] for ax in axs[ind_country,:]])
        [ax.set_ylim([0,y_max]) for ax in axs[ind_country,:]]

    [axs[0,ind_scenario].set_title(scenario) for ind_scenario,scenario in enumerate(Scenarios)]
    [axs[ind_country,0].set_ylabel(country+'\n'+Imaclim_data[Imaclim_data.Variables==Variables[0]]['Unit'].values[0]) for ind_country,country in enumerate(Countries)]

    fig.legend([x[0] for x in alines],Variables,
            loc='center right',
            bbox_to_anchor=(1.4, 0.5),)
    
    return fig

fig = plot_energy_mix(Imaclim_data,Countries,Scenarios,Energy_colors(),Primary_energy_variables)
fig = plot_energy_mix(Imaclim_data,Countries,Scenarios,Energy_colors(),Elec_variables)
