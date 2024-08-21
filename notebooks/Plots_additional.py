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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import seaborn as sns
import pyam

# Plotting functions
import plotting_functions as pf

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
provincesChina, provincesIndia = pf.defining_province_grid()

# ===========================================================================================================================
#Unit
exa2giga        =                 1e9 # G / E
tep2gj          =              41.855 # GJ/tep
mtoe2gj         =        1e6 * tep2gj # GJ/Mtep
mtoe2ej         =  mtoe2gj / exa2giga # EJ/Mtep

Colors = pf.defining_waysout_colour_scheme()
# ===========================================================================================================================
#%% 1) Different ways to look at unemployment

Sectors = ['Coal','Oil','Gas','ET','Elec','BTP','Services','Air','Mer','Ot','Agri','Indus']

cols = ['k','darkgrey','grey']+[sns.color_palette('pastel')[x] for x in [8,9,7,3,4,5,6,2,0] ]+[sns.color_palette()[1]]

Scenarios = ['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']
Scenarios_name = ['NPI','WO-NDCLTT','WO-15C']

Regions = ['China','India']

fig, axss = plt.subplots(6,3, figsize=(20/2.54,15/2.54))

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
            cumulated_data = pf.get_cumulated_array(data, min=0)
            cumulated_data_neg = pf.get_cumulated_array(data, max=0)

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
        cumulated_data = pf.get_cumulated_array(data, min=0)
        cumulated_data_neg = pf.get_cumulated_array(data, max=0)

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

fig, axs = plt.subplots(1,3, figsize=pf.standard_figure_size())

for ind_region, region in enumerate(Regions):
    ax = axs[ind_region]
    for ind_scenario, scenario in enumerate(Scenarios):
        y = Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==variable)].values[0][5:]
        ax.plot(T, y, label=Scenarios[ind_scenario], color = Colors[scenario])

[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)'])]
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
                        figsize=pf.standard_figure_size())


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

fig, axs = plt.subplots(1,2, figsize=pf.standard_figure_size())
for ind_reg, region in enumerate(Regions):
    ax = axs[ind_reg]
    base = Imaclim_data[(Imaclim_data['Scenario']=='WO-NPi-ElecIndus') & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]
    for ind_scenario, scenario in enumerate(Scenarios):
        y = (Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]-base)/base*100

        y2 = [np.mean(y[i-4:i+4]) for i in range(4,len(y)-4)]
        ax.plot(T, y, label=Scenarios[ind_scenario], color = Colors[scenario], linewidth= 0.5)
        ax.plot(T[4:-4], y2, label=Scenarios[ind_scenario], color = Colors[scenario])



        print(f'In 2050 the GDP loss in {region} is {y2[2050-2015-4]:.0f}% for {Scenarios[ind_scenario]}')
    ax.set_title(['China','India'][ind_reg])
    ax.set_ylabel('GDP loss [%]')
    ax.set_ylim([-30,15])

[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]

# %% 5) Comparing power generation technology costs with AR6

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

fig, axs = plt.subplots(2,int(round(len(variables_ar6)/2)), figsize=pf.standard_figure_size())
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

fix, axs = plt.subplots(1,2,figsize=pf.standard_figure_size())

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

Scenarioss = [ ['NPI','NDC','NZ']]*3
Scenarioss_name = [[ 'NPI', 'NDC\nLTT', '1.5°C']]*3

T0s = [2020]*3
T1s = [2030,2050,'80%']

Xs = [
    list(range(len(Scenarioss[0])))
]*3
data_save = {}
fig, axs = plt.subplots(2,
                        len(T1s),
                        figsize=pf.standard_figure_size(),
                        )
for c_index in [0, 1]:
    x = 0
    provinces = Provinces[c_index]
    region = ['Shanxi', 'Jharkhand'][c_index]
    
    for stype_index, (t0,T1) in enumerate(zip(T0s,T1s)):
        ax = axs[c_index][stype_index]


        if stype_index == 0:
            ax.set_ylabel(region+'\nMillion workers')
        else:
            ax.set_yticklabels([])
        Scenarios = Scenarioss[stype_index]
        Sc = Scenarios
        X = Xs[stype_index]
        for s_index, Scenario in enumerate(Scenarios):
            if type(T1) is str:
                threshold = 0.8
                
                t1 =pf.finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==region)&
                                (Result_data.Scenario==Scenario)],threshold,T)

            else:
                t1=T1
            data_save, alines = pf.destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
            if stype_index == 2:
                ax.text(X[s_index],data_save[(region,Scenario,t1)][:-1].sum(),t1,style='italic')
            x += 1


            if (Scenario == 'NPI')&(stype_index == 1):
                u = round(data_save[(region, Scenario,2050)][3]*1000)
                print(f'In {region}, {u} thousand workers will not find new employment by 2050')

            
        alines = [x[0] for x in alines]
        labs = [la.get_label() for la in alines]
        ax.axhline(y=0, color='k', linewidth=0.9)



        if c_index == 0:
            ax.set_title(str(t0) + '-' + str(T1))
            ax.set_xticks([])
        else:
            ax.set_xticks(X)
            ax.set_xticklabels(
            Scenarioss_name[stype_index],
            rotation=90,
            )

        ax.set_ylim([-0.15, 1.1])

fig.legend(handles=alines,
           labels=labs,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.2),
           frameon=False)

fig.subplots_adjust(hspace=0.02, wspace=0.1)


## Aditional informations ================
ds = pd.DataFrame(data_save).T
ds.columns = ['R', 'D', 'I', 'U', 'H']
s_ndc = ds.loc[('Shanxi','NDC',2050),'U']*1000
s_nze = ds.loc[('Shanxi','NZ',2050),'U']*1000
j_ndc = ds.loc[('Jharkhand','NDC',2050),'U']*1000
j_nze = ds.loc[('Jharkhand','NZ',2050),'U']*1000
print(f'That is northern Chinese province of Shanxi and the eastern Indian of Jharkhand where {s_ndc:.0f} thousand ({s_nze:.0f}) and {j_ndc:.0f} thousand ({j_nze:.0f}) workers are found not to find new employment respectively by 2050 under a NDC-LTT (1.5°C) scenario')

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
Colors=pf.defining_waysout_colour_scheme()
    
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
    


fix, axs = plt.subplots(1,len(Countries),figsize=pf.standard_figure_size())
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

[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]

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




fig = pf.plot_energy_mix(Imaclim_data,Countries,Scenarios,pf.Energy_colors(),Primary_energy_variables,T)
fig = pf.plot_energy_mix(Imaclim_data,Countries,Scenarios,pf.Energy_colors(),Elec_variables,T)


# %% 12) Boxplot of share not finding per scenario
destination_variables = [x for x in Result_data.Variable.unique() if 'Coal Worker Destination' in x if 'Retire' not in x and 'Hire' not in x]

Countries = ["CHN",'IND']
exclude_downscaled_regions = ['China','India']

Scenarios = [
            'NPI',
            'NDC',
            'NZ',
            ]



t0s = [2020,2020,2020, 2020]
t1s = [2030,2040,2050, '80%']

fig, axs = plt.subplots(2,2,figsize=pf.standard_figure_size())

for ind_t,(t0,t1) in enumerate(zip(t0s,t1s)):
    ax = axs.flatten()[ind_t]
    x=0
    
    
    for ind_country, country in enumerate(Countries):
        for ind_scenario, scenario in enumerate(Scenarios):

            if type(t1) is str:
                threshold = 0.8
                T1 = pf.finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==exclude_downscaled_regions[ind_country])&(Result_data.Scenario==scenario)],threshold,T)

            else:
                T1 = t1


            Destination = Result_data[(Result_data.Variable.isin(destination_variables))&
                                    (Result_data.Region==country)&
                                    (~Result_data['Downscaled Region'].isin(exclude_downscaled_regions))&
                                    (Result_data.Scenario==scenario)]
            U = Destination[Destination.Variable=='Coal Worker Destination|Unemployment'].drop(['Model','Region','Scenario','Variable','Unit'],axis=1).groupby('Downscaled Region').sum().loc[:,[str(x) for x in range(2020,T1)]].sum(axis=1)
            TotDestination = Destination.drop(['Model','Region','Scenario','Variable','Unit'],axis=1).groupby('Downscaled Region').sum().loc[:,[str(x) for x in range(t0,T1)]].sum(axis=1)
            TotDestination[TotDestination==0]=np.nan
            Share_U = U/TotDestination
            Share_U=Share_U.dropna()

            print(f'Under {scenario}, between {t0}-{t1} in {country}, the range of share of workers leaving into unemployment is {max(Share_U)-min(Share_U):0.2f} ({min(Share_U):0.2f}-{max(Share_U):0.2f})' )
            
            pos = x+[-0.28,0,0.28][ind_scenario]
            bxplot=ax.boxplot(Share_U,
                    positions=[pos],
                    vert=False,
                    patch_artist=True,
                    showfliers=False,
                    whiskerprops={'linestyle': 'none'},
                    capprops={'linestyle':'none'},
                    zorder=1 )
            
            # Customizing the appearance
            for box in bxplot['boxes']:
                box.set_facecolor(pf.defining_waysout_colour_scheme()[scenario])
                box.set_alpha(0.5)  # Reduce the alpha of the face color

            # Set the median line color to black
            for median in bxplot['medians']:
                median.set_color('black')

            for box in bxplot['boxes']:
                box.set_edgecolor('black')

            ax.scatter(np.mean(Share_U),pos,s=3,color='k',zorder=2)
            # Data points
            for ind in Share_U.index:
                y = Share_U.loc[ind]
                xs = np.random.normal(pos, 0.07, size=1)
                ax.scatter(y,xs,s=5e-5*TotDestination[ind],color=pf.defining_waysout_colour_scheme()[scenario])
                if ind in ['Shanxi','Odisha','Jharkhand']:
                    ax.text(y+1e-2,xs,ind,fontsize=5.5,verticalalignment='center_baseline')

        x+=1
    ax.set_yticks([0,1])
    ax.set_yticklabels(['China','India'])
    ax.set_ylim([-0.5,1.5])
    ax.set_title(str(t0)+' - '+str(t1))
    ax.set_xlim([0,1])
    if ind_t not in [2,3]:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Share layoffs not finding employment')
[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)'])]

fig.set_tight_layout('tight')
#%% 13) Consumption and extraction budget
Emi_coef = pd.read_csv("data/Emissions_coefficients.csv")


Regions = [x for x in Imaclim_data.Region.unique() if x!='World']
Scenarios = [
            'WO-NPi-ElecIndus',
            'WO-NDCLTT-ElecIndus',
            'WO-15C-ElecIndus',
            ]
fig, axs = plt.subplots(2,2,figsize=pf.standard_figure_size())

ax = axs[0,0]

data = []
for ind_region, region in enumerate(Regions):
    data.append([])
    for ind_scenario, scenario in enumerate(Scenarios):
        data[-1].append(
            Imaclim_data[(Imaclim_data.Region==region)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Emissions|CO2|Energy and Industrial Processes')].loc[:,[str(x) for x in range(2020,2101)]].sum(axis=1).values[0]
        )

for norm in [0,1]:
    ax = axs[norm,0]
    if norm == 1: 
            data=np.array(data)/np.array(data).sum(axis=0)
    data_shape, data_stack = pf.get_data_stack(data)
    alines = []

    for i in np.arange(0, data_shape[0]):
        alines.append([
            ax.bar([1,2,3],
                    data[i],
                    bottom=data_stack[i],
                    width=0.8,
                    color = pf.defining_regions_colors()[Regions[i]]
                    )
        ])



data = []
df = pd.DataFrame(columns=Scenarios)
fuels = ['Coal','Oil','Gas']
Colors = []
Alphas = []
for ind_fuel, fuel in enumerate(fuels):
    for ind_region, region in enumerate(Regions):
        data.append([])
        variable  = 'Resource|Extraction|'+fuel

        coef = Emi_coef[Emi_coef.Variable=='Coef|Emissions|CO2|'+fuel]['Value'].values[0]

        for ind_scenario, scenario in enumerate(Scenarios):
            data[-1].append(
                coef*Imaclim_data[(Imaclim_data.Region==region)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables==variable)].loc[:,[str(x) for x in range(2020,2101)]].sum(axis=1).values[0]/mtoe2ej/1e6
            )
            

        Colors.append(pf.defining_regions_colors()[region])
        Alphas.append([1,0.75,0.3][ind_fuel])

for ind_region, region in enumerate(Regions):
        data.append([])
        variable  = 'Carbon Capture'

        for ind_scenario, scenario in enumerate(Scenarios):
            data[-1].append(
                -Imaclim_data[(Imaclim_data.Region==region)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables==variable)].loc[:,[str(x) for x in range(2020,2101)]].sum(axis=1).values[0]
            )

        Colors.append(pf.defining_regions_colors()[region])
        Alphas.append(0.6)

for norm in [0,1]:
    ax = axs[norm,1]
    df = pd.DataFrame(data,columns=Scenarios,index=list(np.array([[x,y] for y in fuels+['Carbon Capture'] for x in Regions]).T))
    cum_data = [[df.loc[(slice(None),fuel),scenario].sum() for scenario in Scenarios] for fuel in fuels+['Carbon Capture']]
    if norm == 1:
        data  = (df.loc[(slice(None),fuels),:]/df.loc[(slice(None),fuels),:].sum(axis=0)).values 
        cum_data = np.array(cum_data[:-1])/df.loc[(slice(None),fuels),:].sum(axis=0).values
        
    data_shape, data_stack = pf.get_data_stack(data)
    alines = []

    for i in np.arange(0, data_shape[0]):
        alines.append([
            ax.bar([1,2,3],
                    data[i],
                    bottom=data_stack[i],
                    width=0.8,
                    alpha=1,
                    color = Colors[i]
                    )
        ])

    data_shape, data_stack = pf.get_data_stack(cum_data)
    alines = []
    for i in np.arange(0, data_shape[0]):
        alines.append([
            ax.bar(np.array([1,2,3])-0.25,
                    cum_data[i],
                    bottom=data_stack[i],
                    width=0.3,
                    color = ['k','dimgrey','silver','green'][i]
                    )
        ])


share_data = 100*pd.DataFrame(data,columns=Scenarios,index=list(np.array([[x,y] for y in fuels for x in Regions]).T))
share_data=share_data.reset_index().set_axis(labels=['Region','Fuel']+Scenarios,axis=1)
share_data=share_data.groupby("Region").sum()
ch_0 = share_data.loc['CHN',Scenarios[0]]
ch_2 = share_data.loc['CHN',Scenarios[2]]
in_0 = share_data.loc['IND',Scenarios[0]]
in_2 = share_data.loc['IND',Scenarios[2]]
print(f'The share of the extraction budget goes from {ch_0:0.1f}% to {ch_2:0.1f}% in China and {in_0:0.1f}% to {in_2:0.1f}% in India')

[ax.set_ylim([-.7e6,3.1e6]) for ax in axs[0,:]]
[ax.axhline(y=0,color='k',linewidth=0.75) for ax in axs.flatten()]
[ax.set_ylim([0,1]) for ax in axs[1,:]]
[ax.set_xticks([1,2,3]) for ax in axs.flatten()]
[ax.set_xticklabels([]) for ax in axs[0,:]]
[ax.set_xticklabels(['NPI','NDC-LTT','1.5°C']) for ax in axs[1,:]]
[ax.set_yticks([]) for ax in axs[:,1]]
axs[0,0].set_ylabel('MtC02')
axs[1,0].set_ylabel('-')
axs[0,0].set_title('Carbon consumption budget')
axs[0,1].set_title('Carbon extraction budget')

[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)'])]

fig.set_tight_layout('tight')
# %% Meant to be temporary, but checking evolution of key parameters in Madhya Pradesh

Scenarios = ['NPI','NDC','NZ']

region = 'Madhya Pradesh'

variables = ['Employment|Coal|Downscaled','Unemployment|Downscaled','Employment|Agriculture|Downscaled','Employment|Services|Downscaled','Employment|Industry|Downscaled']

fig, axs = plt.subplots(1,len(variables))
for ind_var, variable in enumerate(variables):
    ax = axs[ind_var]
    for ind_scenario, scenario in enumerate(Scenarios):
        y = [
            float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&
                                        (Result_data.Scenario==scenario)&
                                        (Result_data.Variable==variable)].values[0][6:]
        ]
        ax.plot(T,y,color=pf.defining_waysout_colour_scheme()[scenario])