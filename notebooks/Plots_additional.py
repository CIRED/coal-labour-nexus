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


file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']] for y in ['','_PG0','_R55','_min','_max']]).flatten())

Result_data = []
for file in file_name:
    data = pd.read_csv(file)
    data.Scenario = data.Scenario
    Result_data.append(data)


#======
Result_data = pd.concat(Result_data, ignore_index=True)

Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].applymap(lambda x: str(x).replace('D', 'E'))
Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].apply(pd.to_numeric, errors='coerce')



#%% Importing Imaclim results

scenarios = list(np.array([[ x + y  for x in ['WO-NPi-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0','WO-15C-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS1','WO-15C-ElecIndus-CCS1']] for y in ['']]).flatten())

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


region_indices = pd.read_csv('..\coal.labour.nexus\data\Coal_labour\Downscaling\Indexes.csv')

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

Scenarios = ['WO-NPi-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0','WO-15C-ElecIndus-CCS0']
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
        
        axs[0].set_ylabel(['NPi','NDC-LTT','NZ'][ind_scenario]+' '+['China','India'][ind_region])
        [ax.set_ylim([0,[1100,1200][ind_region]]) for ax in axs]


#%% 2) Carbon price
Regions = ['World','CHN','IND']
variable = 'Price|Carbon'
Scenarios = ['WO-NPi-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0','WO-15C-ElecIndus-CCS0']

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
    
# %% 4) "Macroeconomic cost"

Scenarios = ['WO-NDCLTT-ElecIndus-CCS0','WO-15C-ElecIndus-CCS0']
var = 'GDP|PPP'
Regions = ['CHN','IND']

fig, axs = plt.subplots(1,2, figsize=pf.standard_figure_size())
for ind_reg, region in enumerate(Regions):
    ax = axs[ind_reg]
    ax.axhline(y=0,color='k',linewidth=0.75,linestyle='--')
    base = Imaclim_data[(Imaclim_data['Scenario']=='WO-NPi-ElecIndus-CCS0') & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]
    
    df_base = Imaclim_data[(Imaclim_data['Scenario']=='WO-NPi-ElecIndus-CCS0') & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].apply(pd.to_numeric,errors='coerce').reset_index(drop=True)
    
    for ind_scenario, scenario in enumerate(Scenarios):
        y = (Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]-base)/base*100

        df = Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].apply(pd.to_numeric,errors='coerce').reset_index(drop=True)

        result = (df-df_base)/df_base*100
        smoothed_result = result.rolling(axis=1,window=8,center=True).mean()
        # ax.plot(T[T<2070], [result.loc[0,str(x)] for x in T[T<2070]], label=Scenarios[ind_scenario], color = Colors[scenario], linewidth= 0.5)
        ax.plot(T[T<2070], [smoothed_result.loc[0,str(x)] for x in T[T<2070]], label=Scenarios[ind_scenario], color = Colors[scenario])


        value = smoothed_result['2050'].values[0]
        print(f'In 2050 the GDP loss in {region} is {value:.0f}% for {Scenarios[ind_scenario]}')
    ax.set_title(['China','India'][ind_reg])
    ax.set_ylabel('GDP loss [%]')
    ax.set_ylim([-30,15])

[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]

alines = []
for Scen_type_ind, scenario in enumerate(Scenarios):
    alines.append(axs[0].plot([],[],
                              color=Colors[scenario],
                              label=['NDC-LTT','1.5°C'][Scen_type_ind],
                              linewidth = 0.5
                             ))
    # alines.append(axs[0].plot([],[],
    #                           color=Colors[scenario],
    #                           label=['NDC-LTT 8 year rolling average','1.5°C 8 year rolling average'][Scen_type_ind],
    #                          ))

labels = [la[0].get_label() for la in alines]
handles = [label[0] for label in alines]

fig.legend(handles=handles,
           labels=labels,
           loc='lower center',
           ncol=2,
           bbox_to_anchor=(0.5, -0.15),
           frameon=False)


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

        
        y = Imaclim_data[(Imaclim_data['Scenario']==['WO-NPi-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0'][i_p]) & (Imaclim_data['Region']=='World') & (Imaclim_data['Variables']==var)].values[0][5:]
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
        y = Imaclim_data[(Imaclim_data['Scenario']=='WO-NDCLTT-ElecIndus-CCS0') & (Imaclim_data['Region']==country) & (Imaclim_data['Variables']==variable)].values[0][5:]
        ax.plot(T,y,label=variable,color=colors[ind_var])
    
    ax.set_label("EJ/yr")

ax.legend()

#%% 8) Destination graph for Shanxi and Jharkhand only

Provinces = [{'Shanxi':'a'}, {'Jharkhand':'b'}]

nls = [6, 8]

Scenarioss = [ ['NPI','NDC','NZ']]*3
Scenarioss_name = [[ 'NPi', 'NDC\nLTT', '1.5°C']]*3

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
[ax.set_yticklabels([]) for ax in axs[:,[1,2]].flatten()]
[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)','e)','f)'])]


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
    'NPI':'WO-NPi-ElecIndus-CCS0',
    'NDC':'WO-NDCLTT-ElecIndus-CCS0',
    'NZ':'WO-15C-ElecIndus-CCS0',
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
            'WO-NPi-ElecIndus-CCS0',
            'WO-NDCLTT-ElecIndus-CCS0',
            'WO-15C-ElecIndus-CCS0',
            ]




fig = pf.plot_energy_mix(Imaclim_data,Countries,Scenarios,pf.Energy_colors(),Primary_energy_variables,T)
fig = pf.plot_energy_mix(Imaclim_data,Countries,Scenarios,pf.Energy_colors(),Elec_variables,T)


# %% 12) Boxplot of share not finding per scenario
destination_variables = [x for x in Result_data.Variable.unique() if 'Coal Worker Destination' in x if 'Retire' not in x and 'Hire' not in x]

Countries = ["CHN",'IND']
exclude_downscaled_regions = ['China','India']

Scenarios = [
            'NPI',
            'NDC_CCS1',
            'NDC',
            'NZ_CCS1',
            'NZ',
            ]



t0s = [2020, 2020]
t1s = [2030,2050]

fig, axs = plt.subplots(2,1,figsize=(16/2.54,9/2.54))

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
            
            pos = x+[-0.35,-0.17,0,0.17,0.35][ind_scenario]
            bxplot=ax.boxplot(Share_U,
                    positions=[pos],
                    vert=False,
                    patch_artist=True,
                    showfliers=False,
                    whiskerprops={'linestyle': 'none'},
                    capprops={'linestyle':'none'},
                    zorder=1,
                    widths=0.09 )
            
            # Customizing the appearance
            for box in bxplot['boxes']:
                box.set_facecolor(pf.defining_waysout_colour_scheme()[scenario])
                box.set_alpha(0.75)  
                
            # Set the median line color to black
            for median in bxplot['medians']:
                median.set_color('black')

            for box in bxplot['boxes']:
                box.set_edgecolor('black')

            ax.scatter(np.mean(Share_U),pos,s=3,color='k',zorder=2)
            # Data points
            for ind in Share_U.index:
                y = Share_U.loc[ind]
                xs = np.random.normal(pos, 0.05, size=1)
                ax.scatter(y,xs,s=4.5e-5*TotDestination[ind],color=pf.defining_waysout_colour_scheme()[scenario])
                if ind in ['Shanxi','Odisha','Jharkhand']:
                    ax.text(y+1e-2,xs,region_indices.loc[region_indices.Subregion_name==ind,'Subregion_iso'].values[0],fontsize=5,verticalalignment='center_baseline')

        x+=1
    ax.set_yticks([0,1])
    ax.set_yticklabels(['China','India'])
    ax.axhline(y=0.5, color='k', linestyle=':',linewidth = 0.5)
    ax.set_ylim([-0.5,1.5])
    ax.set_title(str(t0)+' - '+str(t1))
    ax.set_xlim([0,1])
    if ind_t not in [1]:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Share layoffs not finding employment')
[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]


#Legend
alines = []
for ind_scenario in range(4,-1,-1):
    scenario = Scenarios[ind_scenario]
    alines.append(ax.scatter([],[],
                             color=pf.defining_waysout_colour_scheme()[scenario],
                             label=['NPi','NDC-LTT','NDC-LTT w/CCS','1.5°C','1.5°C w/CCS'][ind_scenario]))
    
alines.append(ax.scatter([],[],s=0,label='Job destruction'))
for size in [5000,50000,500000]:
    alines.append(ax.scatter([],[],s=4.5e-5*size,color='grey',label=size))
labels = [la.get_label() for la in alines]
handles = [label for label in alines]

fig.legend(handles=alines,
           labels=labels,
            loc='center left',
           bbox_to_anchor=(1, 0.5),
           frameon=False)






fig.set_tight_layout('tight')
#%% 13) Consumption and extraction budget
Emi_coef = pd.read_csv("data/Emissions_coefficients.csv")

t0 = 2020
t1 = 2101

Regions = [x for x in Imaclim_data.Region.unique() if x!='World']
Scenarios = [
            'WO-NPi-ElecIndus-CCS0',
            'WO-NDCLTT-ElecIndus-CCS0',
            'WO-15C-ElecIndus-CCS0',
            ]


# We must first compute a emission coefficient for oil as this is not calculated direclty in Imaclim
# In Imaclim emissions from oil depend on the burning of refined fuel as part of the ET sector

scenario = Scenarios[0]
Emi_2015 = Imaclim_data[(Imaclim_data.Region=='World')&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Emissions|CO2|Energy and Industrial Processes')].loc[:,[str(x) for x in range(t0,t1)]].sum(axis=1).values[0]
fuels = ['Coal','Gas']
for fuel in fuels:
    variable  = 'Resource|Extraction|'+fuel
    coef = Emi_coef[Emi_coef.Variable=='Coef|Emissions|CO2|'+fuel]['Value'].values[0]
    Emi_2015-= coef*Imaclim_data[(Imaclim_data.Region=='World')&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables==variable)].loc[:,[str(x) for x in range(t0,t1)]].sum(axis=1).values[0]/mtoe2ej/1e6
    print(Emi_2015)
variable = 'Resource|Extraction|Oil'
Emi_coef.loc[Emi_coef.Variable=='Coef|Emissions|CO2|Oil','Value'] = Emi_2015/(Imaclim_data[(Imaclim_data.Region=='World')&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables==variable)].loc[:,[str(x) for x in range(t0,t1)]].sum(axis=1).values[0]/mtoe2ej/1e6)

fig, axs = plt.subplots(2,2,figsize=pf.standard_figure_size())

ax = axs[0,0]

data = []
for ind_region, region in enumerate(Regions):
    data.append([])
    for ind_scenario, scenario in enumerate(Scenarios):
        data[-1].append(
            Imaclim_data[(Imaclim_data.Region==region)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Emissions|CO2|Energy and Industrial Processes')].loc[:,[str(x) for x in range(t0,t1)]].sum(axis=1).values[0]
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
                coef*Imaclim_data[(Imaclim_data.Region==region)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables==variable)].loc[:,[str(x) for x in range(t0,t1)]].sum(axis=1).values[0]/mtoe2ej/1e6
            )
            

        Colors.append(pf.defining_regions_colors()[region])
        Alphas.append([1,0.75,0.3][ind_fuel])

for ind_region, region in enumerate(Regions):
        data.append([])
        variable  = 'Carbon Capture'

        for ind_scenario, scenario in enumerate(Scenarios):
            data[-1].append(
                -Imaclim_data[(Imaclim_data.Region==region)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables==variable)].loc[:,[str(x) for x in range(t0,t1)]].sum(axis=1).values[0]
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
                    color = Colors[i],
                    label = (Regions*4)[i],
                    )
        ])

    alines = alines[:len(Regions)]

    data_shape, data_stack = pf.get_data_stack(cum_data)
    for i in np.arange(0, data_shape[0]):
        alines.append([
            ax.bar(np.array([1,2,3])-0.25,
                    cum_data[i],
                    bottom=data_stack[i],
                    width=0.3,
                    color = ['k','dimgrey','silver','green'][i],
                    label = ['Coal','Oil','Gas','Sequestration'][i]
                    )
        ])
    
    if norm==0:
        salines = alines


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
[ax.set_xticklabels(['NPi','NDC-LTT','1.5°C']) for ax in axs[1,:]]
[ax.set_yticklabels([]) for ax in axs[:,1]]
axs[0,0].set_ylabel('MtC02')
axs[1,0].set_ylabel('-')
axs[0,0].set_title('Net carbon consumption budget')
axs[0,1].set_title('Carbon extraction budget')

alines = [x[0] for x in salines]
labs = [la[0].get_label() for la in salines]
fig.legend(handles=alines,
           labels=labs,
           loc='lower center',
           ncol=6,
           bbox_to_anchor=(0.5, -0.2),
           frameon=False)







[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)'])]

fig.set_tight_layout('tight')
# %% 14 ) Plotting job lost to productivity growth since 2020


Colors = pf.defining_waysout_colour_scheme()
Countries = ['China','India']
Regions = [
    Result_data[(Result_data['Region'] == ['CHN','IND'][x])&(~Result_data['Downscaled Region'].isin(['China', 'India']))]['Downscaled Region'].unique()
    for x in [0, 1]
]
Scen_type = ['NPI','NDC','NZ']
Alt_type = [
    '','_PG0']
Scenarios = [x + y for x in Scen_type for y in Alt_type]

Ralpha = [1, 0.75] * 3
Rlinestyle = ['-']*3
Rlinewidth = [1, 0.75]* 3

fig, axs = plt.subplots(2, 2, figsize=pf.standard_figure_size())
for c_index in [0, 1]:
    country = Countries[c_index]

    Variable = "Employment|Coal|Downscaled"

    ax = axs[0][c_index]


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


    Scen_y = pd.DataFrame(Scen_y, index=Scenarios)

    for Scen_type_ind,scenario in enumerate(Scen_type):
        y = (Scen_y.loc[scenario+'_PG0',:]-Scen_y.loc[scenario,:]).values[T<2070]
        
        ax.plot(T[T < 2070],
                y,
                color=Colors[scenario],
                linestyle=Rlinestyle[Scen_type_ind],
                linewidth=Rlinewidth[Scen_type_ind],
                alpha=Ralpha[Scen_type_ind],)
        
        ax.scatter(T[T < 2070][y==max(y)],max(y),s=10,color=Colors[scenario])
        ax.text(T[T < 2070][y==max(y)],max(y),f'({T[T < 2070][y==max(y)][0]},{max(y):0.2f})',fontsize=6)

    # Formatting axes
    ax.set_title(country)
    
    ax.set_ylim([-0.25, 3])

    ax = axs[1][c_index]
    for Scen_type_ind,scenario in enumerate(Scen_type):
        y = -Scen_y.loc[scenario,:].diff().values[T<2070]
        ax.plot(T[T < 2070],
                y,
                color=Colors[scenario],
                linewidth=0.6)
        y = -Scen_y.loc[scenario,:].diff().rolling(window=5,center=True).mean()[T<2070]    
        ax.plot(T[T < 2070],
                y,
                color=Colors[scenario],
                linewidth=1)
        
        average_destruction_rate = -Scen_y.loc[scenario,:].diff()[(T>=2020)&(T<2030)].mean()
        print(f'In {country}, between 2020 and 2030 under {scenario}: {(100*average_destruction_rate):0.1f} thousand jobs are destroyed per annum')

    ax.set_ylim([-0.02,0.3])

alines=[]
for Scen_type_ind, scenario in enumerate(Scen_type):
    # scenario = Scenarios[Scen_type_ind]
    alines.append(axs[0][0].plot([], [],
                              color=Colors[scenario.split('_')[0]],
                              label=['NPi','NDC-LTT','1.5°C'][Scen_type_ind],
                              linestyle=Rlinestyle[Scen_type_ind],
                linewidth=Rlinewidth[Scen_type_ind],
                alpha=Ralpha[Scen_type_ind] )[0])

[ax.axvline(x=2020, color='k', linestyle='--', linewidth=0.8) for ax in axs.flatten()]
[ax.axhline(y=0, color='k', linewidth=0.8) for ax in axs.flatten()]

axs[0,0].set_ylabel('Million workers')
axs[1,0].set_ylabel('Million workers per year')

[ax.set_yticklabels([]) for ax in axs[:,1]]
[ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)'])]
labels = [la.get_label() for la in alines]
handles = [label for label in alines]

fig.legend(handles=alines,
           labels=labels,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.1),
           frameon=False)


#%% 15) Comparing Unemployment and destruction by region 







for ind_country, country in enumerate(['CHN','IND']):
    fig, axs = plt.subplots(8,[6,8][ind_country],figsize=(20,20))
    
    pf.Grid_Unemployment_Destruction(fig,axs,T,Result_data,ind_country)



#%% 16 ) 


# Importing trajectory files for comparison



# - décroissance de la production :"Resource|Extraction|Coal|Downscaled"
#  la croissance de la productivité :  "Labour Productivity|Coal|Downscaled"


Scenarios = ['NPI','NDC','NZ']
Regions = ['China','Shanxi','Henan','India','Jharkhand','Odisha']
# Regions = ['Henan','Odisha','Xinjiang','Chhattisgarh']

Variables = ["Resource|Extraction|Coal|Downscaled"]

fig, axs = plt.subplots(len(Scenarios), len(Regions), figsize=(10,7))

for ind_scenario, scenario in enumerate(Scenarios):
    for ind_region, region in enumerate(Regions):
        ax = axs[ind_scenario, ind_region]


        y = -Result_data[
            (Result_data['Downscaled Region']==region)&
            (Result_data['Scenario']==scenario)&
            (Result_data['Variable']=="Resource|Extraction|Coal|Downscaled")
        ].iloc[:,6:].pct_change(axis=1).rolling(window=5, axis=1).mean()*100

        ax.plot(T,y.values[0],label="Production\n drop",linewidth=0.7)

        
        y = Result_data[
            (Result_data['Downscaled Region']==region)&
            (Result_data['Scenario']==scenario)&
            (Result_data['Variable']=="Labour Productivity|Coal|Downscaled")
        ].iloc[:,6:].pct_change(axis=1).rolling(window=5, axis=1).mean()*100

        ax.plot(T,y.values[0],label="Productivity\n increase",linewidth=0.7)

        y = -100*Result_data[
            (Result_data['Downscaled Region']==region)&
            (Result_data['Scenario']==scenario)&
            (Result_data['Variable']=='Employment|Coal|Downscaled')
        ].iloc[:,6:].pct_change(axis=1).rolling(window=5, axis=1).mean()

        ax.plot(T,y.values[0],label="Coal labour\n drop",color='k',linewidth=1)


        y = 100*(Result_data[
            (Result_data['Downscaled Region']==region)&
            (Result_data['Scenario']==scenario)&
            (Result_data['Variable']=='Unemployment|Downscaled')
        ].iloc[:,6:].reset_index()/Result_data[
            (Result_data['Downscaled Region']==region)&
            (Result_data['Scenario']==scenario)&
            (Result_data['Variable']=='Labour Force|Downscaled')
        ].iloc[:,6:].reset_index()).iloc[:,1:].apply(pd.to_numeric,errors='coerce')#.rolling(window=5, axis=1).mean()

        ax.plot(T,y.values[0],label="Unemployment\n rate",linewidth=1)

        




[ax.set_title(region) for ax, region in zip(axs[0,:],Regions) ]
[ax.set_ylabel(scenario+'\n [%]') for ax, scenario in zip(axs[:,0],Scenarios) ]
# 
[ax.axhline(y=0,color='k',linewidth=0.7) for ax in axs.flatten()]
# axs.flatten()[-1].legend(fontsize=5)
axs[1,-1].legend(fontsize=8, 
           bbox_to_anchor=(1, 0.75))

fig.set_tight_layout('tight')

[ax.set_ylim([-10,20]) for ax in axs.flatten()]
[ax.set_xlim([2015,2070]) for ax in axs.flatten()]

fig.suptitle('5 year rolling average of coal production drop and productivity increase \n in selected regions under central scenarios', fontsize=9)

#%%
# 17) Grid of employment in all regions

def CCS_linestyle(CCS):
    CCS_linestyle={
        '':'-',
        'CCS0':'-',
        'CCS1':':',
    }[CCS]
    return CCS_linestyle

def Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,U):
    Step = 5
    [ax.axis('off') for ax in axs.flatten()]
    regions = pf.defining_province_grid()[ind_country]

    Scenarios = ['NZ','NDC','NPI','NZ_CCS1','NDC_CCS1']

    t0 = 2020
    t1 = 2060

    for region, position in regions.items():

        ax = axs[position]
        ax.axis('on')
        if U:
            ax1, ax2 = pf.halve_axes(fig,ax)
            ax1.axhline(y=0,color='k',linewidth=0.5)
            
            variable = 'Unemployment|Downscaled'
            variable = 'Labour Productivity|Coal|Downscaled'

            for ind_scenario, scenario in enumerate(Scenarios):
                ax1.plot(
                    T[(t0<=T)&(T<=t1)][0:-1:Step],
                    Result_data[
                        (Result_data['Downscaled Region']==region)&
                        (Result_data.Scenario==scenario)&
                        (Result_data.Variable==variable)
                    ].values[0][6:][(t0<=T)&(T<=t1)][0:-1:Step],
                    color=pf.defining_waysout_colour_scheme()[scenario]
                )
            ax1.set_xticks([])
        
        else:
            ax2=ax

        
        ax2.axhline(y=0,color='k',linewidth=0.5)
        
        variable = 'Employment|Coal|Downscaled'
        for ind_scenario, scenario in enumerate(Scenarios):
  
            y = Result_data[
                    (Result_data['Downscaled Region']==region)&
                    (Result_data.Scenario==scenario)&
                    (Result_data.Variable==variable)
                ].apply(pd.to_numeric,errors='coerce')*1e-3
            

            
            if y.sum(axis=1).values[0]==0:
                ax2.set_facecolor('lightgrey')
                ax2.set_yticks([])
                ax2.set_xticks([])
            else:
                ax2.plot(
                    T[(t0<=T)&(T<=t1)][0:-1:Step],
                    y.values[0][6:][(t0<=T)&(T<=t1)][0:-1:Step],
                    color=pf.defining_waysout_colour_scheme()[scenario.split('_PG0')[0]],
                )

                # Only for regions that have coal labour in the first place we also plot without productivity growth

                y = Result_data[
                    (Result_data['Downscaled Region']==region)&
                    (Result_data.Scenario==scenario+'_PG0')&
                    (Result_data.Variable==variable)
                ].apply(pd.to_numeric,errors='coerce')*1e-3

                ax2.plot(
                    T[(t0<=T)&(T<=t1)][0:-1:Step],
                    y.values[0][6:][(t0<=T)&(T<=t1)][0:-1:Step],
                    color=pf.defining_waysout_colour_scheme()[scenario.split('_')[0]],
                    linestyle = (0,(1,3))
                )

                if region == 'Shanxi':
                    ax2.set_ylim([-20,900])
                    ax2.spines['left'].set_color('red') 
                    ax.tick_params(axis='y', colors='red')
                elif region =='Jharkhand':
                    ax2.set_ylim([-7,375])
                    ax2.spines['left'].set_color('red') 
                    ax.tick_params(axis='y', colors='red')
                else:
                    ax2.set_ylim([[-20,350],[-7,200]][ind_country])


        
        ax2.text(0.05,0.05,region,transform=ax.transAxes, fontsize= 13, fontweight='bold')
        

    
    # Legend in bottom right corner
    ax = axs[-1,[2,-2][ind_country]]
    alines = []
    for ind_scenario, scenario in enumerate(Scenarios[:5]):
        alines.append(ax.plot([],[],color=pf.defining_waysout_colour_scheme()[scenario.split('_PG0')[0]], label=['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS'][ind_scenario]))
    alines.append(ax.plot([],[], color='k', linestyle=':',label='No growth'))
    ax.legend(fontsize=25,ncol=2)
    
    return fig




for ind_country, country in enumerate(['CHN','IND']):
    fig, axs = plt.subplots(8,[6,8][ind_country],figsize=(26,20))
    
    Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,False)


# %%
# 18 ) Regional sequences

country = 'IND'
countries = ['CHN','IND']

fig, axs = plt.subplots(2,1,figsize=(10,5), height_ratios = [22/12,1])

for ind_country, country in enumerate(countries):
    ax = axs[ind_country]
    # Sort by highest 
    df = Result_data[
        (Result_data.Scenario=='NPI')&
        (Result_data.Variable=='Employment|Coal|Downscaled')&
        (~Result_data['Downscaled Region'].isin(['China','India']))&
        (Result_data['2015']!=0)&
        (Result_data.Region==country)]

    df.loc[:,'2015'] = df.loc[:,'2015'].apply(pd.to_numeric,errors='coerce')

    sorted_regions = df.sort_values(by='2015',ascending=False)['Downscaled Region'].values
    print(len(sorted_regions))

    Scenarios = ['NPI','NDC','NZ']


    for ind_region, region in enumerate(sorted_regions):

        for ind_scenario, scenario in enumerate(Scenarios):

            y = len(sorted_regions)-ind_region + [0.25,0,-0.25][ind_scenario]

            x = Result_data[
                (Result_data.Scenario==scenario)&
                (Result_data.Variable=='Employment|Coal|Downscaled')&
                (Result_data['Downscaled Region']==region)].values[0][6:]

            peak =  T[x==max(x)]
            half = T[(x<=max(x)/2)&(T>=peak)]
            half = half[0] if len(half)!=0 else 2100
            phase = T[x<=max(x)*0.05]
            phase = phase[0] if len(phase)!=0 else 2101

            ax.plot([peak,phase],[y,y],color=pf.defining_waysout_colour_scheme()[scenario],alpha=0.5)
            ax.scatter(peak,y,s=5,marker='x',color=pf.defining_waysout_colour_scheme()[scenario])
            ax.scatter(half,y,s=5,color=pf.defining_waysout_colour_scheme()[scenario])
            ax.scatter(phase,y,s=5,marker='^',color=pf.defining_waysout_colour_scheme()[scenario])


    ax.set_xlim([2010,2100])
    ax.set_yticks(range(1,1+len(sorted_regions)))
    ax.set_yticklabels(sorted_regions[::-1])

axs[0].set_xticks([])
fig.set_tight_layout('tight')

[ax.text(0.02,0.97, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]


ax = axs[1]
alines = []
for scenario in Scenarios:
    alines.append(
        ax.plot([],[], color=pf.defining_waysout_colour_scheme()[scenario],label=scenario)[0]
    )
alines.append(ax.scatter([],[],s=10,color='k',marker='x',label='Peak'))
alines.append(ax.scatter([],[],s=10,color='k',label='50% of peak'))
alines.append(ax.scatter([],[],s=10,color='k',marker='^',label='95% phase out peak'))


labels = [la.get_label() for la in alines]
handles = [label for label in alines]

fig.legend(handles=handles,
           labels=labels,
           loc='lower center',
           ncol=2,
           fontsize=7,
           bbox_to_anchor=(0.5, -0.05),#-0.15),
           frameon=False)

#%% 19 ) GDP and oil prices

Countries = ['World','CHN','IND']
Variables = ['GDP|PPP','Price|Primary Energy|Oil','Z']
Scenarios = ['WO-NPi-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS0', 'WO-15C-ElecIndus-CCS0']


fig, axs = plt.subplots(5,3,figsize=(13,5))

for ind_variable, variable in enumerate(Variables):
    for ind_country, country in enumerate(Countries):
        ax = axs[ind_variable+1,ind_country]

        for ind_scenario, scenario in enumerate(Scenarios):
            y = Imaclim_data[
                (Imaclim_data.Scenario==scenario)&
                (Imaclim_data.Variables==variable)&
                (Imaclim_data.Region==country)
            ].iloc[:,5:]

            if variable=='GDP|PPP':
                y =y.pct_change(axis=1).rolling(window=10, axis=1).mean()*100
            if variable=='Z':
                y = y.apply(pd.to_numeric,errors='coerce').rolling(window=10, axis=1).mean()
            
            ax.plot(T[T<2050],y.values[0][T<2050],color=pf.defining_waysout_colour_scheme()[scenario],linewidth=0.7)

for ind_scenario, scenario in enumerate(Scenarios):
    for ind_country, country in enumerate(Countries):
        ax = axs[4,ind_country]
        gdp = Imaclim_data[
                    (Imaclim_data.Scenario==scenario)&
                    (Imaclim_data.Variables=='GDP|PPP')&
                    (Imaclim_data.Region==country)
                ].iloc[:,5:].pct_change(axis=1).rolling(window=10, axis=1).mean()
        z = Imaclim_data[
                    (Imaclim_data.Scenario==scenario)&
                    (Imaclim_data.Variables=='Z')&
                    (Imaclim_data.Region==country)
                ].iloc[:,5:]
        gdp_z = (gdp.reset_index()/z.reset_index())
        ax.plot(T[T<2050],gdp_z.values[0][1:][T<2050],color=pf.defining_waysout_colour_scheme()[scenario],linewidth=0.7)
        ax.set_ylim([[0,-0.5,-0.5][ind_country],[0.03,2,2][ind_country]])

for ind_country, country in enumerate(Countries):
    for ind_scenario, scenario in enumerate(Scenarios[1:]):
        ax = axs[0,ind_country]
        gdp0 = Imaclim_data[
                    (Imaclim_data.Scenario==scenario)&
                    (Imaclim_data.Variables=='GDP|PPP')&
                    (Imaclim_data.Region==country)
                ].iloc[:,5:]
        gdp1 = Imaclim_data[
                    (Imaclim_data.Scenario==Scenarios[0])&
                    (Imaclim_data.Variables=='GDP|PPP')&
                    (Imaclim_data.Region==country)
                ].iloc[:,5:]
        dgdp = (gdp0.reset_index()-gdp1.reset_index())/gdp1.reset_index()
        ax.plot(T[T<2050],dgdp.values[0][1:][T<2050],color=pf.defining_waysout_colour_scheme()[scenario],linewidth=0.7)
        

[ax.set_ylim([-0.2,0.2]) for ax in axs[0,:]]
[ax.axhline(y=0,color='k',linewidth=0.75) for ax in axs[0,:]]
[ax.set_ylim([2,10.5]) for ax in axs[1,:]]
[ax.set_ylim([1,15]) for ax in axs[2,:]]
[ax.set_ylim([0.08,0.26]) for ax in axs[3,1:]]
[ax.axvline(x=2027,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,2]]
[ax.axvline(x=2038,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,2]]

[ax.axvline(x=2032,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,1]]
[ax.axvline(x=2042,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,1]]

[ax.set_title(country) for ax, country in zip(axs[0,:],Countries)]
[ax.set_xlim([2020,2050]) for ax in axs.flatten()]
axs[0,0].set_ylabel('GDP losses\n[%]')
axs[1,0].set_ylabel('GDP growth\n[%]')
axs[2,0].set_ylabel('Oil Price\n[US$2010/GJ]')
axs[3,0].set_ylabel('Z\n[-]')
axs[4,0].set_ylabel('dGDPdt/Z\n[-]')


#%% 20) Decomposing GDP differences


window = 5

Variables = ['Household','Government','Investment','Trade']

Countries = ['World','CHN','IND']
Scenarios = ['WO-NDCLTT-ElecIndus-CCS0', 'WO-15C-ElecIndus-CCS0']
npi = 'WO-NPi-ElecIndus'

fig, axs = plt.subplots(2,3,figsize=(10,5))

for ind_country, country in enumerate(Countries):
    
    C0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='Expenditure|Household')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    G0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='Expenditure|Government')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    I0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='Investment')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    Y0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='GDP|MER')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    T0 = Y0-I0-C0-G0

    for ind_scenario, scenario in enumerate(Scenarios):
        ax = axs[ind_scenario,ind_country]

        C1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Expenditure|Household')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        G1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Expenditure|Government')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        I1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Investment')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        Y1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='GDP|MER')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        T1 = Y1-I1-C1-G1


        dC = (C1-C0)/(Y0)
        dG = (G1-G0)/(Y0)
        dI = (I1-I0)/(Y0)
        dT = (T1-T0)/(Y0)
        dY = (Y1-Y0)/(Y0)

        data = [dC,dG,dI,dT]

        data_shape, data_stack = pf.get_data_stack(data)
        alines = []
        
        for i in np.arange(0, data_shape[0]):
            alines.append([
                ax.bar(T[T<2050],
                        data[i][T<2050],
                        bottom=data_stack[i][T<2050],
                        width=1,
                        alpha=0.6,
                        label = Variables[i],
                        )
            ])
        ax.plot(T[T<2050],dY[T<2050],color='k',linestyle='--')

        ax.set_ylim([[-0.06,-0.2][ind_scenario],[0.03,0.2][ind_scenario]])

handles = [aline[0] for aline in alines]
labelz = [aline[0].get_label() for aline in alines]
fig.legend(handles=handles,
           labels=labelz,
           loc='lower center',
           ncol=4,
           bbox_to_anchor=(0.5, +0.01),
           frameon=False)
[ax.axvline(x=2027,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,2]]
[ax.axvline(x=2038,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,2]]

[ax.axvline(x=2032,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,1]]
[ax.axvline(x=2042,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,1]]


[ax.set_title(country) for ax, country in zip(axs[0,:],Countries)]
[ax.set_xlim([2020,2050]) for ax in axs.flatten()]
[ax.set_ylabel(scenario) for ax, scenario in zip(axs[:,0],Scenarios)]
[ax.set_title(country) for ax, country in zip(axs[0,:],Countries)]


#%% 21) Decomposing GDP differences by sector


window = 5

Variables = ['Services','Industry and construction','Agriculture','Other']

Countries = ['World','CHN','IND']
Scenarios = ['WO-NDCLTT-ElecIndus-CCS0', 'WO-15C-ElecIndus-CCS0']
npi = 'WO-NPi-ElecIndus-CCS0'

fig, axs = plt.subplots(2,3,figsize=(10,5))

for ind_country, country in enumerate(Countries):
    
    S0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='Value Added|Services')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    M0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='Value Added|Industry and Construction')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    A0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables== 'Value Added|Agriculture')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    Y0 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==npi)&(Imaclim_data.Variables=='GDP|MER')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
    O0 = Y0-S0-M0-A0

    if min(O0)<0:
        print(f'{country} O0 smaller than one')

    for ind_scenario, scenario in enumerate(Scenarios):
        ax = axs[ind_scenario,ind_country]

        S1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Value Added|Services')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        M1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='Value Added|Industry and Construction')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        A1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables== 'Value Added|Agriculture')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        Y1 = Imaclim_data[(Imaclim_data.Region==country)&(Imaclim_data.Scenario==scenario)&(Imaclim_data.Variables=='GDP|MER')].iloc[:,5:].apply(pd.to_numeric).rolling(window=window, axis=1).mean().values[0]
        O1 = Y1-S1-M1-A1

        if min(O1)<0:
            print(f'{country} O1 {scenario} smaller than one')


        dS = (S1-S0)/(Y0)
        dM = (M1-M0)/(Y0)
        dA = (A1-A0)/(Y0)
        dO = (O1-O0)/(Y0)
        dY = (Y1-Y0)/(Y0)

        data = [dS,dM,dA,dO]

        data_shape, data_stack = pf.get_data_stack(data)
        alines = []
        
        for i in np.arange(0, data_shape[0]):
            alines.append([
                ax.bar(T[T<2050],
                        data[i][T<2050],
                        bottom=data_stack[i][T<2050],
                        width=1,
                        alpha=0.6,
                        label = Variables[i],
                        )
            ])
        ax.plot(T[T<2050],dY[T<2050],color='k',linestyle='--')

        ax.set_ylim([[-0.06,-0.2][ind_scenario],[0.03,0.2][ind_scenario]])

handles = [aline[0] for aline in alines]
labelz = [aline[0].get_label() for aline in alines]
fig.legend(handles=handles,
           labels=labelz,
           loc='lower center',
           ncol=4,
           bbox_to_anchor=(0.5, +0.01),
           frameon=False)
[ax.axvline(x=2027,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,2]]
[ax.axvline(x=2038,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,2]]

[ax.axvline(x=2032,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,1]]
[ax.axvline(x=2042,color='k',linewidth=0.75,linestyle='--') for ax in axs[:,1]]


[ax.set_title(country) for ax, country in zip(axs[0,:],Countries)]
[ax.set_xlim([2020,2050]) for ax in axs.flatten()]
[ax.set_ylabel(scenario) for ax, scenario in zip(axs[:,0],Scenarios)]
[ax.set_title(country) for ax, country in zip(axs[0,:],Countries)]

# %% 22) Emissions 

Scenarios = ['WO-NPi-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0','WO-15C-ElecIndus-CCS0']
var = 'Emissions|CO2|Energy and Industrial Processes'
var = 'GDP|PPP'
var = 'Price|Carbon'
# var = 'Secondary Energy|Electricity|Coal'
Regions = ['World','CHN','IND']

fig, axs = plt.subplots(1,3, figsize=pf.standard_figure_size())
for ind_reg, region in enumerate(Regions):
    ax = axs[ind_reg]
    ax.axhline(y=0,color='k',linewidth=0.75,linestyle='--')
    
    for ind_scenario, scenario in enumerate(Scenarios):
        y = Imaclim_data[(Imaclim_data['Scenario']==scenario) & (Imaclim_data['Region']==region) & (Imaclim_data['Variables']==var)].values[0][5:]
        ax.plot(T,y)
# %% 23) Regionally gridded destination bar graph



def Grid_Destination(fig,axs,T,Result_data,ind_country,dynamic,Scenario):
    Step = 5
    [ax.axis('off') for ax in axs.flatten()]
    regions = pf.defining_province_grid()[ind_country]

    Scenarios = ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']

    t0 = 2020
    t1 = 2060

    for region, position in regions.items():
        
        ax = axs[position]
        ax.axis('on')
        data_save = {}
        X = [0]
        s_index = 0

        y = Result_data[
                    (Result_data['Downscaled Region']==region)&
                    (Result_data.Scenario==Scenario)&
                    (Result_data.Variable=='Employment|Coal|Downscaled')
                ].loc[:,"2015"].values[0]

        if y==0:
            ax.set_yticks([])
            ax.set_facecolor('lightgrey')
            ax.set_xticks([])
        else:
            if dynamic:
                X = range(t0,t1)
                for ind_t,t in enumerate(X):
                    _, alines = pf.destination_bar(Result_data, X, T, t-1, t+1, Scenario, ax, ['China','India'][ind_country], {region:''}, data_save, ind_t)

        ax.text(0.05,0.05,region,transform=ax.transAxes, fontsize= 11, fontweight='bold')  
    return fig



Scenario = 'NPI'
for ind_country, country in enumerate(['CHN','IND']):
    fig, axs = plt.subplots(8,[6,8][ind_country],figsize=(20,20))
    
    Grid_Destination(fig,axs,T,Result_data,ind_country,True,Scenario)
