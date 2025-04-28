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
import matplotlib.pyplot as plt
import matplotlib
import xycmap
from matplotlib import ticker
import seaborn as sns
import pyam
import string
import re
import matplotlib.patches as mpatches

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



#======
file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']] for y in ['','_tra','_PG0','_R55','_min','_max','_C0I0_P0','_C0I0_P100','_C20I5_P0','_C20I5_P100','_C40I10_P100','_pop','_dose','_dpop','_sc']]).flatten())

Result_data = []
for file in file_name:
    data = pd.read_csv(file)
    data.Scenario = data.Scenario# + '_old'
    Result_data.append(data)#, dtype=str))


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

# ===========================================================================================================================
#Unit
exa2giga        =                 1e9 # G / E
tep2gj          =              41.855 # GJ/tep
mtoe2gj         =        1e6 * tep2gj # GJ/Mtep
mtoe2ej         =  mtoe2gj / exa2giga # EJ/Mtep

Colors = pf.defining_waysout_colour_scheme()
# ===========================================================================================================================

Region_indexes = pd.read_csv('../coal.labour.nexus/data/Coal_labour/Downscaling/Indexes.csv')


#%%
# ===========================================================================================================================
# ===========================================================================================================================
#                                              Plots for the core of the article
# ===========================================================================================================================
# ===========================================================================================================================

#%% MAIN ====================================================================================
# ===========================================================================================================================
#%% Main 1-  National employment trajectories
# ===========================================================================================================================
Step = 5
Show_uncertainty = False
Show_alternatives = False
Show_supply = False
fig = pf.plot_national_employment_trajectories(T,Result_data,Historical_data,Step,Show_alternatives,Show_supply,Show_uncertainty)
pf.save_figure(fig,'M1_employment_trajectories','svg',dpi=600)


#%% Main 2 - Subnational employment trajectories
# ===========================================================================================================================
Show_alternatives=False
representation = 1
for ind_country, country in enumerate(['CHN','IND']):
    grid_size = pf.regional_grid(representation)[2][ind_country]
    
    fig, axs = plt.subplots(grid_size[0],grid_size[1],figsize=(26,15))
    
    pf.Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,False,Show_alternatives=Show_alternatives,grid_scale_same=False,representation=representation)
    
    pf.save_figure(fig,'M2_Grid_employment_'+['','alternatives_'][Show_alternatives]+country+str(representation),'svg')

# Outputing some results for text
# Phase out years in Shanxi and Jharkhand
for region in ['Shanxi','Jharkhand']:
    l = Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=='Employment|Coal|Downscaled')&(Result_data.Scenario=="NZ")].values[0][6:] 
    t95 = T[l<=l[5]*0.05][0]
    print(f'Reaching 1.5°C with limited CCS availability necessitates a near total phase-out by {t95} in {region} ')

    LF = Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=='Labour Force|Downscaled')&(Result_data.Scenario=="NZ")].values[0][6:]
    print(f'{region} is one of the most exposed with {100*(l[5]/LF[5]):0.1f}% of 2015 labour force working in the coal sector ')
    
    reduc = (l[5]-l[10])/LF[5]*100
    print(f'Even with access to CCS, under 1.5°C, {region} sees {reduc:0.1f}% of its labour force laid off between 2020 and 2025.')

    l = Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=='Employment|Coal|Downscaled')&(Result_data.Scenario=="NDC")].values[0][6:] 
    reduc = (1-l[T==t95]/l[5])*100
    l = Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=='Employment|Coal|Downscaled')&(Result_data.Scenario=="NDC_CCS1")].values[0][6:] 
    reducCCS = (1-l[T==t95]/l[5])*100
    print(f'By that date, coal labour must have been reduced by {reduc[0]:0.0f}%  ({reducCCS[0]:0.0f}% with CCS) to achieve NDC-LTT goals')


for region in ['Henan','Shandong']:
    for scenario in ['NPI','NDC','NZ']:
        l = Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=='Employment|Coal|Downscaled')&(Result_data.Scenario==scenario)].values[0][6:] 
        t95 = T[l<=l[5]*0.05][0]
        print(f'{scenario} a near total phase-out by {t95} in {region} ')

#%% Main 3 - Exposure of regions to coal transition
# ===========================================================================================================================

def exposure_scatter(T,Result_data):
    All_data = pf.find_destination_data(Result_data)
    
    T1 = 2035
    fig, (ax, ax2) = plt.subplots(1,2,figsize=(7,6),width_ratios=[1, 0.15])
    ax.axvline(x=0, color='k', linestyle='-',linewidth = 0.5,zorder=0)
    for ind_region, region in enumerate(All_data.index):
        ax.scatter(All_data.loc[region, 'Workforce'],len(All_data.index)-ind_region-1, s=25,color='k', label=region)
        ax2.scatter(All_data.loc[region, 'Workforce'],len(All_data.index)-ind_region-1, s=25,color='k', label=region)
        for ind_scenario, scenario in enumerate(Scenarios_names):
            ax.scatter(All_data.loc[region, scenario+'\n'+str(T1)],len(All_data.index)-ind_region-1, s=25, label=region, color=pf.defining_waysout_colour_scheme()[Scenarios[ind_scenario]])
        ax.plot([min([All_data.loc[region, x+'\n'+str(T1)] for x in Scenarios_names]),max([All_data.loc[region, x+'\n'+str(T1)] for x in Scenarios_names])],
                [len(All_data.index)-ind_region-1]*2,color='gray',zorder=0,alpha=0.4,linewidth=3.75)


    ax.set_yticks(range(len(All_data.index)))
    ax.set_yticklabels(All_data.index[::-1])
    ax.set_ylim([-0.5,33.5])
    ax.set_xticklabels([f'{x*100:.0f}%' for x in ax.get_xticks()])
    ax.set_xlim([ax.get_xticks()[0],4.9e-2])
    ax2.set_xlim([5.1e-2,8e-2])
    ax2.set_xticklabels([f'{x*100:.0f}%' for x in ax2.get_xticks()])

    ax.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    
    ax.tick_params(labelright=False)  # don't put tick labels at the top
    ax2.set_yticks([])

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)


    ax.set_xlabel('Share of labour force')

    alines = []
    alines.append(ax.scatter([],[],color='k',label='Share of labour force in coal workforce'))
    for ind_scenario, scenario in enumerate(Scenarios):
        alines.append(ax.scatter([],[],
                                color=pf.defining_waysout_colour_scheme()[scenario],
                                label=['NPi','NDC-LTT','NDC-LTT w/CCS','1.5°C','1.5°C w/CCS'][ind_scenario]))

    labels = [la.get_label() for la in alines]
    handles = [label for label in alines]

    fig.legend(handles=alines,
                labels=labels,
                    loc='upper center',
                bbox_to_anchor=(0.5, 0.05),
                frameon=False,
                ncol=2)
    return fig

fig = exposure_scatter(T,Result_data)
pf.save_figure(fig,'M3_Exposure_scatter','svg')

#%% Main 4 - Boxplot of share not finding per scenario
# ===========================================================================================================================
def boxplot_share_not_finding():
    all_region_names = False

    region_indices = pf.def_region_indices()
    destination_variables = [x for x in Result_data.Variable.unique() if 'Coal Worker Destination' in x if 'Retire' not in x and 'Hire' not in x]

    Countries = ["CHN",'IND']
    exclude_downscaled_regions = ['China','India']

    Scenarios = [
                'NPI',
                'NDC_CCS1',
                'NDC',
                'NZ_CCS1',
                'NZ',
                ][::-1]


    loc = {"Shanxi":(0.75,-0.1),
        "Jharkhand":(0.7,1.4),
        "Odisha":(0.85,0.8),
        'Chhattisgarh':(0.1,1)}


    t0s = [2020, 2020]
    t1s = [2030,2050] 
    fig, axs = plt.subplots(2,2,figsize=(16/2.54,9/2.54))



    for ind_t,(t0,t1) in enumerate(zip(t0s,t1s)):
        
        x=0
        
        
        for ind_country, country in enumerate(Countries):
            for ind_scenario, scenario in enumerate(Scenarios):

                if type(t1) is str:
                    threshold = 0.8
                    T1 = pf.finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==exclude_downscaled_regions[ind_country])&(Result_data.Scenario==scenario)],threshold,T)

                else:
                    T1 = t1

                Share_U, Share_R, TotDestination = pf.destination_share(Result_data,country,scenario,t0,T1)

                if (scenario == "NZ") & (t1==2030):

                    u_national = sum((TotDestination*Share_U).dropna())/sum(TotDestination.dropna())
                    r_national = ((TotDestination*Share_R/(1-Share_R))).dropna()
                    r_national = sum(r_national.dropna())/(sum(r_national.dropna())+sum(TotDestination.dropna()))
                    print(f'Under {scenario}, between {t0}-{t1} in {country}, the share of workers leaving into unemployment is {u_national:0.2f} ({min(Share_U):0.2f}-{max(Share_U):0.2f}), that\'s {sum((TotDestination*Share_U).dropna()):0.0f} workers' )
                    print(f'Under {scenario}, between {t0}-{t1} in {country}, the share of workers leaving into retirement is {r_national:0.2f} ({min(Share_R):0.2f}-{max(Share_R):0.2f})' )
                elif (scenario=="NZ") & (t1==2050):
                    region = ['Shanxi','Jharkhand'][ind_country]
                    su = Share_U[region]
                    tu = (TotDestination*Share_U)[region]
                    print(f'Under {scenario} in {region}, {su:0.2f}% (ie {tu:0.0f} workers) may not find employment')

                pos = x+[-0.35,-0.17,0,0.17,0.35][ind_scenario]

                for ind_var, var in enumerate([Share_U,Share_R]):
                    
                    ax = axs[ind_t,ind_var]
                    # ax = axs[ind_var]

                    if ind_var==0:
                        ax.arrow(0.8,0.5,0.1,0,head_width=0.03,color='k')
                        ax.text(0.85,0.34,"increased\n vulnerability",fontsize=4,fontweight='normal')
                    else:
                        ax.arrow(0.2,0.5,-0.1,0,head_width=0.03,color='k')
                        ax.text(0.11,0.34,"increased\n vulnerability",fontsize=4,fontweight='normal')


                    bxplot=ax.boxplot(var,
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
                        box.set_alpha(0.5)  
                        box.set_edgecolor('black')
                        box.set_linewidth(0.5)  # Set the edge linewidth here
                        
                    # Set the median line color to black
                    for median in bxplot['medians']:
                        median.set_color('black')


                    ax.scatter(np.mean(var),pos,s=3,color='k',zorder=2)
                    # Data points
                    for ind in var.index:
                        y = var.loc[ind]
                        xs = np.random.normal(pos, 0.05, size=1)
                        ax.scatter(y,xs,s=4.5e-5*TotDestination[ind],color=pf.defining_waysout_colour_scheme()[scenario])
                        if ind in ['Shanxi','Odisha','Jharkhand','Chhattisgarh'] and ind_scenario==0:
                            ax.text(y,xs,region_indices.loc[region_indices.Subregion_name==ind,'Subregion_iso'].values[0],fontsize=5,verticalalignment='center_baseline')
                        elif all_region_names:
                            ax.text(y,xs,region_indices.loc[region_indices.Subregion_name==ind,'Subregion_iso'].values[0].split('-')[1],fontsize=4,verticalalignment='center_baseline')
                    

            x+=1
        for ind_var in [0,1]:
            ax = axs[ind_t,ind_var]
            ax.set_yticks([0,1])
            ax.set_yticklabels(['China','India'])
            ax.axhline(y=0.5, color='k', linestyle=':',linewidth = 0.5)
            ax.set_ylim([-0.5,1.5])
            ax.set_title(str(t0)+' - '+str(t1))
            ax.set_xlim([0,1])

    if np.size(axs[0])>1:
        [ax.set_xticklabels([]) for ax in axs[0,:]]
        axs[1,0].set_xlabel('Share layoffs not finding employment')
        axs[1,1].set_xlabel('Share destruction in retirement')
    else:
        axs[0].set_xlabel('Share layoffs not finding employment')
        axs[1].set_xlabel('Share destruction in retirement')



        [ax.text(0.02,0.92, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)'])]

    #Legend
    alines = []
    for ind_scenario in range(4,-1,-1):
        scenario = Scenarios[ind_scenario]
        alines.append(ax.scatter([],[],
                                color=pf.defining_waysout_colour_scheme()[scenario],
                                label=['NPi','NDC-LTT','NDC-LTT w/CCS','1.5°C','1.5°C w/CCS'][ind_scenario]))
        
    alines.append(ax.scatter([],[],s=0,label='Number of Workers'))
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

    return fig

fig = boxplot_share_not_finding()
pf.save_figure(fig,'M4_Vulnerability_Boxplot','jpg',dpi=700)




#%% EXTENDED DATA =============================================================================
# ===========================================================================================================================

#%% ED1 - Coal production and AR6 comparison
# ===========================================================================================================================




#%% ED2 - Use of coal primary energy
# ===========================================================================================================================

fig = pf.Stacked_decomposition_coal_demand(Imaclim_data, T, Result_data, mtoe2ej)
pf.save_figure(fig,'ED2_Use_primary_energy','svg')


#%% ED3 - Mobility of laid-off coal workers
# ===========================================================================================================================
fig, data_save = pf.main_regions_destination_bars(provincesChina, provincesIndia, Result_data, T)
pf.print_destination_results(data_save)

#%% ED4 - Drivers of job cuts 
# ===========================================================================================================================




#%% SUPPLEMENTARY INFORMATION ============================================================================= 
# ===========================================================================================================================

#%% SI3 - Employment trajectories in China and India: calibration uncertainty
# ===========================================================================================================================

#%% SI4 - Labour sectoral mobility: calibration uncertainty
# ===========================================================================================================================
fig, data_save = pf.bar_calibration(Result_data, T, provincesChina, provincesIndia)




#%% SI5 - Labour sectoral mobility: retirement age sensitivity
# ===========================================================================================================================



fig, data_save = pf.bar_retirement_age(Result_data, T, provincesChina, provincesIndia)
pf.print_destination_results_retirement(data_save,t1=2050)

    

#%% SI6 - Employment trajectories: productivity growth sensitivity
# ===========================================================================================================================

#%% SI7 - Labour sectoral mobility: productivity growth sensitivity
# ===========================================================================================================================
fig, data_save = pf.bar_productivity(Result_data, T, provincesChina, provincesIndia)

#%% SI9 - National employment trajectories: supply vs demand-driven sensitivity
# ===========================================================================================================================


#%% SI10 - Subnational employment trajectories: supply vs demand-driven sensitivity
# ===========================================================================================================================
Show_alternatives=True
representation = 1

ind_country = 0
country = 'CHN'

grid_size = pf.regional_grid(representation)[2][ind_country]
fig, axs = plt.subplots(grid_size[0],grid_size[1],figsize=(26,15))

pf.Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,False,Show_alternatives=Show_alternatives,grid_scale_same=False,representation=representation)
pf.save_figure(fig,'S10_Grid_employment_'+['','alternatives_'][Show_alternatives]+country+str(representation),'svg')

#%% SI11 - Vulnerability of regions to coal transition: supply vs demand-driven sensitivity
# ===========================================================================================================================

import importlib
importlib.reload(pf)

fig = pf.boxplot_share_not_finding_demand(Result_data, T)
pf.save_figure(fig,'SI4_Boxplot_presentation','jpg',dpi=700)

#%% PRESENTATIONS ============================================================================= 
#%% P1 - Scenarios descriptions
# ===========================================================================================================================
Step = 5

Countries = ['World','CHN','IND']
Scenarios = ['WO-15C-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS0','WO-NPi-ElecIndus-CCS0',
        'WO-15C-ElecIndus-CCS1','WO-NDCLTT-ElecIndus-CCS1']


Scenarios_names = ['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS']
Variables = ['Emissions|CO2|Energy and Industrial Processes',
             'Resource|Extraction|Coal',]

fig, axs = plt.subplots(len(Variables),len(Countries),figsize=pf.standard_figure_size())

for ind_country, country in enumerate(Countries):
    for ind_var, variable in enumerate(Variables):
        ax = axs[ind_var,ind_country]
        alines = []
        for ind_scen, scenario in enumerate(Scenarios):

            y = Imaclim_data[(Imaclim_data.Variables==variable)&
                             (Imaclim_data.Scenario == scenario)&
                             (Imaclim_data.Region   == country)].values[0][5:]
            
            ax.axhline(y=0,color='k',linewidth=0.75)
            alines.append(ax.plot(T[0:-1:Step],y[0:-1:Step],label=Scenarios_names[ind_scen],color=Colors[scenario])[0])

        

pf.add_coaloutput_comparisons([[axs[1,0]],[axs[1,1]],[axs[1,2]]],Countries,[0])

[ax.set_xticks([]) for ax in axs[:-1,:].flatten()]
[ax.set_ylabel(unit) for ax, unit in zip(axs[:,0],['Emissions\n MtCO2/yr','Coal extraction\n EJ/yr','Power from\n coal\n EJ/yr'])]
[ax.set_title(country) for ax, country in zip(axs[0,:],['World','China','India'])]
[ax.set_xlim([2005,2100]) for ax in axs.flatten()]

alines.append(ax.plot([],[],color='k',label='Historical data')[0])
alines.append(ax.plot([],[],color='red',linestyle='--',label='Planned production')[0])

labels = [la.get_label() for la in alines]
handles = [label for label in alines]

fig.legend(handles=alines,
           labels=labels,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.18),
           frameon=False)

fig.set_tight_layout('tight')

pf.save_figure(fig,'0_scenario','jpg',dpi=600)

#%% MAIN ====================================================================================
if __name__ == "__main__":
    plot_main = True
    plot_extended = True
    plot_supplementary = True
    plot_presentation = False





# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================



#%%
# ===========================================================================================================================
# ===========================================================================================================================
#                                              Plots for the annex
# ===========================================================================================================================
# ===========================================================================================================================

#%% 1) Sensitivity analyses
# ===========================================================================================================================
# ===========================================================================================================================

#%%
# ===========================================================================================================================
#                                              Comparing to AR6 database
# ===========================================================================================================================
"""

Here we benchmark the WO scenarios with scenarios from the AR6 database
This requires pulling scenarios from the IIASA database using pyam

"""
#%%  Importing AR6 database

chn_name = 'Countries of centrally-planned Asia; primarily China'
ind_name = 'Countries of South Asia; primarily India'
regions_ar6 = ['World', chn_name, ind_name]

variables_ar6 = ['Primary Energy|Coal','Trade|Primary Energy|Coal|Volume',
             'Final Energy|Industry|Solids|Coal','Emissions|CO2|Energy and Industrial Processes',
             'Carbon Sequestration|CCS|Biomass','Secondary Energy|Electricity|Coal',
             'Primary Energy|Oil','Primary Energy|Gas','Unemployment|Rate','Employment|Industry|Mining',
             "Investment|Energy Supply|Extraction|Coal",'Capacity|Electricity|Coal']
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


#%% 3.1) Emissions
import importlib
importlib.reload(pf)
var = 'Emissions|CO2|Energy and Industrial Processes'
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,False)

#%% 3.2) Coal production
import importlib
importlib.reload(pf)
var = 'Output|Coal'
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)



# %% 3.2.b
import importlib
importlib.reload(pf)


var = 'Output|Coal'
fig = pf.plot_AR6_range_production(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)
pf.save_figure(fig,'ED1_Coal_Production','png',dpi=1200)

#%% 3.3) Electricity from coal
var = 'Secondary Energy|Electricity|Coal'
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)

# pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)


#%% 3.4) Investment in coal supply
var = "Investment|Energy Supply|Extraction|Coal"
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)

#%% 3.5) Investment in coal supply
var = 'Carbon Sequestration|CCS|Biomass'
var_im = 'Carbon Capture|Storage|Biomass'
regions = regions_ar6[1:3]
categories = categories[0:2]
pf.plotting_with_AR6_range(var,var_im,regions,categories,df,cols,Imaclim_data,T,False)

#%% 3.6) Investment in coal supply
var = 'Capacity|Electricity|Coal'
var_im = 'Capacity|Electricity|Coal'
regions = regions_ar6
categories = categories
pf.plotting_with_AR6_range(var,var_im,regions,categories,df,cols,Imaclim_data,T,False)


#=========================================================================================================================================================================
#=========================================================================================================================================================================

#%% =======================================================================================================

regions = ["China","India","Shanxi","Jharkhand","Inner Mongolia","Henan"]
divide_by = "2020"

def def_destruction_denominator(divide_by,df,scenario,region):
    if divide_by == "2020":
        numerator = df.loc[(scenario,region),"2020"]
    elif divide_by == "Total Destruction":
        numerator = (df.loc[('NPI_PG0',region)].values-df.loc[(scenario,region)].values) 
    else:
        return "Invalid numerator"
    return numerator


variable = 'Employment|Coal|Downscaled'


destruction_data = pd.DataFrame(columns=['Region',"Scenario",'Variable']+list(range(2015,2101)))
destruction_data.set_index(['Region',"Scenario",'Variable'],inplace=True)

diff = True

for ind_region, region in enumerate(regions):
    country = Result_data.loc[Result_data["Downscaled Region"]==region,"Region"].values[0]

    if region in ['China','India']:

        df = Result_data[(Result_data.Variable==variable)&
                        (Result_data.Region==country)&
                        (Result_data['Downscaled Region']!=region)]

        df=df.groupby(["Scenario","Model","Region","Variable","Unit"]).sum().reset_index()

        df['Downscaled Region'] = region
    else:
        df = Result_data[(Result_data.Variable==variable)&
                        (Result_data['Downscaled Region']==region)]

    df=df.set_index(['Scenario','Downscaled Region']).drop(['Model', 'Region', 'Variable', 'Unit'],axis=1)
    # NPI



    numerator =def_destruction_denominator(divide_by,df,scenario,region)  
    destruction_data.loc[(region,"NPI","NPI productivity"),:] = (df.loc[('NPI_PG0',region)].values-df.loc[('NPI',region)].values)/numerator
    destruction_data.loc[(region,"NPI","Total destruction"),:] =  (df.loc[('NPI_PG0',region)].values-df.loc[("NPI",region)].values)/numerator
    # Other scenarios
    for ind_scenario, scenario in enumerate(["NDC","NZ"]):
        destruction_data.loc[(region,scenario,"Production"),:] = (df.loc[('NPI_PG0',region)].values-df.loc[(scenario+'_PG0',region)].values)/numerator
        destruction_data.loc[(region,scenario,"NPI productivity"),:] = (df.loc[('NPI_PG0',region)].values-df.loc[('NPI',region)].values)/numerator
        destruction_data.loc[(region,scenario,"Additional productivity"),:] = (df.loc[(scenario+'_PG0',region)].values-df.loc[(scenario,region)].values-(df.loc[('NPI_PG0',region)].values-df.loc[('NPI',region)].values))/numerator
        destruction_data.loc[(region,scenario,"Total destruction"),:] =  (df.loc[('NPI_PG0',region)].values-df.loc[(scenario,region)].values)/numerator

scenarios = ["NPI","NDC","NZ"]

color_dict = {
    "Production":sns.color_palette()[1],
    "NPI productivity":sns.color_palette()[2],
    "Additional productivity":sns.color_palette()[3],
}


def fix_xticks(ax,indices):
    # Fix the tick placement issue
    tick_positions = range(0, len(indices), 10)  # Positions based on categorical indexing
    tick_labels = indices[::10]  # Corresponding year labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)  # Rotate for readability
    return ax


fig, axs = plt.subplots(len(regions),len(scenarios),figsize=(7,5))
for ind_region, region in enumerate(regions):
    for ind_scenario, scenario in enumerate(scenarios):
        ax = axs[ind_region, ind_scenario]
        df_plot = destruction_data.loc[(region, scenario, destruction_data.index.get_level_values("Variable") != "Total destruction"),:
                                ].transpose()
        df_plot.index = df_plot.index.astype(int)
        df_plot.plot.bar(stacked=True, ax=ax, width=1, alpha=0.5, color=[color_dict[var] for var in df_plot.columns.get_level_values("Variable")])

        ax.plot(destruction_data.loc[(region, scenario,"Total destruction"),:].values,color='k')

        ax.legend([])
        ax.set_ylim([-1.5,2.5])
        ax = fix_xticks(ax,df_plot.index)


[ax.set_ylabel(country+'\n [-]') for ax, country in zip(axs[:,0],regions)]
[ax.set_title(scenario) for ax, scenario in zip(axs[0,:],scenarios)]

import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color=color_dict[var], label=var, alpha=0.5) for var in df_plot.columns.get_level_values("Variable")]
ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1, 1))




