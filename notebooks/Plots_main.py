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
def M1():
    Step = 5
    Show_uncertainty = False
    Show_alternatives = False
    Show_supply = False
    fig = pf.plot_national_employment_trajectories(T,Result_data,Historical_data,Step,Show_alternatives,Show_supply,Show_uncertainty)
    pf.save_figure(fig,'M1_employment_trajectories','svg',dpi=600)


#%% Main 2 - Subnational employment trajectories
# ===========================================================================================================================
def M2():
    Show_alternatives=False
    representation = 1
    for ind_country, country in enumerate(['CHN','IND']):
        grid_size = pf.regional_grid(representation)[2][ind_country]
        
        fig, axs = plt.subplots(grid_size[0],grid_size[1],figsize=(26,15))
        
        pf.Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,False,Show_alternatives=Show_alternatives,grid_scale_same=False,representation=representation)
        
        pf.save_figure(fig,'M2_Grid_employment_'+['','alternatives_'][Show_alternatives]+country+str(representation),'svg')


    pf.print_subnational_employment_results(T,Result_data)

#%% Main 3 - Exposure of regions to coal transition
# ===========================================================================================================================
def M3():
    fig = pf.exposure_scatter(T,Result_data)
    pf.save_figure(fig,'M3_Exposure_scatter','svg')

#%% Main 4 - Boxplot of share not finding per scenario
# ===========================================================================================================================
def M4():
    fig = pf.boxplot_share_not_finding(Result_data,T)
    pf.save_figure(fig,'M4_Vulnerability_Boxplot','jpg',dpi=700)

#%% EXTENDED DATA =============================================================================
# ===========================================================================================================================

#%% ED1 - Coal production and AR6 comparison
# ===========================================================================================================================
def ED1():
    df, regions_ar6, cols = pf.initiate_ar6()
    categories = ['C1','C3','C6']
    var = 'Output|Coal'
    fig = pf.plot_AR6_range_production(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)
    pf.save_figure(fig,'ED1_Coal_Production','png',dpi=1200)


#%% ED2 - Use of coal primary energy
# ===========================================================================================================================
def ED2():
    fig = pf.Stacked_decomposition_coal_demand(Imaclim_data, T, Result_data, mtoe2ej)
    pf.save_figure(fig,'ED2_Use_primary_energy','svg')
    


#%% ED3 - Mobility of laid-off coal workers
# ===========================================================================================================================
def ED3():
    fig, data_save = pf.main_regions_destination_bars(provincesChina, provincesIndia, Result_data, T)
    pf.print_destination_results(data_save)
    pf.save_figure(fig,'ED3_Mobility_laid_off_workers','svg',dpi=600)

#%% ED4 - Drivers of job cuts 
# ===========================================================================================================================
# This code is outdated : proper graph was not commited and needs to be redone







#%% SUPPLEMENTARY INFORMATION ============================================================================= 
# ===========================================================================================================================

#%% SI3 - Employment trajectories in China and India: calibration uncertainty
# ===========================================================================================================================

def SI3():
    Step = 5
    Show_uncertainty = True
    Show_alternatives = False
    Show_supply = False
    fig = pf.plot_national_employment_trajectories(T,Result_data,Historical_data,Step,Show_alternatives,Show_supply,Show_uncertainty)
    pf.save_figure(fig,'SI3_employment_trajectories_calibration_uncertainty','svg',dpi=600)

#%% SI4 - Labour sectoral mobility: calibration uncertainty
# ===========================================================================================================================
def SI4():
    fig, data_save = pf.bar_calibration(Result_data, T, provincesChina, provincesIndia)
    pf.save_figure(fig,'SI4_Mobility_calibration_uncertainty','svg')



#%% SI5 - Labour sectoral mobility: retirement age sensitivity
# ===========================================================================================================================
def SI5():
    fig, data_save = pf.bar_retirement_age(Result_data, T, provincesChina, provincesIndia)
    pf.print_destination_results_retirement(data_save,t1=2050)
    pf.save_figure(fig,'SI5_Mobility_retirement_age_sensitivity','svg')
    

#%% SI6 - Employment trajectories: productivity growth sensitivity
# ===========================================================================================================================
def SI6():
    Step = 5
    Show_uncertainty = False
    Show_alternatives = True
    Show_supply = False
    fig = pf.plot_national_employment_trajectories(T,Result_data,Historical_data,Step,Show_alternatives,Show_supply,Show_uncertainty)
    pf.save_figure(fig,'SI6_employment_productivity_sensitivity','svg',dpi=600)


#%% SI7 - Labour sectoral mobility: productivity growth sensitivity
# ===========================================================================================================================
def SI7():
    fig, data_save = pf.bar_productivity(Result_data, T, provincesChina, provincesIndia)
    pf.save_figure(fig,'SI7_Mobility_productivity_sensitivity','svg')

#%% SI9 - National employment trajectories: supply vs demand-driven sensitivity
# ===========================================================================================================================
def SI9():
    Step = 5
    Show_uncertainty = False
    Show_alternatives = False
    Show_supply = True
    fig = pf.plot_national_employment_trajectories(T,Result_data,Historical_data,Step,Show_alternatives,Show_supply,Show_uncertainty)
    pf.save_figure(fig,'SI9_employment_trajectories_demanddrivenscenarios','svg',dpi=600)

#%% SI10 - Subnational employment trajectories: supply vs demand-driven sensitivity
# ===========================================================================================================================
def SI10():
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
def SI11():
    fig = pf.boxplot_share_not_finding_demand(Result_data, T)
    pf.save_figure(fig,'SI4_Boxplot_presentation','jpg',dpi=700)

#%% PRESENTATIONS ============================================================================= 
#%% P1 - Scenarios descriptions
# ===========================================================================================================================
def P1():
    fig = pf.plot_scenario_description(Imaclim_data, T)
    pf.save_figure(fig,'P0_scenario','svg',dpi=600)

#%% MAIN ====================================================================================
if __name__ == "__main__":
    plot_main = True
    plot_extended = True
    plot_supplementary = True
    plot_presentation = False

    if plot_main:
        print('Plotting main figures')
        M1()
        M2()
        M3()
        M4()
    if plot_extended:
        print('Plotting extended data figures')
        ED1()
        ED2()
        ED3()
    if plot_supplementary:
        print('Plotting supplementary information figures')
        SI3()
        SI4()
        SI5()
        SI6()
        SI7()
        SI9()
        SI10()
        SI11()
    if plot_presentation:
        print('Plotting presentation figures')
        P1()

