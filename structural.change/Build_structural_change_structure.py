#%% Importing libraries
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import pandas as pd
import numpy as np
import os



matplotlib.rcParams['axes.unicode_minus'] = False
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

plt.rcParams['font.size'] = 9
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["legend.fancybox"] = False
matplotlib.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-white')
matplotlib.rcParams['font.family'] = 'Arial'

#%%
# Importing Imaclim results

T = range(2015, 2101)
T = np.array(T)

scenarios = list(np.array([[ x + y  for x in ['WO-NPi-ElecIndus','WO-NDCLTT-ElecIndus','WO-15C-ElecIndus']] for y in ['']]).flatten())

Imaclim_data = []
for scenario in scenarios:
    file_name ='../coal.labour.nexus/input/IMACLIM_waysout_outputs_' + scenario +'.csv'
    Scenario_data= pd.read_csv(file_name)
    Scenario_data['Scenario'] = scenario
    Imaclim_data.append(Scenario_data)

Imaclim_data = pd.concat(Imaclim_data, ignore_index=True)
Imaclim_data.iloc[:, 5:] = Imaclim_data.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')

#%%
# Importing the DOSE data
DOSE = pd.read_csv('../structural.change/data/2023 - Wenz et al - DOSE Global dataset of reported subnational economic output.csv')
Cregion = DOSE[(DOSE.country=="China") & (DOSE.year == 2016)]['region'].unique()
Iregion = DOSE[(DOSE.country=="India") & (DOSE.year == 2016)]['region'].unique()


#%%
year_base = 2016


#Plotting the DOSE data
DOSE = DOSE[DOSE['country'].isin(['China','India'])]
DOSE = DOSE.drop(DOSE[(DOSE['country'] == 'India') & (DOSE['year'] == 2015)].index)

DOSE['share_ag'] = DOSE.ag_grp_pc_usd_2015 / DOSE.grp_pc_usd_2015
DOSE['share_man'] = DOSE.man_grp_pc_usd_2015 / DOSE.grp_pc_usd_2015
DOSE['share_serv'] = DOSE.serv_grp_pc_usd_2015 / DOSE.grp_pc_usd_2015

DOSE['ln_share_agman'] = np.log(DOSE.share_ag / DOSE.share_man)
DOSE['ln_share_servman'] = np.log(DOSE.share_serv / DOSE.share_man)
DOSE['ln_gdp'] = np.log(DOSE.grp_pc_usd_2015)



#%% Aggregate DOSE data at the national level

DOSE['ag_grp_usd_2015'] = DOSE.ag_grp_pc_usd_2015 * DOSE['pop'] 
DOSE['man_grp_usd_2015'] = DOSE.man_grp_pc_usd_2015 * DOSE['pop']
DOSE['serv_grp_usd_2015'] = DOSE.serv_grp_pc_usd_2015 * DOSE['pop']
DOSE['grp_usd_2015'] = DOSE.grp_pc_usd_2015 * DOSE['pop']

NDOSE = DOSE.groupby(['country','year']).sum().reset_index().drop(columns= ['region','GID_0','GID_1'])

NDOSE['ag_grp_pc_usd_2015'] = NDOSE.ag_grp_usd_2015 / NDOSE['pop'] 
NDOSE['man_grp_pc_usd_2015'] = NDOSE.man_grp_usd_2015 / NDOSE['pop']
NDOSE['serv_grp_pc_usd_2015'] = NDOSE.serv_grp_usd_2015 / NDOSE['pop']
NDOSE['grp_pc_usd_2015'] = NDOSE.grp_usd_2015 / NDOSE['pop']


NDOSE['share_ag'] = NDOSE.ag_grp_pc_usd_2015 / NDOSE.grp_pc_usd_2015
NDOSE['share_man'] = NDOSE.man_grp_pc_usd_2015 / NDOSE.grp_pc_usd_2015
NDOSE['share_serv'] = NDOSE.serv_grp_pc_usd_2015 / NDOSE.grp_pc_usd_2015

NDOSE['ln_share_agman'] = np.log(NDOSE.share_ag / NDOSE.share_man)
NDOSE['ln_share_servman'] = np.log(NDOSE.share_serv / NDOSE.share_man)
NDOSE['ln_gdp'] = np.log(NDOSE.grp_pc_usd_2015)

#%%# Evaluating the economic and population growth from 2016:
evol={}
pevol={}
Colors = ['#F18872','#A7C682','#97CEE4']
fig, axs = plt.subplots(1,2,figsize=(10,4))
for ind_country, country in enumerate(['CHN','IND']):
    for ind_scenario, scenario in enumerate(['WO-NPi-ElecIndus', 'WO-NDCLTT-ElecIndus', 'WO-15C-ElecIndus']):
        ax = axs[ind_country]
        evol[(country,ind_scenario)] = Imaclim_data[(Imaclim_data.Region==country)&
                                                    (Imaclim_data.Variables=='GDP|PPP')&
                                                    (Imaclim_data.Scenario==scenario)].values[0][5:]/Imaclim_data[
                                                    (Imaclim_data.Region==country)&
                                                    (Imaclim_data.Variables=='Population')&
                                                    (Imaclim_data.Scenario==scenario)].values[0][5:]/(Imaclim_data[
                                                    (Imaclim_data.Region==country)&
                                                    (Imaclim_data.Variables=='GDP|PPP')&
                                                    (Imaclim_data.Scenario==scenario)].loc[:,'2015'].values[0]/Imaclim_data[
                                                    (Imaclim_data.Region==country)&
                                                    (Imaclim_data.Variables=='Population')&
                                                    (Imaclim_data.Scenario==scenario)].loc[:,'2015'].values[0])
        
        pevol[(country,ind_scenario)] = Imaclim_data[(Imaclim_data.Region==country)&
                                                     (Imaclim_data.Variables=='Population')&
                                                    (Imaclim_data.Scenario==scenario)].values[0][5:]/(Imaclim_data[
                                                    (Imaclim_data.Region==country)&
                                                    (Imaclim_data.Variables=='Population')&
                                                    (Imaclim_data.Scenario==scenario)].loc[:,'2015'].values[0])

        ax.plot(T,evol[(country,ind_scenario)],color= Colors[ind_scenario])
        ax.set_ylim([0,15])
        ax.set_ylabel(["GDP growth/population growth"])
        ax.set_title(["China","India"][ind_country])


#%%
# Calculating regional structure and absolute production values
coef_directory = 'results/'
Model_coefs = pd.read_csv(coef_directory+'model_coefficients.csv',index_col=0)
Model_coefs.loc['Model_AG_China',:].iloc[1:].values


coef_a_ss = [Model_coefs.loc['Model_AG_China',:].iloc[1:].values,Model_coefs.loc['Model_AG_India',:].iloc[1:].values]
coef_s_ss = [Model_coefs.loc['Model_Serv_China',:].iloc[1:].values,Model_coefs.loc['Model_Serv_India',:].iloc[1:].values]

country = "China"
cntry = 'CHN'
xm = {}
xa = {}
xs = {}

totm = {}
tota = {}
tots = {}
totg = {}

ysave={}

for ind_country, country in enumerate(['China','India']):
    cntry = ['CHN','IND'][ind_country]
    coef_a_s = coef_a_ss[ind_country]
    coef_s_s = coef_s_ss[ind_country]
    for ind_region, region in enumerate(DOSE[(DOSE.country==country) & (DOSE.year == year_base)]['region'].unique()):
        for ind_scenario, scenario in enumerate(['WO-NPi-ElecIndus', 'WO-NDCLTT-ElecIndus', 'WO-15C-ElecIndus']):
            # print(region)
            y = float(DOSE[(DOSE.region == region) & (DOSE.year == year_base)].ln_gdp.iloc[0])+np.array([np.log(x) for x in evol[(cntry,ind_scenario)]])
            dy  = [y[i+1]-y[i] for i in range(len(y)-1)]
            dy2 = [y[i+1]**2-y[i]**2 for i in range(len(y)-1)]
            dy3 = [y[i+1]**3-y[i]**3 for i in range(len(y)-1)]
            popu= float(DOSE[(DOSE.region == region) & (DOSE.year == year_base)]['pop'].iloc[0])

            ln_share_agman = [float(DOSE[(DOSE.region == region) & (DOSE.year == year_base)].ln_share_agman.iloc[0])]
            ln_share_servman = [float(DOSE[(DOSE.region == region) & (DOSE.year == year_base)].ln_share_servman.iloc[0])]

            for i in range(len(dy)):
                ln_share_agman.append(float(ln_share_agman[-1]+dy[i]*coef_a_s[0]+dy2[i]*coef_a_s[1]+dy3[i]*coef_a_s[2]))
                ln_share_servman.append(float(ln_share_servman[-1]+dy[i]*coef_s_s[0]+dy2[i]*coef_s_s[1]+dy3[i]*coef_s_s[2]))

            xm[region,ind_scenario] = 1 / (1 + np.exp(ln_share_agman) + np.exp(ln_share_servman))
            xa[region,ind_scenario] = np.exp(ln_share_agman) * xm[region,ind_scenario]
            xs[region,ind_scenario] = np.exp(ln_share_servman) * xm[region,ind_scenario]

            totm[region,ind_scenario] = xm[region,ind_scenario] * np.exp(y) * pevol[(cntry,ind_scenario)]*popu
            tota[region,ind_scenario] = xa[region,ind_scenario] * np.exp(y) * pevol[(cntry,ind_scenario)]*popu
            tots[region,ind_scenario] = xs[region,ind_scenario] * np.exp(y) * pevol[(cntry,ind_scenario)]*popu
            totg[region,ind_scenario] = np.exp(y) * pevol[(cntry,ind_scenario)]

            ysave[region,ind_scenario] = y

#%%
# Plotting regional production
fig, axs = plt.subplots(3,3,figsize=(10,6))

for ind_var, variable in enumerate(['agriculture','industry','services']):
    output = [xa,xm,xs][ind_var]
    for ind_scenario, scenario in enumerate(['WO-NPi-ElecIndus', 'WO-NDCLTT-ElecIndus', 'WO-15C-ElecIndus']):
        ax = axs[ind_var,ind_scenario]
        for ind_country, country in enumerate(['China','India']):
            regions = DOSE[(DOSE.country==country) & (DOSE.year == year_base)]['region'].unique()
            for ind_region, region in enumerate(regions):
                if region in ['Jharkhand', 'Bihar']:
                    ax.plot(T,output[region,ind_scenario],color='k')
                    ax.text(T[-1],output[region,ind_scenario][-1],region,horizontalalignment='right')
                else:
                    ax.plot(T,output[region,ind_scenario],color=['red','orange'][ind_country])
        ax.set_title(variable)


#%%
# Plotting regional structural change
fig, axs = plt.subplots(2,3,figsize=(10,6))
scenario = 'WO-NPi-ElecIndus'
variables = ['agriculture','industry','services']
for ind_var, variable in enumerate(variables):
    output = [xa,xm,xs][ind_var]
    for ind_country, country in enumerate(['China','India']):
        ax = axs[ind_country, ind_var]
        regions = DOSE[(DOSE.country==country) & (DOSE.year == year_base)]['region'].unique()
        for ind_region, region in enumerate(regions):
            if region in ['Jharkhand', 'Bihar']:
                ax.plot(ysave[region,ind_scenario],output[region,ind_scenario],color=['red','orange'][ind_country],linewidth=0.5)
                ax.text(ysave[region,ind_scenario][-1],output[region,ind_scenario][-1],region,horizontalalignment='left')
            else:
                ax.plot(ysave[region,ind_scenario],output[region,ind_scenario],color=['red','orange'][ind_country],linewidth=0.2)
        
        ax.scatter(NDOSE.loc[NDOSE.country==country,'ln_gdp'][0::5],NDOSE.loc[NDOSE.country==country,['share_ag','share_man','share_serv'][ind_var]][0::5],color='k',s=5)
        ax.set_ylim([0,1])
        ax.set_xlim([5,12])
    
[ax.set_title(variable) for ax,variable in zip(axs[0,:],variables)]

    


#%%
# Calculating contributions
contriba = {}
contribm = {}
contribs = {}
contribg = {}

for ind_country, country in enumerate(['China','India']):
    cntry = ['CHN','IND'][ind_country]
    regions = [Cregion,Iregion][ind_country]
    for ind_region, region in enumerate(regions):
        for ind_scenario, scenario in enumerate(['WO-NPi-ElecIndus', 'WO-NDCLTT-ElecIndus', 'WO-15C-ElecIndus']):
            contriba[region,ind_scenario] = tota[region,ind_scenario] / np.array([tota[x,ind_scenario] for x in regions]).sum(axis=0)
            contribm[region,ind_scenario] = totm[region,ind_scenario] / np.array([totm[x,ind_scenario] for x in regions]).sum(axis=0)
            contribs[region,ind_scenario] = tots[region,ind_scenario] / np.array([tots[x,ind_scenario] for x in regions]).sum(axis=0)
            contribg[region,ind_scenario] = totg[region,ind_scenario] / np.array([totg[x,ind_scenario] for x in regions]).sum(axis=0)

#%%
# Plotting evolution of regional contributions
import seaborn as sns
import random
fig, axs = plt.subplots(1,3,figsize=(10,6))

for ind_var, variable in enumerate(['agriculture','industry','services']):
    output = [contribm,contribm,contribs][ind_var]
    # for ind_scenario, scenario in enumerate(['WO-NPi-ElecIndus', 'WO-NDCLTT-ElecIndus', 'WO-15C-ElecIndus']):
    ind_scenario = 0
    scenario = 'WO-NPi-ElecIndus'
    scen_name = ['NPI','NDC','NZ'][ind_scenario]
    ax = axs[ind_var]
    for ind_country, country in enumerate(['China','India']):
        regions = DOSE[(DOSE.country==country) & (DOSE.year == year_base)]['region'].unique()

        for ind_region, region in enumerate([x for x in regions[((regions!='Shanxi')|(regions!='Jharkhand')).any()][0]]+['Shanxi','Jharkhand']):
            if region in ['Shanxi','Jharkhand']:
                code = 'CN-SX' if region == "Shanxi" else 'IN-JH'

                linewidth = 1.5
                ax.text(T[-1],output[region,ind_scenario][-1]/output[region,ind_scenario][0],code,ha='left',va= 'center',fontsize=8)
                ax.plot(T,output[region,ind_scenario]/output[region,ind_scenario][0],color='k',linewidth=1.25)
            else:
                linewidth = 0.75
                ax.plot(T,output[region,ind_scenario]/output[region,ind_scenario][0],color=[sns.color_palette('Reds',n_colors=20),sns.color_palette('Oranges',n_colors=20)][ind_country][random.randint([15,5][ind_country],[19,15][ind_country])],linewidth=0.75)
            # ax.text(T[0],output[region,ind_scenario][0],region)
    # if ind_var == 0:
    ax.set_title(variable.capitalize())

    if ind_var == 0:
        ax.set_ylabel('Evolution of the\n contribution [-]')

    ax.axhline(1, color='k',linewidth=0.5)
    ax.set_ylim([0.5,1.5])
    ax.set_xlim([2015,2117])


#%%
# Importing original downscaling matrix to copy structures
Econ_structure0 = pd.read_csv(os.path.join('..','coal.labour.nexus','data','Coal_labour','Downscaling','Econ_structure.csv'))

output_path = os.path.join('..','coal.labour.nexus','data','Coal_labour','Downscaling')

if not os.path.exists(output_path):
    os.makedirs(output_path) 


ind_ind = ['indus'+ str(x) for x in T]
ind_agr = ['agric'+ str(x) for x in T]
ind_ser = ['servi'+ str(x) for x in T]
ind_gdp = ['gdp'+ str(x) for x in T]

for ind_scenario, scenario in enumerate(['NPI','NDC','NZ']):

    Econ_structure1 = Econ_structure0.copy()
    Econ_structure1.loc[Econ_structure1["Subregion_name"]=="Chhattisgarh",'Subregion_name'] = 'Chattisgarh'
    Econ_structure1.loc[Econ_structure1["Subregion_name"]=="Andaman & Nicobar Islands",'Subregion_name'] = 'Andaman Nicobar'
    regions = Econ_structure1['Subregion_name'].unique()

    for region in regions:
        if (region not in Cregion) and (region not in Iregion):
            if region in ['China','India']:
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_ind]=1
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_agr]=1
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_ser]=1
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_gdp]=1
            else:
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_ind]=0
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_agr]=0
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_ser]=0
                Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_gdp]=0

        else:
            Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_ind]=contribm[region,ind_scenario]
            Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_agr]=contriba[region,ind_scenario]
            Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_ser]=contribs[region,ind_scenario]
            Econ_structure1.loc[Econ_structure1["Subregion_name"]==region,ind_gdp]=contribg[region,ind_scenario]
            
            
    Econ_structure1.loc[Econ_structure1["Subregion_name"]=='Chattisgarh','Subregion_name'] = "Chhattisgarh"
    Econ_structure1.loc[Econ_structure1["Subregion_name"]=='Andaman Nicobar','Subregion_name'] = "Andaman & Nicobar Islands"
    Econ_structure1.to_csv(os.path.join(output_path,'Econ_structural_change_'+scenario+'.csv'))

    if scenario == "NPI":
        Econ_structure1.to_csv(os.path.join(output_path,'Econ_structural_change.csv'))

# Saving headers
econ_struct_indices = pd.DataFrame([''] + list(Econ_structure1.columns)).transpose()
econ_struct_indices.to_csv(os.path.join(output_path, "Econ_structural_change_index.csv"),index=False,header=False)
# %%
