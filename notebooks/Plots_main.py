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
# Plotting functions
import plotting_functions as pf
# import importlib
# importlib.reload(pf)


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

# file_name = list(np.array([['../coal.labour.nexus/output/20241016_Coal_labour_results/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NDC','NPI','NZ']] for y in ['','_PG0','_R55','_gem','_EW','_min','_max','_H40','_H0','_H60','_H20_P0','_H20_P100']]).flatten())

# Result_data = []
# for file in file_name:
#     Result_data.append(pd.read_csv(file))#, dtype=str))

#======
#Load old data
# file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NDC','NPI','NZ']] for y in ['','_PG0','_R55','_gem','_EW']]).flatten())
file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']] for y in ['','_PG0','_R55','_min','_max']]).flatten())

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


#%%
# ===========================================================================================================================
# ===========================================================================================================================
#                                              Plots for the core of the article
# ===========================================================================================================================
# ===========================================================================================================================


#%% 1) Employment trajectories
# ===========================================================================================================================

Step = 5

Countries = ['China','India']
Regions = [
    Result_data[(Result_data['Region'] == ['CHN','IND'][x])&(~Result_data['Downscaled Region'].isin(['China', 'India']))]['Downscaled Region'].unique()
    for x in [0, 1]
]
Scen_type = ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']




Show_uncertainty = False
Show_alternatives = False


if Show_alternatives:
    Alt_type = [
    '','_PG0','_H0','_H40','_H60','_H20_P0','_H20_P100']
    Ralpha = [1, 0.75, 1,1,1,1,1] * 3
    Rlinestyle = ['-',':','--','--','--','--','--']*3
    Rlinewidth = [1, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]* 3
    Rmarker = ['', '', '^', 'o', 'x','s','D'] * 3
    Scen_list = [0,7,14,1,8,15]
else:
    Alt_type = [
    '','_PG0']
    Ralpha = [1,1] * 5
    Rlinestyle = ['-',':']*5
    Rlinewidth = [1,0.75]* 5
    Rmarker = ['',''] * 5
    Scen_list = [0,2,4,6,8,1,3,5,7,9]
    # Scen_list = [0,3,6,1,4,7,2,5,8]
Scenarios = [x + y for x in Scen_type for y in Alt_type]



fig, axs = plt.subplots(1, 2, figsize=pf.standard_figure_size())
for c_index in [0, 1]:
    country = Countries[c_index]

    Variable = "Employment|Coal|Downscaled"

    ax = axs[c_index]

    ax.axhline(y=0, color='k', linewidth=0.8)

    ax.scatter(Historical_data['Year'],Historical_data[country]/1e6,color='k',s=5)


    Scen_y = []
    S_maxy = []
    S_miny = []
    for j,scenario in enumerate(Scenarios):
        COAL_emp = Result_data[(Result_data['Downscaled Region'] == country)
                               & (Result_data['Variable'] == Variable) &
                               (Result_data['Scenario'] == Scenarios[j])]

        y = np.zeros(86)
        if scenario in Scen_type:
            miny = y
            maxy = y 
        for region in Regions[c_index]:
            y = y + np.array(Result_data[
                (Result_data['Downscaled Region'] == region)
                & (Result_data['Variable'] == Variable)
                &
                (Result_data['Scenario'] == Scenarios[j])].values[0][6:]) / 1e6
            
            if scenario in Scen_type:
                miny = miny + np.array(Result_data[
                    (Result_data['Downscaled Region'] == region)
                    & (Result_data['Variable'] == Variable)
                    &
                    (Result_data['Scenario'] == Scenarios[j]+'_min')].values[0][6:]) / 1e6
                maxy = maxy + np.array(Result_data[
                    (Result_data['Downscaled Region'] == region)
                    & (Result_data['Variable'] == Variable)
                    &
                    (Result_data['Scenario'] == Scenarios[j]+'_max')].values[0][6:]) / 1e6

        Scen_y.append(y)
        
        if scenario in Scen_type:
            S_maxy.append(maxy)
            S_miny.append(miny)

        if Scenarios[j] in  ['NPI','NDC','NZ']:
            if COAL_emp.values[0][6:][
                    T < 2070][-1] < COAL_emp.values[0][6:][5] / 2:
                
                ax.scatter(T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] /2][0],
                        -0.125,
                        color=Colors[scenario.split('_PG0')[0]],
                        marker='o',
                        s=20)
                
                if COAL_emp.values[0][6:][
                    T < 2070][-1] < COAL_emp.values[0][6:][5] *0.05:
                    ax.scatter(T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] *0.05][0],
                            -0.125,
                            color=Colors[scenario.split('_PG0')[0]],
                            marker='^',
                            s=20)
                    
                if scenario == 'NZ':
                    print(f'In {country}, coal employment must be halved by {T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] /2][0]} to be 1.5°C-aligned')
                if (country == 'China')&(scenario=='NZ'):
                    print(f'In {country}, coal employment must be reduced by 95% by {T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] *0.05][0]} to be consistent with 1.5°C')
                if (country == 'China')&(scenario=='NDC'):
                    print(f'In {country}, coal employment must be reduced by 95% by {T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] *0.05][0]} to be consistent with NDC-LTT')
                # if scenario == 'NPI':
                #     print(f'In {country}, under NPI coal employment must be phased out by {T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] *0.05][0]}')


    Scen_y = pd.DataFrame(Scen_y, index=Scenarios)
    S_maxy = pd.DataFrame(S_maxy, index=Scen_type)
    S_miny = pd.DataFrame(S_miny, index=Scen_type)

    for Scen_type_ind,scenario in enumerate(Scenarios):
        ax.plot(T[T < 2070][0:-1:Step],
                Scen_y.loc[scenario].values[T < 2070][0:-1:Step],
                color=Colors[scenario.split('_PG0')[0]],
                linestyle=Rlinestyle[Scen_type_ind],
                linewidth=Rlinewidth[Scen_type_ind],
                alpha=Ralpha[Scen_type_ind],
                marker= Rmarker[Scen_type_ind],
                markersize = 3,
                markevery=int(Step/5))
        
        
        if (scenario in Scen_type) & Show_uncertainty:
            ax.fill_between(T[T < 2070][0:-1:Step],
                            S_miny.loc[scenario].values[T < 2070][0:-1:Step],
                            S_maxy.loc[scenario].values[T < 2070][0:-1:Step],
                            color=Colors[scenario.split('_PG0')[0]],
                            alpha = 0.2,
                            zorder = 0)

    # Formatting axes
    ax.set_title(country)
    ax.set_ylabel('Million workers')
    ax.set_ylim([-0.25, 5])
    ax.axvline(x=2020, color='k', linestyle='--', linewidth=0.8)

alines = []

for ind, Scen_type_ind in enumerate(Scen_list):#[0,2,4,1,3,5]:
    scenario = Scenarios[Scen_type_ind]
    alines.append(axs[0].plot([], [],
                              color=Colors[scenario.split('_PG0')[0]],
                              label=['NPi','NDC-LTT','1.5°C','NDC-LTT-CCS','1.5°C-CCS','NPi no growth','NDC-LTT no growth','1.5°C no growth','NDC-LTT-CCS no growth','1.5°C-CCS no growth'][ind],
                              linestyle=Rlinestyle[Scen_type_ind],
                linewidth=Rlinewidth[Scen_type_ind],
                alpha=Ralpha[Scen_type_ind] )[0])

alines.append(axs[0].scatter([], [],
                        s= 5,
                        color='k',
                        label='Historical data'))
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



if Show_alternatives:

    alines.append(axs[0].plot([], [],
                            color='k',
                            label='H20')[0])
    alines.append(axs[0].plot([], [],
                            color='k', marker = '^',
                            label='H0')[0])
    alines.append(axs[0].plot([], [],
                            color='k', marker = 'o',
                            label='H40')[0])
    alines.append(axs[0].plot([], [],
                            color='k', marker = 'x',
                            label='H60')[0])
    alines.append(axs[0].plot([], [],
                            color='k', marker = 's',
                            label='H20_P0')[0])
    alines.append(axs[0].plot([], [],
                            color='k', marker = 'D',
                            label='H20_P100')[0])

if Show_uncertainty:
    if ~Show_alternatives:
        alines.append(axs[0].plot([], [],
                                color='k',
                                label='Central estimate')[0])
    alines.append(axs[0].fill_between([], [], [],
                            color='darkgrey',
                            alpha = 0.5,
                            label='Likely range'))

[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]


labels = [la.get_label() for la in alines]
handles = [label for label in alines]

fig.legend(handles=alines,
           labels=labels,
           loc='lower center',
           ncol=4+Show_alternatives+Show_uncertainty,
           bbox_to_anchor=(0.5, -0.18),
           frameon=False)



pf.save_figure(fig,'1_Employment','svg')

# %% 2) Mobility of laid-off coal workers between 2020-2030 and 2020-2050.
# ===========================================================================================================================
# BARCHART DESTINATION 1 Period

import importlib
importlib.reload(pf)

Provinces = [provincesChina, provincesIndia]
Countries = ['China', 'India']
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
    region = ['China', 'India'][c_index]
    
    for stype_index, (t0,T1) in enumerate(zip(T0s,T1s)):
        ax = axs[c_index][stype_index]


        if stype_index == 0:
            ax.set_ylabel(region+'\nMillion workers')
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
            if (stype_index == 2)&(t1!=2100):
                ax.text(X[s_index],data_save[(region,Scenario,t1)][:-1].sum(),t1+1,style='italic')


            x += 1


            if (Scenario == 'NPI')&(stype_index == 1):
                u = round(data_save[(region, Scenario,2030)][3]*1000)
                print(f'Under {Scenario}, in {region}, {u} thousand workers will not find new employment by 2030')


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

        ax.set_ylim([-0.5, 3.8])

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
# Leaving into retirement
scenario = 'NDC'
t1 = 2050
chn = ds.loc[('China',scenario,t1),'R']/sum(ds.loc[('China',scenario,t1),ds.columns!='H'])*100
ind = ds.loc[('India',scenario,t1),'R']/sum(ds.loc[('India',scenario,t1),ds.columns!='H'])*100
print(f'- In all cases, we find that a significant share of workers is able to leave into retirement rather than be laid off.\n By {t1} that is the case of around {chn:.1f}% of workers in China and {ind:.1f}% of workers in India the NDC-LTT.')

df_rrate = ds.loc[[(x,y,z) for x in ['China','India'] for y in ['NPI','NDC','NZ'] for z in [2030,2050]],:]
df_rrate['Rrate'] = df_rrate.R/(df_rrate.R+df_rrate.D+df_rrate.I+df_rrate.U)

t1 = 2030
df_rrate = ds.loc[[(x,y,z) for x in ['China','India'] for y in ['NPI','NDC','NZ'] for z in [2030,2050]],:]
df_rrate['Rrate'] = df_rrate.R/(df_rrate.R+df_rrate.D+df_rrate.I+df_rrate.U)
df_rrate2030NPINDC = df_rrate.loc[[(x,y,2030) for x in ['China','India'] for y in ['NPI','NDC']],'Rrate']

print(f'-In the NPI and NDC-LTT scenarios, we find that {100*min(df_rrate2030NPINDC):.1f}-{100*max(df_rrate2030NPINDC):.1f}% of redudant workers can leave into retirement rather than be laid off.')

df_rrate2030NZ = df_rrate.loc[[(x,'NZ',2030) for x in ['China','India']],'Rrate']

print(f'-In the NZ scenarios this goes down to {100*min(df_rrate2030NZ):.1f}-{100*max(df_rrate2030NZ):.1f}%')


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


# reduction = 100*(1-ds.loc[('China','NPI',2050),'U']/ds.loc[('China','NZ',2032),'U'])
# print(f'-{reduction:.1f}% less Chinese worker go into unemployment by the time 80% coal jobs have been destroyed in NPI compared to 1.5°C')



# Share of lay-offs not finding new employment
scenario = 'NPI'
t1 = 2050
chn0 = 100-(ds.loc[('China',scenario,t1),'D'])/sum(ds.loc[('China',scenario,t1),['D','I','U']])*100
ind0 = 100-(ds.loc[('India',scenario,t1),'D'])/sum(ds.loc[('India',scenario,t1),['D','I','U']])*100
scenario = 'NZ'
chn1 = 100-(ds.loc[('China',scenario,t1),'D'])/sum(ds.loc[('China',scenario,t1),['D','I','U']])*100
ind1 = 100-(ds.loc[('India',scenario,t1),'D'])/sum(ds.loc[('India',scenario,t1),['D','I','U']])*100

print(f'- In the long run, the 1.5°C scenario leads to a significant share of laid-off workers not finding employment by 2050, \n with {chn1:.1f}% of Chinese workers and {ind1:.1f}% of Indian workers not finding new employment \n against {chn0:.1f}% and {ind0:.1f}% respectively in the NPI scenario.')




# %% 3) Exposure and vulnerability of regions to coal transition betweeen 2020 and 2050
# ===========================================================================================================================
import importlib
importlib.reload(pf)

t = 2015
CHN_share_LF=100*float(Result_data[(Result_data.Scenario=='NPI')&(Result_data['Downscaled Region']=='China')&(Result_data.Variable=='Employment|Coal|Downscaled')][str(t)].values[0])/float(Result_data[(Result_data.Scenario=='NPI')&(Result_data['Downscaled Region']=='China')&(Result_data.Variable=='Labour Force|Downscaled')][str(t)].values[0])
IND_share_LF=100*float(Result_data[(Result_data.Scenario=='NPI')&(Result_data['Downscaled Region']=='India')&(Result_data.Variable=='Employment|Coal|Downscaled')][str(t)].values[0])/float(Result_data[(Result_data.Scenario=='NPI')&(Result_data['Downscaled Region']=='India')&(Result_data.Variable=='Labour Force|Downscaled')][str(t)].values[0])

print(f"Indian regions are on average less exposed to the transition as {IND_share_LF:.1f}%  of labour force works in coal against {CHN_share_LF:.1f}% in China).")


Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China','India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)


# Defining the scenarios
Scenarios =  ['NPI','NDC','NZ']
Scenarios_names = ['NPI','NDC','1.5°C']

t0 = 2019
t1 = 2050

# Creating the figure
fig1, axs1 = plt.subplots(2,3, figsize=(20/2.54,13/2.54))
axs = [axs1]
axs = np.array(axs).flatten()


# Initiating the data lists
Data_Chn=[]
Data_Ind=[]
key_data = {}
    

for ind_t, t1 in enumerate(['80%',2040]):
    for s_index, scenario in enumerate(Scenarios):
        ax = axs1[ind_t][s_index]
        Asia_Data, key_data, cmap_zip = pf.plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names)

        if (t1 == '80%')&(s_index==2):
            region='China'
            print(f"Where the national average vulnerability score moves from {(100*key_data[region+'0']['coordinates'][0]):0f}% to {(100*key_data[region+'1']['coordinates'][0]):0f}% in China")
            region='India'
            print(f"it only moves from {(100*key_data[region+'0']['coordinates'][0]):0f}% to {(100*key_data[region+'1']['coordinates'][0]):0f}% in India")
        

cmap, b_xlim, b_ylim, n = cmap_zip

# ====================================
# Legend

# Creating legend axis
cax = fig1.add_axes([0.92, 0.3, 0.4, 0.4])
cax = xycmap.bivariate_legend(ax=cax, sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim) #xlims=(0,0.5),ylims=(0,0.06)
cax.set_xlabel('Share of laid-off workers \n going into unemployment',fontsize=11)
cax.set_ylabel('Decrease in relative coal jobs',fontsize=11)



# If the number of color box is more than 5, not all ticks are shown
if min(n)>5:    
    cax.set_xticks([pf.interpol(x,b_xlim,cax.get_xlim()) for x in [0.5,0.7]])
    cax.set_yticks([pf.interpol(np.log(x),b_ylim,cax.get_ylim()) for x in [1e-3,5e-3,0.01,0.05]])
    cax.set_xticklabels(["50%","70%"],fontsize=11)
    cax.set_yticklabels(["0.1%","0.5%","1%","5%"],fontsize=11)

Evol_regions = ['China', 'Shanxi', 'Henan', 'India', 'Jharkhand']#, 'Odisha']

# Plotting evolution of results across main scenarios for the main regions
for region in Evol_regions:#['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [0,1,2]:
        if key_data[region+str(s_index)]['destruction']>0:
            xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
            ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
        else:
            xs.append(np.nan)
            ys.append(np.nan)

    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)



# Scattering the results for the main regions
for region in Evol_regions:# ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [0,1,2]:
        
        if key_data[region+str(s_index)]['destruction']>0:
            xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
            ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
            cax.scatter(xs[-1],ys[-1],color=Colors[key_data[region+str(s_index)]['Scenario'].split('_')[0]],marker='o',s=0.4*key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)

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
labels = ['NPi','NDC-LTT', '1.5°C','Job losses \n [people]','200k','400k','600k']
cax.legend(handles=alines,
           labels=labels,
           loc='center right',
           bbox_to_anchor=(1.53, 0.5),
           frameon=False,
           fontsize=8)

fig1.subplots_adjust(wspace=0.05,hspace=-0.09)

# Regions = ['Shanxi','Jharkhand']
# for region in Regions:
#     prod = Result_data[(Result_data["Downscaled Region"] == region) & (Result_data["Variable"] == "Resource|Extraction|Coal|Downscaled") & (Result_data["Scenario"] == "NPI_gem")]['2021'].values[0]
#     emp = Result_data[(Result_data["Downscaled Region"] == region) & (Result_data["Variable"] == "Employment|Coal|Downscaled") & (Result_data["Scenario"] == "NPI_gem")]['2021'].values[0]
 
#     print(f'Production in {region} in 2021 is {prod} EJ')     
#     print(f'Employment in {region} in 2021 is {round(emp)} people')

[ax.text(0.02,0.95, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(list(axs)+[cax],['a)','b)','c)','d)','e)','f)','g)'])];



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
    Asia_Data, key_data, cmap_zip = pf.plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names)
    

cmap, b_xlim, b_ylim, n = cmap_zip

cax = fig1.add_axes([0.85, 0.3+0.075, 0.25, 0.25])
cax = xycmap.bivariate_legend(ax=cax, sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim) #xlims=(0,0.5),ylims=(0,0.06)
cax.set_xlabel('Share of workers not \n finding new employment',fontsize=11)
cax.set_ylabel('Decrease in relative coal jobs',fontsize=11)
if min(n)>5:    
    cax.set_xticks([pf.interpol(x,b_xlim,cax.get_xlim()) for x in [0.5,0.7,0.9]])
    cax.set_yticks([pf.interpol(np.log(x),b_ylim,cax.get_ylim()) for x in [1e-3,5e-3,0.01,0.05]])
    cax.set_xticklabels(["50%","70%","90%"],fontsize=11)
    cax.set_yticklabels(["0.1%","0.5%","1%","5%"],fontsize=11)
    
# for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
#     xs = []
#     ys = []
#     for s_index in [0,1,2]:
#         xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
#         ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
#     cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)
# for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
#     xs = []
#     ys = []
#     for s_index in [3,4,5]:
#         xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
#         ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
#     cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)

# for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
#     xs = []
#     ys = []
#     for s_index in [0,1,2,3,4,5]:
#         xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
#         ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
#         cax.scatter(xs[-1],ys[-1],color=Colors[key_data[region+str(s_index)]['Scenario']],marker='o',s=key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)
#     cax.annotate(region,(xs[0],ys[0]-0.1),ha='center',va='top',fontsize=8)



# Defining a legend mannualy 
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
labels = ['NPi','NDC-LTT', '1.5°C','NPi GEM','NDC GEM','1.5°C GEM','Job losses \n [people]','200k','400k','600k']
cax.legend(handles=alines,
           labels=labels,
           loc='center right',
           bbox_to_anchor=(1.5, 0.5),
           frameon=False,
           fontsize=8)

fig1.subplots_adjust(wspace=-0.45,hspace=0.2)

#%% 1.1b) Calibration source
# ===========================================================================================================================

Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China','India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)

Scenarios =  ['NPI','NDC','NZ','NPI_gem','NDC_gem','NZ_gem']

Scenarios = ['NPI_gem','NPI_min','NPI','NPI_max',
             'NDC_gem','NDC_min','NDC','NDC_max',
             'NZ_gem','NZ_min','NZ','NZ_max']


Scenarios_names = ['NPI','NDC','1.5°C','NPI GEM','NDC GEM','1.5 GEM']

Scenarios_names = Scenarios

fig1, axs1 = plt.subplots(3,4, figsize=(18/1.6, 15.3/1.6))

axs = [axs1]

axs = np.array(axs).flatten()
t0 = 2019
t1 = 2050
Data_Chn=[]
Data_Ind=[]

key_data = {}
s0 = 0
for s_index, scenario in enumerate(Scenarios):# [0,3,1,4,2,5]:
    ax = axs[s_index]
    # s0+=1
    # scenario = Scenarios[s_index]
    Asia_Data, key_data, cmap_zip = pf.plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names)
    

cmap, b_xlim, b_ylim, n = cmap_zip

cax = fig1.add_axes([0.85, 0.3+0.075, 0.25, 0.25])
cax = xycmap.bivariate_legend(ax=cax, sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim) #xlims=(0,0.5),ylims=(0,0.06)
cax.set_xlabel('Share of workers not \n finding new employment',fontsize=11)
cax.set_ylabel('Decrease in relative coal jobs',fontsize=11)
if min(n)>5:    
    cax.set_xticks([pf.interpol(x,b_xlim,cax.get_xlim()) for x in [0.5,0.7,0.9]])
    cax.set_yticks([pf.interpol(np.log(x),b_ylim,cax.get_ylim()) for x in [1e-3,5e-3,0.01,0.05]])
    cax.set_xticklabels(["50%","70%","90%"],fontsize=11)
    cax.set_yticklabels(["0.1%","0.5%","1%","5%"],fontsize=11)
    
for region in ['China','India','Shanxi','Jharkhand']:
    xs = []
    ys = []
    for s_index in [0,1,2]:
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)
for region in ['China','India','Shanxi','Jharkhand']:
    xs = []
    ys = []
    for s_index in [3,4,5]:
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)

for region in ['China','India','Shanxi','Jharkhand']:
    xs = []
    ys = []
    for s_index in [0,1,2,3,4,5]:
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
        cax.scatter(xs[-1],ys[-1],color=Colors[key_data[region+str(s_index).split('_')[0]]['Scenario']],marker='o',s=key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)
    cax.annotate(region,(xs[0],ys[0]-0.1),ha='center',va='top',fontsize=8)



# Defining a legend mannualy 
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
labels = ['NPi','NDC-LTT', '1.5°C','NPi GEM','NDC GEM','1.5°C GEM','Job losses \n [people]','200k','400k','600k']
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
                        figsize=(20/2.54,15/2.54))


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

cax.set_ylabel('Unemployment rate [%]')


# Line graph of unemployment in China and India in all three scenarios
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

        axn.plot(T[T < 2070][0:-1:Step],u[T < 2070][0:-1:Step],label=country,color=Colors[scenario],linestyle=['-','--'][c_i])

axn.set_ylabel('Unemployment rate [%]')
axn.set_ylim([0,14])

[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 12, fontweight='bold', va='top', ha='left') for ax, label in zip([axs[0,0],axn],['a)','b)'])]

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

Xs = [
    [x for x in range(len(Scenarioss[0]))],
    [x for x in range(len(Scenarioss[1]))],
    [x for x in range(len(Scenarioss[2]))]
    ]

data_save = {}
fig, axs = plt.subplots(2,
                        3,
                        figsize=pf.standard_figure_size(),
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
        for s_index, Scenario in enumerate(Scenarios):
            data_save, alines = pf.destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
            x += 1

        alines = [x[0] for x in alines]
        labs = [lab.get_label() for lab in alines]
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
        ax.set_ylim([-1, 3.8])

[ax.set_yticklabels([]) for ax in axs[:,[1,2]].flatten()]
[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)','e)','f)'])]


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
Scenarios =['WO-NPi-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS0', 'WO-15C-ElecIndus-CCS0']
Outputs_name = ['NPi','NDC-LTT','1.5°C']

Variabless = [
    'Import|Coal',
    'Final Energy|Industry|Solids|Fossil',
    'Secondary Energy|Electricity|Coal',
    'Power Plants|Coal',
    'Refineries|Coal',
    'Final Energy|Commercial|Solids', 
    'Final Energy|Residential|Solids|Fossil',
    'Export|Coal'
    ]

Variabless_names = ['Import','Industry','Electricity','Power plants losses','Refineries','Commercial','Residential','Export']


Variable2 = ['Primary Energy|Coal','Output|Coal']
Variable2_name = ['Consumption','Production']


Colorss = [sns.color_palette()[x] for x in [0,7,1]]+[sns.color_palette("pastel")[1]]+[sns.color_palette()[x] for x in [2,3,4]]+[sns.color_palette("pastel")[0]]

Regions = ['World', 'CHN', 'IND','USA','AFR','EUR']

fig, axs = plt.subplots(len(Regions), len(Scenarios), figsize=pf.standard_figure_size())

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

        for ind_var, variable in enumerate(Variables):

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
        cumulated_data = pf.get_cumulated_array(data, min=0)
        cumulated_data_neg = pf.get_cumulated_array(data, max=0)

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
        for ind_var, variable in enumerate(Variable2):

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

chn_name = 'Countries of centrally-planned Asia; primarily China'
ind_name = 'Countries of South Asia; primarily India'
regions_ar6 = ['World', chn_name, ind_name]

variables_ar6 = ['Primary Energy|Coal','Trade|Primary Energy|Coal|Volume',
             'Final Energy|Industry|Solids|Coal','Emissions|CO2|Energy and Industrial Processes',
             'Carbon Sequestration|CCS|Biomass','Secondary Energy|Electricity|Coal',
             'Primary Energy|Oil','Primary Energy|Gas','Unemployment|Rate','Employment|Industry|Mining',
             "Investment|Energy Supply|Extraction|Coal"]
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
var = 'Output|Coal'
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)

#%% 3.3) Electricity from coal
var = 'Secondary Energy|Electricity|Coal'
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)

# pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)


#%% 3.4) Investment in coal supply
var = "Investment|Energy Supply|Extraction|Coal"
pf.plotting_with_AR6_range(var,var,regions_ar6,categories,df,cols,Imaclim_data,T,True)

#%% 3.4) Investment in coal supply
var = 'Carbon Sequestration|CCS|Biomass'
var_im = 'Carbon Capture|Storage|Biomass'
regions = regions_ar6[1:3]
categories = categories[0:2]
pf.plotting_with_AR6_range(var,var_im,regions,categories,df,cols,Imaclim_data,T,False)


#=========================================================================================================================================================================
#=========================================================================================================================================================================
#%%=======================================================================================================================================================================
# Univariate maps

import importlib
importlib.reload(pf)

Scenarios = ['NPI','NDC','NDC_CCS1','NZ','NZ_CCS1']
Scenarios_names = ['NPi','NDC-LTT','NDC-LTT w/CCS','1.5°C','1.5°C w/CCS']



# 1) 
#  Mapping Exposition
Ts = ['80%',2040]
fig, axs = plt.subplots(len(Ts),len(Scenarios),
                        figsize=(20/2.54,15/2.54))


Data_Chn = []
Data_Ind = []
var = 'share_destruction'

cax = axs[0,4].inset_axes([1.05, 0, 0.1, 1])
t0=2020
colormap='BrBG'
zlim= [pf.pseudo_log(-5e-2)
       ,pf.pseudo_log(5e-2)]
colormap='Reds'
zlim = [0,pf.pseudo_log(5e-2)]
for t_index, T1 in enumerate(Ts):
    for s_index, scenario in enumerate(Scenarios):
        ax = axs[t_index,s_index]
        ax = pf.monovariate_map(var,t0,T1,ax,cax,zlim,colormap,Result_data,scenario,Asia)

[ax.set_title(Scenarios_names[s_index]) for ax,s_index in zip(axs[0,:],range(5))]
[ax.set_ylabel(T1) for ax,T1 in zip(axs[:,0],Ts)]

fig.subplots_adjust(hspace=-0.4)

#%%
# 2) 
#  Mapping Vulnerability
fig, axs = plt.subplots(len(Ts),len(Scenarios),
                        figsize=(20/2.54,15/2.54))

Data_Chn = []
Data_Ind = []
var = 'Finding_new_emp'

cax = axs[0,2].inset_axes([1.05, 0, 0.1, 1])
t0=2020
colormap='Reds'
zlim = [0,1]
zlim = [0.3,0.88]
for t_index, T1 in enumerate(Ts):
    for s_index, scenario in enumerate(Scenarios):
        ax = axs[t_index,s_index]
        ax = pf.monovariate_map(var,t0,T1,ax,cax,zlim,colormap,Result_data,scenario,Asia)

[ax.set_title(Scenarios_names[s_index]) for ax,s_index in zip(axs[0,:],range(3))]
[ax.set_ylabel(T1) for ax,T1 in zip(axs[:,0],Ts)]

fig.subplots_adjust(hspace=-0.4)



# %%
# 3) Mapping exposisition



import importlib
importlib.reload(pf)

Scenarios = ['NDC_CCS1','NZ_CCS1','NPI','NDC','NZ']
Scenarios_names = ['NDC-LTT w/CCS','1.5°C w/CCS','NPi','NDC-LTT','1.5°C']

T1 = 2035
fig, axs = plt.subplots(2,3,
                        figsize=(20/2.54,15/2.54))


Data_Chn = []
Data_Ind = []
var = 'share_destruction'

cax2 = axs[0,2].inset_axes([1.15, -1.3, 0.1, 2.3])


t0=2020
colormap='BrBG'
zlim= [pf.pseudo_log(-5e-2)
       ,pf.pseudo_log(5e-2)]
colormap='Reds'
zlim = [0,pf.pseudo_log(5e-2)]

from matplotlib.colors import TwoSlopeNorm
colormap = plt.cm.PiYG_r

# Define the range and midpoint
vmin = -0.005
vmax = 0.05  
vcenter = 0.0 # Midpoint at 0
stretch = 0.1

# Create a normalization object with a custom center
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
# norm = PseudologNorm(vmin=vmin, vcenter=vcenter, vmax=vmax, stretch=stretch)



ax = axs[0,0]
ax.set_title('2015 Workforce')
ax = pf.monovariate_map('Workforce',t0,T1,ax,cax,zlim,colormap,Result_data,scenario,Asia,norm)
for s_index, scenario in enumerate(Scenarios):
    ax = axs.flatten()[1+s_index]
    ax = pf.monovariate_map(var,t0,T1,ax,cax2,zlim,colormap,Result_data,scenario,Asia,norm)

[ax.set_title(Scenarios_names[s_index]) for ax,s_index in zip(axs.flatten()[1:],range(5))]

# cax.set_axis_off()
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax2, orientation='vertical')
cax2.set_yscale('linear')
cax2.set_yticks([-0.004,0,0.02,0.04])
cax2.set_yticklabels(['-0.4%','0%','2%','4%'])
fig.subplots_adjust(hspace=-0.4)

