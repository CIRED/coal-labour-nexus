#%%
#Importing libraries
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import xycmap
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import PowerNorm
import sys
import os
import mpltern
import sys
sys.path.append("..")
import notebooks.plotting_functions as pf

# Get current working directory
current_dir = os.getcwd()

# Assuming your script runs from the 'structural.change' directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

# Now you can import the module
from notebooks import plotting_functions as pf



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

Colors = pf.defining_waysout_colour_scheme()

T = range(2015, 2101)
T = np.array(T)

scenarios = list(np.array([[ x + y  for x in ['WO-NPi-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0','WO-15C-ElecIndus-CCS0']] for y in ['']]).flatten())

Imaclim_data = []
for scenario in scenarios:
    file_name ='../coal.labour.nexus/input/IMACLIM_waysout_outputs_' + scenario +'.csv'
    Scenario_data= pd.read_csv(file_name)
    Scenario_data['Scenario'] = scenario
    Imaclim_data.append(Scenario_data)

Imaclim_data = pd.concat(Imaclim_data, ignore_index=True)
Imaclim_data.iloc[:, 5:] = Imaclim_data.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')

#%%
# Map shapefile from naturalearthdata.com
Asia = gpd.read_file('../notebooks/shapefiles/Asia.shp')
Asia.loc[Asia.Region_Nam=="Orissa","Region_Nam"] = "Odisha"

#%%
#Importing module results
T = range(2015, 2101)
T = np.array(T)

input_directory = os.path.join("..","coal.labour.nexus","output")


file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']] for y in ['','_tra','_PG0','_R55','_min','_max','_C0I0_P0','_C0I0_P100','_C20I5_P0','_C20I5_P100','_C40I10_P100','_pop','_dose','_dpop','_sc']]).flatten())

Result_data = []
for file in file_name:
    Result_data.append(pd.read_csv(file))#, dtype=str))

Result_data = pd.concat(Result_data, ignore_index=True)

Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].applymap(lambda x: str(x).replace('D', 'E'))
Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].apply(pd.to_numeric, errors='coerce')
Region_indexes = pd.read_csv('../coal.labour.nexus/data/Coal_labour/Downscaling/Indexes.csv')


#%%
# Importing the DOSE data
DOSE = pd.read_csv('../structural.change/data/2023 - Wenz et al - DOSE Global dataset of reported subnational economic output.csv')
Cregion = DOSE[(DOSE.country=="China") & (DOSE.year == 2016)]['region'].unique()
Iregion = DOSE[(DOSE.country=="India") & (DOSE.year == 2016)]['region'].unique()

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

#%%
#Libraries and functions necessary for ternary map


def define_ternary_color_space(fig,subdivisions):
    ax = fig.add_axes([0.1, 0.4, 0.2, 0.2], projection="ternary", ternary_sum=1.0)

    ax.set_tlabel("Agriculture (%)")
    ax.set_llabel("Industry (%)")
    ax.set_rlabel("Services (%)")

    ag_lim=[0, 0.6]
    ind_lim=[0, 0.7]
    ser_lim=[0.2, 1]

    ag_lim=[0.0, 1]
    ind_lim=[0, 1]
    ser_lim=[0.0, 1]

    # Define the color space
    cs = [[0,209,208],[207,176,0],[255,128,247]]
    col_bin = pd.DataFrame(columns=['color','ag','ind','serv'])
    for i in range(0,subdivisions):
        for j in range(0,subdivisions+1-i):
            for k in range(max(0,subdivisions -2- i - j),min(subdivisions - i - j,subdivisions+1)):
                new_row = {}
                if sum([i,j,k]) == subdivisions-1:
                    a = [(i+1)/subdivisions, i/subdivisions, i/subdivisions]
                    b = [j/subdivisions, j/subdivisions, (j+1)/subdivisions]
                    c = [k/subdivisions, (k+1)/subdivisions, k/subdivisions]
                else:
                    a = [(i+1)/subdivisions, i/subdivisions, (i+1)/subdivisions]
                    b = [j/subdivisions, (j+1)/subdivisions, (j+1)/subdivisions]
                    c = [(k+1)/subdivisions, (k+1)/subdivisions, k/subdivisions]
                ct = [np.mean(a),np.mean(b),np.mean(c)]
                # Define the transformed coordinates of the corners
                corner1 = np.array([1+(1-ag_lim[1]), 0, -(1-ag_lim[1])])
                corner2 = np.array([0, 1+(1-ind_lim[1]), -(1-ind_lim[1])])
                corner3 = np.array([0, 0, 1])
                # Calculate the transformed coordinates of ct
                transformed_ct = ct[0] * corner1 + ct[1] * corner2 + ct[2] * corner3
                # Calculate the color based on the transformed coordinates
                color = tuple([min(max(x,0),1) for x in (transformed_ct[0] * np.array(cs[0]) +
                            transformed_ct[1] * np.array(cs[1]) +
                            transformed_ct[2] * np.array(cs[2])) / 255])
                new_row['color'] = [color]
                new_row['ag'] = max(a)
                new_row['ind'] = max(b)
                new_row['serv'] = max(c)
                col_bin = pd.concat([col_bin,pd.DataFrame(new_row)], ignore_index=True)
                ax.fill(a,b,c,
                        color=color, alpha=1)
    ax.set_tlim(ag_lim[0], ag_lim[1])
    ax.set_llim(ind_lim[0], ind_lim[1])
    ax.set_rlim(ser_lim[0], ser_lim[1])

    return ax, col_bin


def get_colors(df,t,scenario,variables,Regions,col_bin):
    colors = []

    for region in Regions:

        filtered_df = df[(df.Scenario==scenario)&(df['Downscaled Region']==region)]
        agri = float(filtered_df.loc[filtered_df.Variable==variables['agri'],str(t)].values[0])
        indus = float(filtered_df.loc[filtered_df.Variable==variables['indus'],str(t)].values[0])
        serv = float(filtered_df.loc[filtered_df.Variable==variables['serv'],str(t)].values[0])


        color = col_bin_f(agri,indus,serv,col_bin, subdivisions)
        colors.append(color)



    Colors = pd.DataFrame(data=np.array([list(Regions), colors],dtype=object).transpose(),
                        columns=['Region_Nam', 'colors'])
    return Colors


def ternary_map(ax,df,t,scenario,variables,Regions,Asia,col_bin):

    Colors = get_colors(df,t,scenario,variables,Regions,col_bin)

    Asia_Data_with_colors = Asia[Asia['CNTRY_Name'].isin(['China','India'])].merge(Colors, on='Region_Nam', how='left').fillna('lightgrey')
    Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
        edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)
    
    Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color='whitesmoke', edgecolor='black', linestyle='--',linewidth=0.5)

    return ax


def col_bin_f(agri,indus,serv,col_bin, subdivisions):
    agri = np.ceil(agri*subdivisions)/subdivisions
    indus = np.ceil(indus*subdivisions)/subdivisions
    serv = np.ceil(serv*subdivisions)/subdivisions
    return col_bin[(col_bin['ag']==agri)&(col_bin['ind']==indus)&(col_bin['serv']==serv)]['color'].values[0]

#%%
#Mapping structural change in all scenarios
Regions = Result_data[~Result_data['Downscaled Region'].isin(['China','India','Andaman & Nicobar Islands','Puducherry'])]['Downscaled Region'].unique()
scenario =  'NPI'

fig = plt.figure(figsize=(25, 20))

ts = [2015,2040,2060]

subdivisions = 20
ax, col_bin = define_ternary_color_space(fig,subdivisions)


# Plot evolution of key regions on the ternary plot
for region in ['Shanxi','Jharkhand']:
    alines = []
    for ind_scenario, scenario in enumerate(['NPI','NPI_pop','NPI_dose','NPI_sc']):

        agri =  [float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Agriculture|Share|Downscaled")&(Result_data.Scenario==scenario)].values[0][6:-1]]
        indus = [float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Industry|Share|Downscaled")&(Result_data.Scenario==scenario)].values[0][6:-1]]
        serv = [float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Services|Share|Downscaled")&(Result_data.Scenario==scenario)].values[0][6:-1]]
        
        
        alines.append(ax.plot(agri, indus, serv,color=['k','k','grey','grey'][ind_scenario],
                linestyle = ['-',':','-','--'][ind_scenario], label = scenario))
        ax.scatter([agri[x] for x in [5,25,45]], 
                    [indus[x] for x in [5,25,45]],
                    [serv[x] for x in [5,25,45]],
                marker='.',
                color=['k','k','grey','grey'][ind_scenario])
    ax.text(agri[0]+0.01, indus[0]-0.005, serv[0]-0.005,region,fontsize=8)

labels = [scenario for scenario in ['NPI','NPI_pop','NPI_dose','NPI_sc']]
handles = [aline[0] for aline in alines]
ax.legend(handles, labels, loc='lower center', fontsize=12,
           bbox_to_anchor=(0.4, -0.4),
           frameon=False,
           )

Va_Variables = {
    "agri" :"Employment|Agriculture|Share|Downscaled",
    'indus' : "Employment|Industry|Share|Downscaled",
    "serv" : "Employment|Services|Share|Downscaled"
}

# Plot data    
t = 2015
ax = fig.add_axes([0.1,1-0.3, 0.25, 0.25])
ax = ternary_map(ax, Result_data, t, scenario, Va_Variables, Regions, Asia,col_bin)


ax.set_title(t)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([65,140])
ax.set_ylim([7,55])



for ind_scenario, scenario in enumerate(['NPI','NPI_pop','NPI_dose','NPI_sc','NPI_dpop']):
    for t_index in range(len(ts)):
        ax = fig.add_axes([0.4+0.255*t_index, 0.7-0.25*ind_scenario, 0.25, 0.25])
        t = ts[t_index]
        ax = ternary_map(ax, Result_data, t, scenario, Va_Variables, Regions, Asia,col_bin)

        if ind_scenario == 0:
            ax.set_title(t, fontsize=14)
        if t_index == 0:
            ax.set_ylabel(scenario, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([65,140])
        ax.set_ylim([7,55])




#%%
#(Heat)mapping structural change in all scenarios
Regions = Result_data[~Result_data['Downscaled Region'].isin(['China','India','Andaman & Nicobar Islands','Puducherry'])]['Downscaled Region'].unique()
scenario =  'NPI'

fig = plt.figure(figsize=(13, 11))

ts = [2050]

subdivisions = 20
ax, col_bin = define_ternary_color_space(fig,subdivisions)
ax.text(0.02,0.96, 'b)', transform=ax.transAxes, fontsize= 14, fontweight='bold', va='top', ha='left')



# Plot evolution of key regions on the ternary plot
for region in ['Shanxi','Jharkhand']:
    alines = []
    for ind_scenario, scenario in enumerate(['NPI','NPI_pop','NPI_dose','NPI_sc']):

        agri =  [float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Agriculture|Share|Downscaled")&(Result_data.Scenario==scenario)].values[0][6:-1]]
        indus = [float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Industry|Share|Downscaled")&(Result_data.Scenario==scenario)].values[0][6:-1]]
        serv = [float(x) for x in Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Services|Share|Downscaled")&(Result_data.Scenario==scenario)].values[0][6:-1]]
        
        
        alines.append(ax.plot(agri, indus, serv,color=['k','k','grey','grey'][ind_scenario],
                linestyle = ['-',':','-','--'][ind_scenario], label = scenario))
        ax.scatter([agri[x] for x in [5,25,45]], 
                    [indus[x] for x in [5,25,45]],
                    [serv[x] for x in [5,25,45]],
                marker='.',
                color=['k','k','grey','grey'][ind_scenario])
    ax.text(agri[0]+0.01, indus[0]-0.005, serv[0]-0.005,region,fontsize=8)

labels = [scenario for scenario in ['NPI','NPI_pop','NPI_dose','NPI_sc']]
handles = [aline[0] for aline in alines]
ax.legend(handles, labels, loc='lower center', fontsize=12,
           bbox_to_anchor=(0.4, -0.4),
           frameon=False, ncols=2
           )

Va_Variables = {
    "agri" :"Employment|Agriculture|Share|Downscaled",
    'indus' : "Employment|Industry|Share|Downscaled",
    "serv" : "Employment|Services|Share|Downscaled"
}

# Plot data    
t = 2015
ax = fig.add_axes([0.1,1-0.3, 0.25, 0.25])
ax = ternary_map(ax, Result_data, t, scenario, Va_Variables, Regions, Asia,col_bin)
ax.text(0.02,0.96, 'a)', transform=ax.transAxes, fontsize= 14, fontweight='bold', va='top', ha='left')


ax.set_title(t)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([65,140])
ax.set_ylim([7,55])


Scenarios = ['NPI','NPI_pop','NPI_dose','NPI_sc','NPI_dpop']


for ind_country, country in enumerate(['CHN','IND']):
    ax = fig.add_axes([0.45+0.35*ind_country, 0.4, 0.25, 0.55])
    ax.text(0.02,0.96, ['c)','d)'][ind_country], transform=ax.transAxes, fontsize= 14, fontweight='bold', va='top', ha='left')
    Regions = Result_data[(~Result_data['Downscaled Region'].isin(['China','India','Andaman & Nicobar Islands','Puducherry']))&(Result_data.Region==country)]['Downscaled Region'].unique()
    Results = pd.DataFrame(index=Regions,columns=Scenarios+['employment'])
    for ind_scenario, scenario in enumerate(['NPI','NPI_pop','NPI_dose','NPI_sc','NPI_dpop']):
        
        for t_index in range(len(ts)):
            
            t = ts[t_index]
            Colors = get_colors(Result_data,t,scenario,Va_Variables,Regions,col_bin).set_index('Region_Nam')
        Results[scenario] = Colors['colors']
            # ax.imshow([[[y for y in x[0]]] for x in Colors.loc[Regions].values])
            
    Base_value = Result_data[(Result_data.Region==country)&(Result_data.Scenario=='NPI')&(Result_data.Variable=='Employment|Coal|Downscaled')&(~Result_data['Downscaled Region'].isin(['China','India']))].loc[:,["Downscaled Region","2015"]].set_index("Downscaled Region")
    Results["employment"] = Base_value
    Results = Results.sort_values(by='employment',ascending=False)
    Results = Results.drop('employment',axis=1)
    ax.imshow([[[y for y in z] for z in x] for x in Results.values], aspect='auto')
    ax.set_xticks(range(len(Scenarios)))
    ax.set_xticklabels(Scenarios)

    ax.set_yticks(range(len(Regions)))
    ax.set_yticklabels(Results.index)
    ax.set_xticks(np.arange(Results.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(Results.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)

#%%
# BARCHART DESTINATION 1 Period

provincesChina = Result_data[(Result_data["Region"]=="CHN") & (Result_data["Downscaled Region"]!="China")]["Downscaled Region"].unique()
provincesIndia = Result_data[(Result_data["Region"]=="IND") & (Result_data["Downscaled Region"]!="India")]["Downscaled Region"].unique()

Provinces = [provincesChina, provincesIndia]
# Provinces = [{'Shanxi':[]},{'Jharkhand':[]}]
Countries = ['China', 'India']
nls = [6, 8]

Scenarioss = [['NPI','NDC','NZ'],
            ['NPI_pop','NPI','NPI_dose','NPI_sc','NPI_dpop'],
            ['NDC_pop','NDC','NDC_dose','NDC_sc','NDC_dpop'],
            ['NZ_pop','NZ','NZ_dose','NZ_sc','NZ_dpop']]

Scenarioss= [['NPI','NPI_pop','NPI_dose','NPI_sc','NPI_dpop'],
             ['NDC','NDC_pop','NDC_dose','NDC_sc','NDC_dpop'],
             ['NDC_CCS1','NDC_CCS1_pop','NDC_CCS1_dose','NDC_CCS1_sc','NDC_CCS1_dpop'],
             ['NZ','NZ_pop','NZ_dose','NZ_sc','NZ_dpop'],
             ['NZ_CCS1','NZ_CCS1_pop','NZ_CCS1_dose','NZ_CCS1_sc','NZ_CCS1_dpop']]


Scenarioss_name = [['Central','pop','dose','sc','dpop']]*5
            

t0 = 2020
t1 = 2040

Xs = [
    [x for x in range(len(y))] for y in Scenarioss
]
data_save = {}
fig, axs = plt.subplots(2,
                        5,
                        figsize=(29.21 / 2.54, 13.09 / 2.54),
                        )
for c_index in [0, 1]:
    x = 0
    provinces = Provinces[c_index]
    region = ['China', 'India'][c_index]
    print(region)
    for stype_index in range(5):
        ax = axs[c_index][stype_index]
        if stype_index == 0:
            ax.set_ylabel(region + '\nMillion workers')
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
            for province in provinces:

                Retirement += sum([float(x) for x in Result_data[
                    (Result_data['Variable'] == 'Coal Worker Destination|Retire')
                    & (Result_data['Scenario'] == Scenario) &
                    (Result_data['Downscaled Region']
                     == province)].values.flat[6:][(T < t1) & (T > t0)]])
                Direct += sum([float(x) for x in Result_data[
                    (Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
                    & (Result_data['Scenario'] == Scenario) &
                    (Result_data['Downscaled Region']
                     == province)].values.flat[6:][(T < t1) & (T > t0)]])
                Indirect += sum([float(x) for x in 
                    Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                & (Result_data['Scenario'] == Scenario) &
                                (Result_data['Downscaled Region']
                                 == province)].values.flat[6:][(T < t1)
                                                               & (T > t0)]])
                Unemployed += sum([float(x) for x in 
                    Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                                & (Result_data['Scenario'] == Scenario) &
                                (Result_data['Downscaled Region']
                                 == province)].values.flat[6:][(T < t1)
                                                               & (T > t0)]])
                Hire += sum([-float(x) for x in Result_data[
                    (Result_data['Variable'] == 'Coal Worker Destination|Hire')
                    & (Result_data['Scenario'] == Scenario)
                    & (Result_data['Downscaled Region'] == province)].values.flat[6:][
                        (T < t1) & (T > t0)]])

            data = np.array([Retirement, Direct, Indirect, Unemployed, Hire
                             ]) / 1e6
            data_save[region+'_'+Scenario]=[region,Scenario]+list(data)
            data_shape = np.shape(data)
            cumulated_data = pf.get_cumulated_array(data, min=0)
            cumulated_data_neg = pf.get_cumulated_array(data, max=0)

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

        alines = [x[0] for x in alines]
        labs = [la.get_label() for la in alines]
        ax.axhline(y=0, color='k', linewidth=0.9)
        if c_index == 1:
            ax.set_xticks(X)
            ax.set_xticklabels(
                Scenarioss_name[stype_index],
                rotation=90,
            )
        else:
            ax.set_xticks([])
            ax.set_title(['NPI', 'NDC', 'NDC w/ CCS',
                          '1.5°C', '1.5°C w/ CCS'][stype_index])
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

[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)'])]


#%%
# Plotting the map

b_xlim = [0.25,1]
b_ylim = [0,0.07]#52]


xcmap = plt.cm.Greens
ycmap = sns.color_palette('OrRd', as_cmap=True)

xcmap = matplotlib.colors.LinearSegmentedColormap.from_list('white_to_blue', [(1, 1, 1), '#A15CC4'], N=256)
ycmap = matplotlib.colors.LinearSegmentedColormap.from_list('white_to_gold', [(1, 1, 1), '#E8CC40'], N=256)

n = (10, 10)  # x, y
cmap = xycmap.mean_xycmap(xcmap=xcmap, ycmap=ycmap, n=n)


corner_colors = ("#E3E5D7", "#539BC0", "#F1B62D", "#484C30")
corner_colors = ("#E3E5D7", "#20A5E8", "#FFB205", "#484C30")
cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)

Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China','India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)



Scenarios =  ['NPI','NDC','NZ','NPI_dose','NDC_dose','NZ_dose','NPI_sc','NDC_sc','NZ_sc']

Scenarios_names = ['NPI','NDC','1.5°C','NPI','NDC','1.5°C','NPI','NDC','1.5°C']

Scenarios = ['NPI','NPI_dose','NPI_sc','NDC','NDC_dose','NDC_sc','NZ','NZ_dose','NZ_sc']
Scenarios_names = ['NPI']*3+['NDC']*3+['1.5°C']*3

fig1, axs1 = plt.subplots(3,3, figsize=(18/1.6, 15.3/1.6))

axs = [axs1]

axs = np.array(axs).flatten()
t0 = 2019
t1 = 2050
Data_Chn=[]
Data_Ind=[]


key_data = {}

for s_index in [0,1,2,3,4,5,6,7,8]:
    ax = axs[s_index]
    scenario = Scenarios[s_index]
    Lperpop = []
    cntry = []
    Dc=[]
    DI=[]
    ortho =[]
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
        ortho.append(float((L[T == 2020] - L[T == 2035]) / LF0))
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
        In = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])

        if sum((U + D + In)) != 0:
            Lperpop.append(1-sum(D) / sum((U + D + In)))
        else:
            Lperpop.append(np.nan)

        cntry.append(Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0])
        if Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0] == 'CHN':	
            Dc.append(Lperpop[-1])
        else:
            DI.append(Lperpop[-1])
    
       

    Regions = list(Regions)
    Lperpop = pd.DataFrame(data=np.array([list(Regions), Lperpop,cntry,ortho,destruction,Ls]).transpose(),
                           columns=['Region_Nam', 'Lperpop','Country','ortho','destruction','Workforce'])

    Lperpop['Lperpop'] = pd.to_numeric(Lperpop['Lperpop'], errors='coerce')
    Lperpop['ortho'] = pd.to_numeric(Lperpop['ortho'], errors='coerce')
    Lperpop['destruction'] = pd.to_numeric(Lperpop['destruction'], errors='coerce')
    Lperpop['Workforce'] = pd.to_numeric(Lperpop['Workforce'], errors='coerce')
    
    Data_Chn.append(Lperpop)
    Asia_Data = Asia.merge(Lperpop, on='Region_Nam')

    Asia_Data['Lperpop'] = pd.to_numeric(Asia_Data['Lperpop'], errors='coerce')
    Asia_Data['ortho'] = pd.to_numeric(Asia_Data['ortho'], errors='coerce')
    Asia_Data['destruction'] = pd.to_numeric(Asia_Data['destruction'], errors='coerce')

    norm = ([PowerNorm(gamma=1, vmin=0, vmax=1)] * 6 +
            [PowerNorm(gamma=1, vmin=0, vmax=1)] * 6)[s_index]


    cmapi = xycmap.bivariate_color(sx=Asia_Data.dropna(subset=['Lperpop','ortho'])['Lperpop'].values, sy=Asia_Data.dropna(subset=['Lperpop','ortho'])['ortho'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim)
    cmapi = pd.DataFrame(data=np.array([cmapi,Asia_Data.dropna(subset=['Lperpop','ortho'])['Region_Nam'].values]).transpose(),
                           columns=['colors', 'Region_Nam'])
    
    Asia_Data_with_colors = Asia_Data.merge(cmapi, on='Region_Nam', how='left').fillna('lightgrey')
    Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
        edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)
    Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color='whitesmoke', edgecolor='black', linestyle='--',linewidth=0.5)
    
    for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
        key_data[region+str(s_index)] = {'Region':region,
                                         'Scenario':s_index,
                                        'coordinates':[float(Lperpop[Lperpop['Region_Nam']==region]['Lperpop'].values[0]),
                                    float(Lperpop[Lperpop['Region_Nam']==region]['ortho'].values[0])],
                                    'destruction':float(Lperpop[Lperpop['Region_Nam']==region]['destruction'].values[0])}

    # for region in ['China','India']:  
    #     key_data[region+str(s_index)] = {'Region':'Average \n'+region,
    #                                      'Scenario':s_index,
    #                                     'coordinates':[pf.weighted_average(Lperpop, 'Lperpop','Workforce', ['IND' if region =='India' else 'CHN'][0]),
    #                                                    pf.weighted_average(Lperpop, 'ortho','Workforce', ['IND' if region =='India' else 'CHN'][0])],
    #                                 'destruction':pf.weighted_average(Lperpop, 'destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])}
        

    if s_index <3:
        ax.set_title(['Default','Dose','Structural change'][s_index])
    if s_index in [0,3,6]:
        ax.set_ylabel(Scenarios_names[s_index])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([65,140])
    ax.set_ylim([7,55])


Cscenario = sns.color_palette()#['#F18872','#A7C682','#97CEE4']


cax = fig1.add_axes([1, 0.18+0.29, 0.29, 0.29])
cax = xycmap.bivariate_legend(ax=cax, sx=Asia_Data.dropna(subset=['Lperpop','ortho'])['Lperpop'].values, sy=Asia_Data.dropna(subset=['Lperpop','ortho'])['ortho'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim) #xlims=(0,0.5),ylims=(0,0.06)
cax.set_xlabel('Share of workers not \n finding new employment',fontsize=11)
cax.set_ylabel('Decrease in relative coal jobs',fontsize=11)
if min(n)>5:
    cax.set_xticks([pf.interpol(x,b_xlim,cax.get_xlim()) for x in [0.5,0.7,0.9]])
    cax.set_yticks([pf.interpol(x,b_ylim,cax.get_ylim()) for x in [0,0.025,0.05]])
    cax.set_xticklabels(["50%","70%","90%"],fontsize=11)
    cax.set_yticklabels(["0%","2.5%","5%"],fontsize=11)

for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [6,7,8]:
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)

for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for col_index in [0,1,2]:
        s_index = [6,7,8][col_index]
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
        cax.scatter(xs[-1],ys[-1],color=Cscenario[col_index],marker='o',s=key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)
    cax.annotate(region,(xs[0],ys[0]-0.1),ha='center',va='top',fontsize=8)

alines = []
alines.append(cax.scatter([], [], color='white', marker='o',s=0))
alines.append(cax.scatter([], [], color=Cscenario[0], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Cscenario[1], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Cscenario[2], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color='white', marker='o',s=0))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=2e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=4e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=6e5/1200))
labels = ['1.5°C \nscenario:','Default','Dose', 'Structural Change','Job losses \n [people]','200k','400k','600k']
cax.legend(handles=alines,
           labels=labels,
           loc='center right',
           bbox_to_anchor=(1.5, 0.5),
           frameon=False,
           fontsize=8)


#%% Plot unemployment



#  Mapping Unemployment
Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)

zlim = [0, 35]
tickz = np.array([5]+list(np.arange(25,176,50)))

Ts = [2020, 2035, 2050]
Scenarios = ['NPI','NDC','NZ']
Scenarios_names = ['NPI','NDC','NZ']

Scenarios = ['NPI','NPI_dose','NPI_sc','NDC','NDC_dose','NDC_sc','NZ','NZ_dose','NZ_sc']

Scenarios = ['NPI','NDC','NZ',
             'NPI_dose','NDC_dose','NZ_dose',
             'NPI_sc','NDC_sc','NZ_sc',
             'NPI_pop','NDC_pop','NZ_pop',
             'NPI_dpop','NDC_dpop','NZ_dpop']

Scenarios_names = Scenarios
ind_t = 0
t = 2040

fig, axs = plt.subplots(5,
                        3,
                        figsize=(20/2.54,30/2.54))


Data_Chn = []
Data_Ind = []
for s_index, scenario in enumerate(Scenarios):
    ax = axs.flatten()[s_index]#[ind_t]
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

    if s_index == 6:
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
axn = axs[3,2].inset_axes([1.3, -1, 3, 3])
us = {}
for s_i, scenario in enumerate(Scenarios[:3]):
    
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

        axn.plot(T[T < 2070],u[T < 2070],label=country,color=Colors[scenario.split('_')[0]],linestyle=['-','--'][c_i])

axn.set_ylabel('Unemployment rate [%]')
axn.set_ylim([0,14])

[ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 12, fontweight='bold', va='top', ha='left') for ax, label in zip([axs[0,0],axn],['a)','b)'])]
# %% Plotting the map
import importlib
importlib.reload(pf)


Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China','India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)


# Defining the scenarios
Scenarios =  ['NPI','NDC','NZ',
              'NPI_sc','NDC_sc','NZ_sc',
              'NPI_pop','NDC_pop','NZ_pop',]
Scenarios_names = Scenarios

t0 = 2019
t1 = 2040

# Creating the figure
fig1, axs1 = plt.subplots(3,3, figsize=(20/2.54,13/2.54))
axs = [axs1]
axs = np.array(axs).flatten()


# Initiating the data lists
Data_Chn=[]
Data_Ind=[]
key_data = {}
    
for s_index, scenario in enumerate(Scenarios):
    ax = axs[s_index]
    Asia_Data, key_data, cmap_zip = pf.plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names)



cmap, b_xlim, b_ylim, n = cmap_zip

# ====================================
# Legend


Cscenario = [sns.color_palette()[x] for x in [0,1,4]]


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

for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [2,5,8]:
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)

for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for col_index in [0,1,2]:
        s_index = [2,5,8][col_index]
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
        cax.scatter(xs[-1],ys[-1],color=Cscenario[col_index],marker='o',s=key_data[region+str(s_index)]['destruction']/1200,edgecolor='k',linewidth=0.4,zorder=1)
    cax.annotate(region,(xs[0],ys[0]-0.1),ha='center',va='top',fontsize=8)


[ax.set_title('') for ax in axs.flatten()]
[ax.set_title(x) for ax, x in zip(axs.flatten()[:3],['NPi','NDC-LTT','1.5°C'])]
[axs[y].set_ylabel(x) for y, x in zip([0,3,6],['Default','Structural Change','Population'])]



alines = []
alines.append(cax.scatter([], [], color='white', marker='o',s=0))
alines.append(cax.scatter([], [], color=Cscenario[0], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Cscenario[1], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color=Cscenario[2], marker='o',edgecolor='k',linewidth=0.4, s=40))
alines.append(cax.scatter([], [], color='white', marker='o',s=0))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=2e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=4e5/1200))
alines.append(cax.scatter([], [], color='Grey', marker='o',edgecolor='k',linewidth=0.4, s=6e5/1200))
labels = ['1.5°C \nscenario:','Default', 'Structural Change', 'Population','Job losses \n [people]','200k','400k','600k']
cax.legend(handles=alines,
           labels=labels,
           loc='center right',
           bbox_to_anchor=(1.7, 0.5),
           frameon=False,
           fontsize=8)
fig1.subplots_adjust(wspace=-0.2)



#%% Mapping ternary plot of evolving economic structure based on Value added
# Value added results are not outputed by coal labour nexus: 
# so it is necessary to recompute them here using structural change indicators


Scenarios_name = {'NPI':'WO-NPi-ElecIndus','NDC': 'WO-NDCLTT-ElecIndus','NZ': 'WO-15C-ElecIndus'}

Sector_variables = {'indus': 'Industry and Construction','agric':'Agriculture','servi':'Services'}

VA_Results = pd.DataFrame(columns = ['Scenario','Region', 'Downscaled Region', 'Variable']+[str(x) for x in range(2015,2101)])
VA_Results=VA_Results.set_index(["Scenario","Region","Downscaled Region",'Variable']) 

Countries = ['CHN','IND']

for scenario, im_scenario in Scenarios_name.items():
    Structure = pd.read_csv('output_data\Econ_structural_change_'+scenario+'.csv')
    
    for sector, variable in Sector_variables.items():
        for country in Countries:
            #national value added
            NVA = Imaclim_data[(Imaclim_data.Variables=='Value Added|'+variable)&
                        (Imaclim_data.Region==country)&
                        (Imaclim_data.Scenario==Scenarios_name[scenario])]

            # Downscaling keys
            DK = Structure[(Structure.Region_name==country)&
                        (~Structure.Subregion_name.isin(['China','India']))]
            DK = DK.loc[:,['Subregion_name']+[sector + str(x) for x in range(2015,2101)]].set_index('Subregion_name') 
            DK.columns = range(2015,2101)

            for region in DK.index:
                VA_Results.loc[(scenario,country,region,sector),:] =  (DK.loc[region,2015]*NVA.values[0][5:])
                VA_Results.loc[(scenario+'_sc',country,region,sector),:] =  (DK.loc[region,:]*NVA.values[0][5:]).values
            
VA_Results=VA_Results.reset_index()
#%%
VAshare_results = pd.DataFrame(columns = ['Scenario', 'Downscaled Region','Variable']+[str(x) for x in range(2015,2101)] )
VAshare_results=VAshare_results.set_index(['Scenario', 'Downscaled Region','Variable'])
for region in VA_Results['Downscaled Region'].unique():
    for scenario in VA_Results.Scenario.unique():
        filtered = VA_Results[(VA_Results['Downscaled Region']==region)&
                                (VA_Results.Scenario==scenario)]
        for variable in VA_Results.Variable.unique():
            tot = filtered.sum(axis=0).loc[[str(x) for x in range(2015,2101)]]
            tot[tot==0]=1
            VAshare_results.loc[(scenario,region,variable),:] = (filtered.loc[filtered.Variable==variable,[str(x) for x in range(2015,2101)]]/tot).values[0]
VAshare_results=VAshare_results.reset_index()
# %%

Regions = Result_data[~Result_data['Downscaled Region'].isin(['China','India','Andaman & Nicobar Islands','Puducherry'])]['Downscaled Region'].unique()


fig = plt.figure(figsize=(25, 20))

ts = [2015,2040,2060]

subdivisions = 20
ax, col_bin = define_ternary_color_space(fig,subdivisions)

Va_Variables = {
    "agri" : "agric",
    'indus' : "indus",
    "serv" : 'servi'
}



for ind_scenario, scenario in enumerate(['NPI','NPI_sc']):
    for t_index in range(len(ts)):
        ax = fig.add_axes([0.4+0.255*t_index, 0.7-0.25*ind_scenario, 0.25, 0.25])
        t = ts[t_index]
        ax = ternary_map(ax, VAshare_results, t, scenario, Va_Variables, Regions, Asia,col_bin)

        if ind_scenario == 0:
            ax.set_title(t, fontsize=14)
        if t_index == 0:
            ax.set_ylabel(scenario, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([65,140])
        ax.set_ylim([7,55])


# %%


# Alternative to maps
t0=2020
# - Heatmap

Scenarios = ['NZ','NZ_dose','NZ_sc','NZ_dpop','NZ_pop']
Scenarios_names = ['Central','Dose','SC','dpop','pop']
scenario = Scenarios[0]
var = 'Workforce'
T1 = 2035
Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                ['Downscaled Region'].values)


Region_with_cntr = [region+' ('+Region_indexes.loc[Region_indexes["Subregion_name"]==region,"Region_name"].values[0]+')' for region in Regions]

zlim, norm, colormap = pf.semisymlognorm()
Ls = []
share_n_finding = []
All_data = pd.DataFrame(index=Region_with_cntr,columns=['Workforce','Destruction','Country'])
Calced_data = []
Destruction = []
CNTRY = [Region_indexes.loc[Region_indexes["Subregion_name"]==region,"Region_name"].values[0] for region in Regions]

# Iterating over regions
for region in list(Regions):
    if type(T1) is str:
        threshold = 0.8
        t1 =pf.finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==region)&
                        (Result_data.Scenario==scenario)],threshold,T)

    else:
        t1=T1
    
    Calced_data = pf.calc_workforce(Result_data,region,Calced_data,scenario) 

    y = Result_data[(Result_data['Downscaled Region'] == region)
                    & (Result_data['Variable'] == 'Employment|Coal|Downscaled') &
                    (Result_data['Scenario'] == scenario)]
    Destruction.append(y.loc[:,str(t1)]-y.loc[:,str(t0)])

Calced_data = pd.DataFrame(data=np.array([Region_with_cntr, Calced_data]).transpose(),
                    columns=['Region_Nam', 'Calced_data']).set_index('Region_Nam')
Destruction = pd.DataFrame(data=np.array([Region_with_cntr, Destruction]).transpose(),
                    columns=['Region_Nam', 'Destruction']).set_index('Region_Nam')


All_data['Workforce'] = Calced_data['Calced_data'].astype(float)
All_data['Destruction'] = Destruction['Destruction'].astype(float)
All_data['Country'] = CNTRY
var = "Finding_new_emp"
for T1 in [2035,2050]:
    for ind_scenario, scenario in enumerate(Scenarios):
        Calced_data = []
        for region in list(Regions):
            if type(T1) is str:
                threshold = 0.8
                t1 =pf.finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==region)&
                                (Result_data.Scenario==scenario)],threshold,T)

            else:
                t1=T1
            Calced_data=pf.calc_share_n_finding(Result_data,region,T,t0,t1,Calced_data,scenario)

        Calced_data = pd.DataFrame(data=np.array([Region_with_cntr, Calced_data]).transpose(),
                            columns=['Region_Nam', 'Calced_data']).set_index('Region_Nam')

        All_data[Scenarios_names[ind_scenario]+'\n'+str(T1)] = Calced_data['Calced_data'].astype(float)


All_data.sort_values(['Country','Workforce'],ascending=False,inplace=True)
All_data.drop(All_data[All_data['Workforce']==0].index,inplace=True)
All_data.drop(['Country'],axis=1,inplace=True)
#%%
# Add a blank column to create a wider gap between 2030 and 2050 scenarios
fig, ax = plt.subplots(figsize= [x*[1,2][ind] for ind,x in enumerate(pf.standard_figure_size())])
sns.heatmap(All_data.drop(['Workforce','Destruction'],axis=1).astype('float'),vmin=0,vmax=1, linewidth = 0.5, cmap="rocket_r", ax=ax,  cbar=True, cbar_kws={"label": "Share not finding new employment" })

ax.axvline(x=5,color='white',linewidth = 7)
ax.axvline(x=1,color='white',linewidth = 2)
ax.axvline(x=6,color='white',linewidth = 2)
ax.axhline(y=12,color='white',linewidth = 4)
ax.xaxis.tick_top()
# cax2 = ax.inset_axes([1.07, 0, 0.05, 1])
# cbar = fig.colorbar(cax=cax2, orientation='vertical')
# cax2.spines[:].set_visible(False)


#%%
# - Scatter

Colors = {
    'NZ':pf.defining_waysout_colour_scheme()['NZ'],
    'NZ_dose':sns.color_palette()[0],
    'NZ_sc':sns.color_palette()[4],
    'NZ_pop':sns.color_palette()[5],
    'NZ_dpop':sns.color_palette()[8],
}

Scen_for_comparison = ['SC','Dose']
Scenarios = ['NZ_dose','NZ_sc','NZ_dpop','NZ_pop','NZ']
Scenarios_names = ['Dose','SC','dpop','pop','Central']

T1 = 2035
fig, ax = plt.subplots(figsize=(7, 7))
for ind_region, region in enumerate(All_data.index):
    for ind_scenario, scenario in enumerate(Scenarios_names):
        if scenario == 'Central':
            edge = 'k'
        else:
            edge = None
        ax.scatter(All_data.loc[region, scenario+'\n'+str(T1)],len(All_data.index)-ind_region-1, s=-All_data.loc[region,'Destruction']/7e3, label=region, color=Colors[Scenarios[ind_scenario]],zorder=1,edgecolors=edge)
    xs = [min([All_data.loc[region, x+'\n'+str(T1)] for x in Scen_for_comparison]),max([All_data.loc[region, x+'\n'+str(T1)] for x in Scen_for_comparison])]
    ax.plot(xs,
            [len(All_data.index)-ind_region-1]*2,color='gray',zorder=0,alpha=0.4)
    ax.text(np.mean(xs),len(All_data.index)-ind_region-1,f'{(xs[1]-xs[0])*100:.0f}%',ha='center',va='center',fontsize=10)


ax.set_yticks(range(len(All_data.index)))
ax.set_yticklabels(All_data.index[::-1])
ax.set_ylim([-0.5,33.5])
ax.set_xlim([0,1])
ax.set_xticklabels([f'{x*100:.0f}%' for x in ax.get_xticks()])


legend_elements = [
    plt.scatter([], [], color=Colors['NZ'], label='Central', edgecolor='k', s=100),
    plt.scatter([], [], color=Colors['NZ_dose'], label='Dose', s=100),
    plt.scatter([], [], color=Colors['NZ_sc'], label='SC', s=100),
    plt.scatter([], [], color=Colors['NZ_dpop'], label='dpop', s=100),
    plt.scatter([], [], color=Colors['NZ_pop'], label='pop', s=100),
    plt.scatter([], [], color='gray', label='Job losses (800k)',  alpha=0.6, s=800/7),
    plt.scatter([], [], color='gray', label='Job losses (400k)',  alpha=0.6, s=400/7),
    plt.scatter([], [], color='gray', label='Job losses (200k)',  alpha=0.6, s=200/7),
]

ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

# %%
