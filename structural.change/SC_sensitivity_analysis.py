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
# Map shapefile from naturalearthdata.com
Asia = gpd.read_file('../notebooks/shapefiles/Asia.shp')
Asia.loc[Asia.Region_Nam=="Orissa","Region_Nam"] = "Odisha"

#%%
#Importing module results
T = range(2015, 2101)
T = np.array(T)

input_directory = os.path.join("..","coal.labour.nexus","output")


file_name = list(np.array([[input_directory+'/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NDC','NPI','NZ']] for y in ['','_Pop','_dpop','_sc','_Dose']]).flatten())

Result_data = []
for file in file_name:
    Result_data.append(pd.read_csv(file))#, dtype=str))

Result_data = pd.concat(Result_data, ignore_index=True)

Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].applymap(lambda x: str(x).replace('D', 'E'))
Result_data.iloc[:, 6:] = Result_data.iloc[:, 6:].apply(pd.to_numeric, errors='coerce')

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


# Legend

subdivisions = 20  # Number of subdivisions per axis
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

# Plot evolution of key regions on the ternary plot
for region in ['Shanxi','Jharkhand']:
    alines = []
    for ind_scenario, scenario in enumerate(['NPI','NPI_Pop','NPI_Dose','NPI_sc']):

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



# Plot data    
t = 2015
# data = output
ax = fig.add_axes([0.1,1-0.3, 0.25, 0.25])

Lperpop = []
cntry = []
cols = []

for region in list(Regions):
    c_index = 0 if region in Cregion else 1

    agri = float(Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Agriculture|Share|Downscaled")&(Result_data.Scenario=='NPI')][str(t)].values[0])
    indus = float(Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Industry|Share|Downscaled")&(Result_data.Scenario=='NPI')][str(t)].values[0])
    serv = float(Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Services|Share|Downscaled")&(Result_data.Scenario=='NPI')][str(t)].values[0])

    if ('nan' in [agri,indus,serv]) or pd.isnull(agri) or pd.isnull(indus) or pd.isnull(serv) :
   
        color = 'lightgrey'
    else:
        color = col_bin_f(agri,indus,serv,col_bin, subdivisions)
    cols.append(color)

Lperpop = pd.DataFrame(data=np.array([list(Regions), cols],dtype=object).transpose(),
                    columns=['Region_Nam', 'colors'])

Asia_Data_with_colors = Asia[Asia['CNTRY_Name'].isin(['China','India'])].merge(Lperpop, on='Region_Nam', how='left').fillna('lightgrey')
Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
    edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)

Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
Asia[Asia['region']=='Disputed'].plot(ax=ax, color='whitesmoke', edgecolor='black', linestyle='--',linewidth=0.5)

ax.set_title(t)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([65,140])
ax.set_ylim([7,55])



for ind_scenario, scenario in enumerate(['NPI','NPI_Pop','NPI_Dose','NPI_sc','NPI_dpop']):
    for t_index in range(len(ts)):
        # ax = fig.add_axes([0.4+[0,0.25,0.5,0.75][ind_scenario],1-0.3*(t_index+1), 0.25, 0.25])
        ax = fig.add_axes([0.4+0.255*t_index, 0.7-0.25*ind_scenario, 0.25, 0.25])
        t = ts[t_index]
        Lperpop = []
        cntry = []
        cols = []
        
        for region in list(Regions):
            c_index = 0 if region in Cregion else 1
            
            agri =float(Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Agriculture|Share|Downscaled")&(Result_data.Scenario==scenario)][str(t)].values[0])
            indus = float(Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Industry|Share|Downscaled")&(Result_data.Scenario==scenario)][str(t)].values[0])
            serv = float(Result_data[(Result_data['Downscaled Region']==region)&(Result_data.Variable=="Employment|Services|Share|Downscaled")&(Result_data.Scenario==scenario)][str(t)].values[0])

            color = col_bin_f(agri,indus,serv,col_bin, subdivisions)
            cols.append(color)


        
        Lperpop = pd.DataFrame(data=np.array([list(Regions), cols],dtype=object).transpose(),
                            columns=['Region_Nam', 'colors'])

        Asia_Data_with_colors = Asia[Asia['CNTRY_Name'].isin(['China','India'])].merge(Lperpop, on='Region_Nam', how='left').fillna('lightgrey')
        Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
            edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)
        
        Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
        Asia[Asia['region']=='Disputed'].plot(ax=ax, color='whitesmoke', edgecolor='black', linestyle='--',linewidth=0.5)

        if ind_scenario == 0:
            ax.set_title(t, fontsize=14)
        if t_index == 0:
            ax.set_ylabel(scenario, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([65,140])
        ax.set_ylim([7,55])

#%%
# BARCHART DESTINATION 1 Period

provincesChina = Result_data[(Result_data["Region"]=="CHN") & (Result_data["Downscaled Region"]!="China")]["Downscaled Region"].unique()
provincesIndia = Result_data[(Result_data["Region"]=="IND") & (Result_data["Downscaled Region"]!="India")]["Downscaled Region"].unique()

Provinces = [provincesChina, provincesIndia]
# Provinces = [{'Shanxi':[]},{'Jharkhand':[]}]
Countries = ['China', 'India']
nls = [6, 8]

Scenarioss = [['NPI','NDC','NZ'],
            ['NPI_Pop','NPI','NPI_Dose','NPI_sc','NPI_dpop'],
            ['NDC_Pop','NDC','NDC_Dose','NDC_sc','NDC_dpop'],
            ['NZ_Pop','NZ','NZ_Dose','NZ_sc','NZ_dpop']]


Scenarioss_name = [['NPI','NDC','NZ'],
            ['NPI_Pop','NPI','NPI_Dose','NPI_sc','NPI_dpop'],
            ['NDC_Pop','NDC','NDC_Dose','NDC_sc','NDC_dpop'],
            ['NZ_Pop','NZ','NZ_Dose','NZ_sc','NZ_dpop']]
            

t0 = 2020
t1 = 2050

Xs = [
    list(range(len(Scenarioss[0]))), [x for x in range(len(Scenarioss[1]))],
    [x for x in range(len(Scenarioss[2]))],
    [x for x in range(len(Scenarioss[3]))]
]
data_save = {}
fig, axs = plt.subplots(2,
                        4,
                        figsize=(29.21 / 2.54, 13.09 / 2.54),
                        gridspec_kw={
                            'width_ratios': [1, 5.5 / 3, 5.5 / 3, 5.5 / 3],
                            'height_ratios': [1, 1]
                        })
for c_index in [0, 1]:
    x = 0
    provinces = Provinces[c_index]
    region = ['China', 'India'][c_index]
    print(region)
    for stype_index in range(4):
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
            ax.set_title(['Central', 'NPI', 'NDC',
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

Scenarios = ['NPI','NPI_Dose','NPI_sc','NDC','NDC_Dose','NDC_sc','NZ','NZ_Dose','NZ_sc']
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

    for region in ['China','India']:  
        key_data[region+str(s_index)] = {'Region':'Average \n'+region,
                                         'Scenario':s_index,
                                        'coordinates':[pf.weighted_average(Lperpop, 'Lperpop','Workforce', ['IND' if region =='India' else 'CHN'][0]),
                                                       pf.weighted_average(Lperpop, 'ortho','Workforce', ['IND' if region =='India' else 'CHN'][0])],
                                    'destruction':pf.weighted_average(Lperpop, 'destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])}
        

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

for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
    xs = []
    ys = []
    for s_index in [6,7,8]:
        xs.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][0],b_xlim,cax.get_xlim()))
        ys.append(pf.interpol(key_data[region+str(s_index)]['coordinates'][1],b_ylim,cax.get_ylim()))
    cax.plot(xs,ys,color='k',alpha=0.25,zorder=-0,linewidth=5)

for region in ['China','India','Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
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