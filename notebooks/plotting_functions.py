# Importing libraries
# import geopandas as gpd
import pandas as pd
import numpy as np
# from pyproj import CRS

# Plotting parameters
import matplotlib.pyplot as plt
import xycmap
import seaborn as sns
import os
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors





#=========================================================================================================
# Standard figure size
def standard_figure_size():
    width = 20
    height= 9
    cm2in = 1/2.54
    return (width*cm2in,height*cm2in)


#=========================================================================================================
# Save figure
def save_figure(fig,name,format,dpi=400):
    figure_path = os.path.join('figures')

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    fig.savefig(os.path.join(figure_path,name+'.'+format),dpi=dpi, bbox_inches='tight')


#=========================================================================================================
# Defining the colors for the different scenarios centrally


def defining_waysout_colour_scheme():
    Colors = {'NPI':"steelblue",
            'NDC':"tomato",
            'NZ':"lightseagreen",
            'WO-NPi-ElecIndus-CCS0':"steelblue",
            'WO-NDCLTT-ElecIndus-CCS0':"tomato",
            'WO-15C-ElecIndus-CCS0':"lightseagreen",
            'WO-NDCLTT-ElecIndus-CCS1':"palevioletred",
            'WO-15C-ElecIndus-CCS1':"yellowgreen",
            'WO-18C-ElecIndus':'blue',
            'NPI_gem':sns.color_palette('pastel')[4],
            'NDC_gem':sns.color_palette('pastel')[3],
            'NZ_gem':sns.color_palette('pastel')[1],
            'NDC_CCS1':"palevioletred",
            'NZ_CCS1':"yellowgreen"}
    return Colors


def defining_regions_colors():
    """
    Adapting color palette used in IPCC AR6 for R10 regions to R12 Imaclim regions 
    """
    Colors = {
        "USA":'#C77B10',
        "CAN":"#A54133",
        "EUR":"#003C6C",
        "JAN":"#5290C8",
        "CIS":"#FCCB76",
        "CHN":"#DD4B15",
        "IND":"#F6AA00",
        "BRA":"#7299A4",
        "MDE":"#7E7E7D",
        "AFR":"#08A7CD",
        "RAS":"#EC9867",
        "RAL":"#797A9C"
    }
    return Colors

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


#=========================================================================================================
#    Misc
def interpol(x, xlim, ylim):

    if (x<xlim[0])|(x>xlim[1]): #Not plotting beyond axis limits
        y= np.nan
    else:
        y = ylim[0] + (ylim[1]-ylim[0])/(xlim[1]-xlim[0])*(x-xlim[0])
    return y


# Functions used for bivariate plot 
def weighted_average(df, value, weight, country):
    """
    The weighted average function is used to when mapping the vulnerability of regions to the coal transition
    It allows to calculate the average of a variable in a country, using a weight such as the workforce
    """
    return (df[df['Region'] == country][value] * df[df['Region'] == country][weight]).sum() / df[df['Region'] == country][weight].sum()


def append_real_results(share_n_finding,D,U,In):
    if sum((U + D + In)) != 0:
        share_n_finding.append(1-sum(D+In) / sum((U + D + In)))
    else:
        share_n_finding.append(np.nan)
    return share_n_finding


#=========================================================================================================
#    Stacked bars function
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


def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

    
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#                                                              Core Plots
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, T1, Asia, s_index, key_data, Scenarios_names):
    """
    This function maps the exposure (ratio of coal job destruction to labour force)
    and vulnerability (share of laid off workers going to unemployment) 
    of regions of China and India to the coal transition
    """
    # Defining the colormap
    b_xlim = [0.3,0.81]
    b_xlim = [0.3,0.88]
    b_ylim = [np.log(3e-4),np.log(0.065)]
    b_ylim = [np.log(9e-4),np.log(0.08)]


    n = (10, 10)  # x, y

    corner_colors = ("#e8e8e8",   "#C85a5a", "#64acbe", "#574249")
    cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
    cmap_zip = [cmap, b_xlim, b_ylim, n]
    
    color_not_plotted = '#EAEAEA' #Applying the same colour to all regions with no coal employment or outside the scope of analysis
    
    # Initializing the lists
    share_n_finding = []
    cntry = []
    Dc=[]
    DI=[]
    share_destruction =[]
    creation = []
    destruction = []
    Ls = []

    # Iterating over regions
    for region in list(Regions):

        if type(T1) is str:
            threshold = 0.8
            t1 =finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==region)&
                            (Result_data.Scenario==scenario)],threshold,T)

        else:
            t1=T1


        L = Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Employment|Coal|Downscaled') 
                        & (Result_data['Scenario'] == scenario)].values[0][6:]
        LF0 = float(
            Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Labour Force|Downscaled')
                        & (Result_data['Scenario'] == scenario)].values[0][6])
        Ls.append(L[0])

        y = float((L[T == 2020] - L[T == t1]) / LF0)
        creation.append(1) if ((L[T==2020]!=0)&(((type(T1) is str)&(t1==2100))|(y<0))) else creation.append(0) #
        y = np.nan if y == 0 else np.log(y)
        
        share_destruction.append(y)

        destroyed = float((L[T == 2020] - L[T == t1]) )
        y = np.nan if destroyed<=0 else y #Mapping regions with no decrease in employment in grey
        destruction.append(destroyed)


        
        D = np.array(
            Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
                        & (Result_data['Scenario'] == scenario) 
                        & (Result_data['Downscaled Region'] == region)].values.flat[6:][(T < t1) & (T > t0)])
        U = np.array(Result_data[
            (Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                     & (Result_data['Scenario'] == scenario) 
                                 & (Result_data['Downscaled Region'] == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])
        In = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])

        share_n_finding=append_real_results(share_n_finding,D,U,In)

        cntry.append(Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0])
        
        if Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0] == 'CHN':	
            Dc.append(share_n_finding[-1])
        else:
            DI.append(share_n_finding[-1])
    

    
    Regions = list(Regions)
    share_n_finding = pd.DataFrame(
        data=np.array([list(Regions), share_n_finding,cntry,share_destruction,destruction,Ls,creation]).transpose(),
        columns=['Region_Nam', 'share_n_finding','Region','share_destruction','destruction','Workforce','creation']
        )

    share_n_finding['share_n_finding'] = pd.to_numeric(share_n_finding['share_n_finding'], errors='coerce')
    share_n_finding['share_destruction'] = pd.to_numeric(share_n_finding['share_destruction'], errors='coerce')
    share_n_finding['destruction'] = pd.to_numeric(share_n_finding['destruction'], errors='coerce')
    share_n_finding['Workforce'] = pd.to_numeric(share_n_finding['Workforce'], errors='coerce')


    
   
    Asia_Data = Asia.merge(share_n_finding, on='Region_Nam')

    Asia_Data['share_n_finding'] = pd.to_numeric(Asia_Data['share_n_finding'], errors='coerce')
    Asia_Data['share_destruction'] = pd.to_numeric(Asia_Data['share_destruction'], errors='coerce')
    Asia_Data['destruction'] = pd.to_numeric(Asia_Data['destruction'], errors='coerce')


    cmapi = xycmap.bivariate_color(sx=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_n_finding'].values, sy=Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['share_destruction'].values, cmap=cmap, xlims=b_xlim,ylims=b_ylim)
    cmapi = pd.DataFrame(data=np.array([cmapi,Asia_Data.dropna(subset=['share_n_finding','share_destruction'])['Region_Nam'].values]).transpose(),
                           columns=['colors', 'Region_Nam'])
    
    
    Asia_Data_with_colors = Asia_Data.merge(cmapi, on='Region_Nam', how='left').fillna(color_not_plotted)
    
    Asia_Data_with_colors.loc[Asia_Data_with_colors['creation']=='1','colors']='lightgreen'
    
    Asia_Data_with_colors.plot(
        ax=ax, 
        color=Asia_Data_with_colors['colors'],
        edgecolor='black',
        linewidth=0.5,
        rasterized=True,
        alpha=1)


    Asia[Asia['region']=='Asia'].plot(ax=ax, color=color_not_plotted, edgecolor='black',linewidth=0.5,rasterized=True)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color=color_not_plotted, edgecolor='black', linestyle='--',linewidth=0.5,rasterized=True)
    
    for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh','Henan','Shandong']:
        key_data[region+str(s_index)] = {'Downscaled Region':region,
                                         'Scenario':scenario,
                                         'coordinates':[
                                             float(share_n_finding[share_n_finding['Region_Nam']==region]['share_n_finding'].values[0]),
                                             float(share_n_finding[share_n_finding['Region_Nam']==region]['share_destruction'].values[0])
                                             ],
                                         'destruction':float(share_n_finding[share_n_finding['Region_Nam']==region]['destruction'].values[0])}

    for region in ['China','India']:  
        key_data[region+str(s_index)] = {'Downscaled Region':'Average \n'+region,
                                         'Scenario':scenario,
                                         'coordinates':[
                                             weighted_average(share_n_finding, 'share_n_finding','Workforce', ['IND' if region =='India' else 'CHN'][0]),
                                             weighted_average(share_n_finding, 'share_destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])
                                             ],
                                         'destruction':weighted_average(share_n_finding, 'destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])}
        

    
    ax.set_title(Scenarios_names[s_index])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([65,140])
    ax.set_ylim([7,55])


    return Asia_Data,key_data,cmap_zip




#=========================================================================================================
def destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index):
    """
    This function is used to graph a stacked bar plot of the destination of exiting coal workers
    """
    
    Retirement = 0
    Direct = 0
    Indirect = 0
    Unemployed = 0
    Hire = 0
    for province, pos in provinces.items():

        R = sum(
            Result_data[
                (Result_data["Variable"] == "Coal Worker Destination|Retire")
                & (Result_data["Scenario"] == Scenario)
                & (Result_data["Downscaled Region"] == province)
            ].values.flat[6:][(T < t1) & (T > t0)]
        )
        D = sum(
            Result_data[
                (Result_data["Variable"] == "Coal Worker Destination|Instant Match")
                & (Result_data["Scenario"] == Scenario)
                & (Result_data["Downscaled Region"] == province)
            ].values.flat[6:][(T < t1) & (T > t0)]
        )
        In = sum(
            Result_data[
                (Result_data["Variable"] == "Coal Worker Destination|Delayed Match")
                & (Result_data["Scenario"] == Scenario)
                & (Result_data["Downscaled Region"] == province)
            ].values.flat[6:][(T < t1) & (T > t0)]
        )
        U = sum(
            Result_data[
                (Result_data["Variable"] == "Coal Worker Destination|Unemployment")
                & (Result_data["Scenario"] == Scenario)
                & (Result_data["Downscaled Region"] == province)
            ].values.flat[6:][(T < t1) & (T > t0)]
        )
        H = sum(
            -Result_data[
                (Result_data["Variable"] == "Coal Worker Destination|Hire")
                & (Result_data["Scenario"] == Scenario)
                & (Result_data["Downscaled Region"] == province)
            ].values.flat[6:][(T < t1) & (T > t0)]
        )

        
        data_save[(province, Scenario,t1)] = np.array([R, D, In, U, H]) / 1e6
        Retirement += R
        Direct += D
        Indirect += In
        Unemployed += U
        Hire += H

    if t1==2100: # Not plotting if employment never meets threshold
        Retirement=0 
        Direct =0 
        Indirect =0 
        Unemployed =0 
        Hire =0 

    data = np.array([Retirement, 
                     Direct, 
                     Indirect, 
                     Unemployed, 
                     Hire
                    ]) / 1e6
    
    data_save[(region, Scenario,t1)]=data
    data_shape = np.shape(data)
    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)

    # Re-merge negative and positive data.
    row_mask = (data < 0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data

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

    return data_save, alines

#=========================================================================================================
def defining_province_grid():
    """
    This function gives approximate relative coordinates of each province/state 
    this enable grid plotting of results 
    """
    provincesChina = {
        'Heilongjiang': (0, 5),
        'Xinjiang': (1, 0),
        'Qinghai': (1, 1),
        'Ningxia': (1, 2),
        'Inner Mongolia': (1, 3),
        'Liaoning': (1, 4),
        'Jilin': (1, 5),
        'Tibet': (2, 0),
        'Gansu': (2, 1),
        'Shaanxi': (2, 2),
        'Shanxi': (2, 3),
        'Hebei': (2, 4),
        'Sichuan': (3, 1),
        'Chongqing': (3, 2),
        'Hubei': (3, 3),
        'Henan': (3, 4),
        'Shandong': (3, 5),
        'Yunnan': (4, 1),
        'Guizhou': (4, 2),
        'Hunan': (4, 3),
        'Anhui': (4, 4),
        'Jiangsu': (4, 5),
        'Guangxi': (5, 2),
        'Jiangxi': (5, 3),
        'Zhejiang': (5, 4),
        'Shanghai': (5, 5),
        'Guangdong': (6, 3),
        'Fujian': (6, 4),
        'Hainan': (7, 4)
    }

    provincesIndia = {
        'Jammu & Kashmir': (0, 2),
        'Punjab': (1, 2),
        'Himachal Pradesh': (1, 3),
        'Uttarakhand': (1, 4),
        'Nagaland': (1, 7),
        'Rajasthan': (2, 1),
        'Haryana': (2, 2),
        'Uttar Pradesh': (2, 3),
        'Bihar': (2, 4),
        'Sikkim': (2, 5),
        'Assam': (2, 6),
        'Manipur': (2, 7),
        'Gujarat': (3, 0),
        'Madhya Pradesh': (3, 1),
        'Delhi': (3, 2),
        'Chhattisgarh': (3, 3),
        'Jharkhand': (3, 4),
        'Meghalaya': (3, 5),
        'Tripura': (3, 6),
        'Maharashtra': (4, 1),
        'Telangana': (4, 2),
        'Odisha': (4, 3),
        'West Bengal': (4, 4),
        'Goa': (5, 1),
        'Karnataka': (5, 2),
        'Andhra Pradesh': (5, 3),
        'Kerala': (6, 2),
        'Puducherry': (6, 3),
        'Tamil Nadu': (7, 3)
    }
    return provincesChina,provincesIndia

#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#                                                              Functions for AR6 comparison plots
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def plot_max_range(tq,q25,q50,q75,cols,cat,ax):
    #Max
    t9_l = tq[q25==max(q25)]
    t9_m = q50.columns[(q50.values[0]==max(q50.values[0]))]
    t9_h = tq[q75==max(q75)]

    t9_l = t9_l[0] if len(t9_l)>0 else 2106
    t9_m = t9_m[0] if len(t9_m)>0 else 2106
    t9_h = t9_h[0] if len(t9_h)>0 else 2106

    ax.bxp(
        [{'med': t9_m,'q1': t9_l,'q3': t9_h,
            'whislo': t9_l,'whishi': t9_h }],
        vert=False,patch_artist=True,boxprops=dict(facecolor=cols[cat], linewidth=0),
        medianprops=dict(color='k'),showfliers = False, showcaps = False, positions = [0.3], widths = [0.075]
    )


def plot_threshold_range(tq,q25,q50,q75,cols,cat,ax):
    #Thresholds
    for ind_box, q in enumerate([0.5,0.05]):
        t9_l = tq[(q25<=q*q25[tq==2020])&(tq>2020)]
        t9_m = q50.columns[(q50.values[0]<=q*q50.values[0][q50.columns==2020])&(q50.columns>2020)]
        t9_h = tq[(q75<=q*q75[tq==2020])&(tq>2020)]

        t9_l = t9_l[0] if len(t9_l)>0 else 2106
        t9_m = t9_m[0] if len(t9_m)>0 else 2106
        t9_h = t9_h[0] if len(t9_h)>0 else 2106

        ax.bxp(
            [{'med': t9_m,'q1': t9_l,'q3': t9_h,
                'whislo': t9_l,'whishi': t9_h }],
            vert=False,patch_artist=True,boxprops=dict(facecolor=cols[cat], linewidth=0),
            medianprops=dict(color='k'),showfliers = False, showcaps = False, positions = [[0.2,0.1][ind_box]], widths = [0.075]
        )


def get_ylim(ax,var,Region):
    y_maxs = {
        'Output|Coal': [375,130,50],
        'Secondary Energy|Electricity|Coal': [50,20,10],
        'Emissions|CO2|Energy and Industrial Processes': [7e4,2e4,5e3],
    }

    y_mins = {
        'Output|Coal': [-1,-1,-1],
        'Secondary Energy|Electricity|Coal': [-1,-1,-1],
        'Emissions|CO2|Energy and Industrial Processes':[-1e4,-0.5e4,-5e2],
    }

    y_mins = pd.DataFrame(y_mins,index=["World","CHN","IND"])
    y_maxs = pd.DataFrame(y_maxs,index=["World","CHN","IND"])

    if var in y_maxs.columns:
        ymin = y_mins.loc[Region,var]
        ymax = y_maxs.loc[Region,var]
    else:
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]

    return [ymin,ymax]


def add_coaloutput_comparisons(axs,regions_ar6,categories):
    
    # Loading data

    #Trajectories from IEA World Energy Balance
    Q_WEB = pd.read_csv('data\\CoalProductionEJ_WEB.csv')

    # Trajectories from SEI 2023. The Production Gap: Phasing down or phasing up? Top fossil fuel producers plan even more extraction despite climate promises. Stockholm Environment Institute. https://doi.org/10.51414/sei2023.050
    SEI = pd.read_csv('data\\SEI.csv')

    #Plotting aditional trajectories
    for i_r, reg in enumerate(regions_ar6):
        for i_c, cat in enumerate(categories):
            ax = axs[i_r][i_c]
            ax.plot(Q_WEB['t'], [float(x) for x in Q_WEB[['World','China','India'][i_r]]], color='k', alpha=1, linestyle='-', linewidth=1.2)

            if i_r != 1:
                ax.plot(SEI['t'][1:], [float(x) for x in SEI[['GPP','CHN_GPP','IND_GPP'][i_r]][1:]], color='red', alpha=1, linestyle='-', linewidth=1.2)


def add_emissions_comparisons(axs,regions_ar6,categories):
    # Trajectories van de Ven et al 2023. A multimodel analysis of post-Glasgow climate targets and feasibility challenges. Nat. Clim. Change 13, 570–578. https://doi.org/10.1038/s41558-023-01661-0
    VdV = pd.read_csv('data\\global_ite2_allmodels.csv')
    VdV.dropna(how = 'all',inplace=True, axis=0)
    VdV.dropna(how = 'all',inplace=True, axis=1)

    # Trajectories from Garg et al 2024. Synchronizing energy transitions toward possible Net Zero for India: Affordable and clean energy for all. Office of the Principle Scientific Advisor (PSA) to  Government of India and Nuclear Power Corporation of India Limited (NPCIL).
    Garg = pd.read_csv('data\\Garg_et_al_2024.csv')
    for i_r, reg in enumerate(regions_ar6):
        for i_c, cat in enumerate(categories):
            ax = axs[i_r, i_c]
            if i_c in [1,2]:

                Regionss = [
                    ['World','WORLD'],
                    ['China','CHN','CHI'],
                    ['India','IND'],  
                ][i_r]
                
                scenario = ['NDC_LTT','CP_EI'][i_c-1]
                ys = VdV[(VdV.Region.isin(Regionss))
                         & (VdV.Scenario==scenario)
                         & (VdV.Variable=='Emissions|CO2|Energy and Industrial Processes')]
                for y in ys.values:
                    ax.plot([float(x) for x in ys.columns[5:]], [float(x) for x in y[5:]], color='k', alpha=0.6, linewidth=0.5)
                    

            if (i_c == 1)&(i_r == 2):
                for ig, sc in enumerate(Garg.columns[1:]):
                    ax.plot(Garg['Year'], Garg[sc], color=([sns.color_palette()[1]]*3+[sns.color_palette('pastel')[1]]*4)[ig],  linewidth=0.5)


def plotting_with_AR6_range(var,var_imaclim,regions_ar6,categories,df,cols,Imaclim_data,T,compare_thresholds):
    """
    This function overlays own trajectory with those from the AR6 database (Byers et al 2022) for comparable emissions pathways
    The compare_thresholds boolean enables the function to plot the time-range at which pathways in the dataset reach certain thresholds and compare that with the pathways studied here
    For certain variables, additional reference scenarios can also be plotted
    """

    # Special case to convert unit incompatibility between AR6 results and Imaclim results
    exa2giga        =                 1e9 # G / E
    tep2gj          =              41.855 # GJ/tep
    mtoe2gj         =        1e6 * tep2gj # GJ/Mtep
    mtoe2ej         =  mtoe2gj / exa2giga # EJ/Mtep
    Imaclim_unit_convert = 1 if var!='Output|Coal' else mtoe2ej


    # Creating plot
    fig, axs = plt.subplots(len(regions_ar6),len(categories))

    if compare_thresholds:
        cax= [[0,0,0],[0,0,0],[0,0,0]]
        for i_r, reg in enumerate(regions_ar6):
            for i_c, cat in enumerate(categories):
                ax = axs[i_r, i_c]
                cax[i_r][i_c] = ax.inset_axes([0, -0.12, 1, 0.1])


    for i_r, reg in enumerate(regions_ar6):
        for i_c, cat in enumerate(categories):
            ax = axs[i_r, i_c]            
            ax.axhline(y=0, color='k', linewidth=0.75)
            ax.axvline(2020, color='k', linestyle='--')
            
            # Plotting AR6 data
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

            # Plotting Imaclim results
            Output = ['WO-15C-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS0','WO-NPi-ElecIndus-CCS0'][i_c]
            Region = ['World','CHN','IND'][i_r]

            y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output)&(Imaclim_data['Variables'] == var_imaclim)].values[0][5:]*Imaclim_unit_convert
            
            ax.plot(T, y, color='k',linewidth=0.75)
            ax.text(2098,0, f'n={n}', fontsize=8, ha='right')

            if compare_thresholds:
                ax2 = cax[i_r][i_c]
                plot_max_range(tq,q25,q50,q75,cols,cat,ax2)
                plot_threshold_range(tq,q25,q50,q75,cols,cat,ax)
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

                # Spetial formatting
                ax2.set_ylim([0, 0.4])
                ax2.set_xlim([2010,2105])
                ax2.set_yticks([])
                ax2.set_xticks([])

                        

            
            

            # Formatting the graph
            ax.set_title("")
            ax.set_ylim(get_ylim(ax,var,Region))
            ax.set_xlim([2010,2105])
            ax.legend().remove()
            

            unit = df.filter(variable=var, region=reg, Category=cat).unit[0]
            if i_r ==0:
                ax.set_title(['1.5°C','NDC LTT','NPI'][i_c]+' - '+cat)

            if (i_r==2):
                if compare_thresholds:
                    ax2.set_xticks(ax.get_xticks())
                    ax2.set_xlim([2010,2105])
                    ax2.set_ylim([0, 0.4])
                    ax2.set_xlabel('Year')
                
                    ax.set_xticks([])
                    ax.set_xlabel('')
                else:
                    ax.set_xlabel('Year')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            
            if i_c == 0:
                ax.set_ylabel(["World","China","India"][i_r]+'\n'+unit)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])


    # Adding extra reference trajectories if they are available for the given variable
    if var == 'Output|Coal':
        add_coaloutput_comparisons(axs,regions_ar6,categories) 
    if var == 'Emissions|CO2|Energy and Industrial Processes':
        add_emissions_comparisons(axs,regions_ar6,categories) 

    fig.suptitle(var)



def plot_energy_mix(Imaclim_data,Countries,Scenarios,Energy_colors,Variables,T):
    

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



def finding_emp_threshold_date(data,threshold,T):
    Q = data[(data.Variable=='Employment|Coal|Downscaled')].values[0][6:]
    T1=T[Q<Q[5]*(1-threshold)]

    if len(T1)>0:
        T1 = T1[0]
    else:
        T1 = 2100
    return T1



#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#                                                              Additional
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def pseudo_log(x, linthresh=5e-4):
    pl = np.sign(x)*np.log1p(np.abs(x))
    return pl

def color_not_plotted():
    return '#EAEAEA'


def calc_share_destruction(Result_data,region,T,t0,t1,Ls,scenario):

    
    L = Result_data[(Result_data['Downscaled Region'] == region)
                    & (Result_data['Variable'] == 'Employment|Coal|Downscaled') 
                    & (Result_data['Scenario'] == scenario)].values[0][6:]
    LF0 = float(
        Result_data[(Result_data['Downscaled Region'] == region)
                    & (Result_data['Variable'] == 'Labour Force|Downscaled')
                    & (Result_data['Scenario'] == scenario)].values[0][6])
    Ls.append(L[0])

    y = float((L[T == t0] - L[T == t1]) / LF0)
    return Ls, y

def calc_share_n_finding(Result_data,region,T,t0,t1,share_n_finding,scenario):

    D = np.array(
        Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
                    & (Result_data['Scenario'] == scenario) 
                    & (Result_data['Downscaled Region'] == region)].values.flat[6:][(T < t1) & (T > t0)])
    U = np.array(Result_data[
        (Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                & (Result_data['Scenario'] == scenario) 
                            & (Result_data['Downscaled Region'] == region)].values.flat[6:][(T < t1)
                                                        & (T > t0)])
    In = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                            & (Result_data['Scenario'] == scenario) &
                            (Result_data['Downscaled Region']
                            == region)].values.flat[6:][(T < t1)
                                                        & (T > t0)])

    share_n_finding=append_real_results(share_n_finding,D,U,In)

    L = Result_data[(Result_data['Downscaled Region'] == region)
                    & (Result_data['Variable'] == 'Employment|Coal|Downscaled') 
                    & (Result_data['Scenario'] == scenario)].values[0][6:]
    
    if L[T == t0] <= L[T == t1]: #Not plotting regions 
        share_n_finding[-1]=np.nan
    
    return share_n_finding

def calc_workforce(Result_data,region,Calced_data,scenario):

    LF = Result_data.loc[(Result_data['Downscaled Region']==region)&(Result_data.Scenario==scenario)&(Result_data.Variable=='Labour Force|Downscaled'),'2015'].values[0]
    CL = Result_data.loc[(Result_data['Downscaled Region']==region)&(Result_data.Scenario==scenario)&(Result_data.Variable=='Employment|Coal|Downscaled'),'2015'].values[0]
    
    Calced_data.append(CL/LF)

    return Calced_data
    
def monovariate_map(var,t0,T1,ax,cax,zlim,colormap,Result_data,scenario,Asia,norm=None):
    Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)
    T = range(2015, 2101)
    T = np.array(T)
    Ls = []
    share_n_finding = []
    Calced_data = []
    # Iterating over regions
    for region in list(Regions):

        if type(T1) is str:
            threshold = 0.8
            t1 =finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==region)&
                            (Result_data.Scenario==scenario)],threshold,T)

        else:
            t1=T1
       
        if var == "share_destruction":
            Ls, y = calc_share_destruction(Result_data,region,T,t0,t1,Ls,scenario)
            y = pseudo_log(y)
            Calced_data.append(y)
        elif var == "Finding_new_emp":
            Calced_data=calc_share_n_finding(Result_data,region,T,t0,t1,Calced_data,scenario)
            # Calced_data.append(share_n_finding)
        elif var== 'Workforce':
            Calced_data = calc_workforce(Result_data,region,Calced_data,scenario) 
        else:
            # return 
            print('unknown')

    Calced_data = pd.DataFrame(data=np.array([list(Regions), Calced_data]).transpose(),
                        columns=['Region_Nam', 'Calced_data'])

    Asia_Data = Asia.merge(Calced_data, on='Region_Nam')

    Asia_Data['Calced_data'] = pd.to_numeric(Asia_Data['Calced_data'], errors='coerce')
    
    
    cbar = Asia_Data.plot(column='Calced_data',
                        cmap=colormap,
                        legend=True,
                        ax=ax,
                        edgecolor='black',
                        missing_kwds={
                            "color": color_not_plotted(),
                            "label": "Missing values",
                        },
                        vmin=zlim[0],
                        vmax=zlim[1],
                        linewidth=0.75,
                        cax=cax,
                        rasterized=True,
                        norm=norm)

    Asia[Asia['region']=='Asia'].plot(ax=ax, color=color_not_plotted(), edgecolor='black',linewidth=0.5,rasterized=True)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color=color_not_plotted(), edgecolor='black', linestyle='--',linewidth=0.5,rasterized=True)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([65,140])
    ax.set_ylim([7,55])

    return ax





#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#                                                              Additional
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def halve_axes(fig,ax):
    pos = ax.get_position()

    # Calculate the new height (half of the original axis height)
    new_height = pos.height / 2

    # Create the two new axes, one on top of the other
    ax1 = fig.add_axes([pos.x0, pos.y0 + new_height, pos.width, new_height])  # Top half
    ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, new_height])  # Bottom half

    # Optionally, remove the original axis if you no longer need it
    ax.remove()

    return ax1,ax2

def Grid_Unemployment_Destruction(fig,axs,T,Result_data,ind_country):
    [ax.axis('off') for ax in axs.flatten()]
    regions = defining_province_grid()[ind_country]

    Scenarios = ['NPI','NDC','NZ']

    t0 = 2020
    t1 = 2050

    for region, position in regions.items():
        ax = axs[position]
        ax1, ax2 = halve_axes(fig,ax)
        
        variable = 'Unemployment|Downscaled'

        for ind_scenario, scenario in enumerate(Scenarios):
            ax1.plot(
                T[(t0<=T)&(T<=t1)],
                Result_data[
                    (Result_data['Downscaled Region']==region)&
                    (Result_data.Scenario==scenario)&
                    (Result_data.Variable==variable)
                ].values[0][6:][(t0<=T)&(T<=t1)],
                color=defining_waysout_colour_scheme()[scenario]
            )

        
        
        variable = 'Employment|Coal|Downscaled'
        for ind_scenario, scenario in enumerate(Scenarios):
            
            y = Result_data[
                    (Result_data['Downscaled Region']==region)&
                    (Result_data.Scenario==scenario)&
                    (Result_data.Variable==variable)
                ].apply(pd.to_numeric,errors='coerce')
            
            
            if y.sum(axis=1).values[0]==0:
                ax2.set_facecolor('lightgrey')
            else:
                ax2.plot(
                    T[(t0<=T)&(T<=t1)],
                    -y.diff(axis=1).rolling(axis=1,window=4,center=True).mean().values[0][6:][(t0<=T)&(T<=t1)],
                    color=defining_waysout_colour_scheme()[scenario]
                )
        
        ax1.set_xticks([])
        ax2.text(0.05,0.05,region,transform=ax.transAxes, fontsize= 11, fontweight='bold')
    return fig



def grid_scale_treatment(region,ax,ax2,ind_country):
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
    return region, ax, ax2



def Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,U):
    grid_scale_same = True
    Step = 1
    [ax.axis('off') for ax in axs.flatten()]
    axs[0,0].text(0.05,0.5,['a)','b)'][ind_country],transform=axs[0,0].transAxes, fontsize= 40, fontweight='bold')
    regions = defining_province_grid()[ind_country]

    Scenarios = ['NZ','NDC','NPI','NZ_CCS1','NDC_CCS1']

    t0 = 2020
    t1 = 2060

    for region, position in regions.items():

        ax = axs[position]
        ax.axis('on')
        if U:
            ax1, ax2 = halve_axes(fig,ax)
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
                    color=defining_waysout_colour_scheme()[scenario]
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
                    color=defining_waysout_colour_scheme()[scenario.split('_PG0')[0]],
                    linewidth=3,
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
                    color=defining_waysout_colour_scheme()[scenario.split('_')[0]],
                    linestyle = (0,(1,3)),
                    linewidth=3
                )

                if grid_scale_same:
                    grid_scale_treatment(region, ax, ax2, ind_country)


        
        ax2.text(0.05,0.05,region,transform=ax.transAxes, fontsize= 17, fontweight='bold')
        
    
    # Legend in bottom right corner
    ax = axs[-1,[2,-2][ind_country]]
    alines = []
    for ind_scenario, scenario in enumerate(Scenarios[:5]):
        alines.append(ax.plot([],[],linewidth=3,color=defining_waysout_colour_scheme()[scenario.split('_PG0')[0]], label=['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS'][ind_scenario]))
    alines.append(ax.plot([],[], color='k', linestyle=':',label='No productivity growth'))
    ax.legend(fontsize=25,ncol=2)
    
    return fig


#========================================================================================================================================
# Defining functions to use nonlinear colormap in monovariate exposure graphs

def _forward(x):
    thresh_neg = -0.00025  # Threshold for negative values
    thresh_pos = 0.002    # Threshold for positive values

    # Apply symlog transformation
    y = np.where(
        x >= 0,
        np.log(1 + x / thresh_pos),
        -np.log(1 + x / thresh_neg)
    )
    return y

# Define the inverse transformation
def _inverse(y):
    thresh_neg = -0.00025  # Threshold for negative values
    thresh_pos = 0.002    # Threshold for positive values

    # Apply inverse symlog transformation
    x = np.where(
        y >= 0,
        thresh_pos * (np.exp(y) - 1),
        thresh_neg * (np.exp(-y) - 1)
    )
    return x


def semisymlognorm():
    # Define a custom colormap to differentiate positive and negative values
    vmin = -0.005
    vmax = 0.05  
    vcenter = 0.0 
    stretch = 0.1
    zlim = [vmin,vmax]
    colormap = plt.cm.PiYG_r
    norm = mcolors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
    return zlim, norm, colormap


#===

def def_region_indices():
    region_indices = pd.read_csv('..\coal.labour.nexus\data\Coal_labour\Downscaling\Indexes.csv')
    return region_indices


def destination_share(Result_data,country,scenario,t0,T1):
    '''
    This functions aims to calculate share of job destruction leaving into retirement and share of layoffs leaving into retirement for given regions
    It is designed specially for use in plotting boxplot of state-provincial-wise results
    '''
    destination_variables = [x for x in Result_data.Variable.unique() if 'Coal Worker Destination' in x if 'Retire' not in x and 'Hire' not in x]
    exclude_downscaled_regions = ['China','India']
    
    Destination = Result_data[(Result_data.Variable.isin(destination_variables))&
                            (Result_data.Region==country)&
                            (~Result_data['Downscaled Region'].isin(exclude_downscaled_regions))&
                            (Result_data.Scenario==scenario)]
    U = Destination[Destination.Variable=='Coal Worker Destination|Unemployment'].drop(['Model','Region','Scenario','Variable','Unit'],axis=1).groupby('Downscaled Region').sum().loc[:,[str(x) for x in range(2020,T1)]].sum(axis=1)
    TotDestination = Destination.drop(['Model','Region','Scenario','Variable','Unit'],axis=1).groupby('Downscaled Region').sum().loc[:,[str(x) for x in range(t0,T1)]].sum(axis=1)
    TotDestination[TotDestination==0]=np.nan
    Share_U = U/TotDestination
    Share_U=Share_U.dropna()

    R = Result_data[(Result_data.Variable=='Coal Worker Destination|Retire')&
                (Result_data.Region==country)&
                (~Result_data['Downscaled Region'].isin(exclude_downscaled_regions))&
                (Result_data.Scenario==scenario)].drop(['Model','Region','Scenario','Variable','Unit'],axis=1).groupby('Downscaled Region').sum().loc[:,[str(x) for x in range(2020,T1)]].sum(axis=1)

    Share_R = R/(R+TotDestination)
    Share_R=Share_R.dropna()
    return Share_U, Share_R, TotDestination