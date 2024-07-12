# Importing libraries
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import date
from pyproj import CRS
import os

# Plotting parameters
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Point
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import xycmap
from matplotlib.colors import PowerNorm
from matplotlib import ticker
import seaborn as sns
#=========================================================================================================
# Defining the colors for the different scenarios centrally


def defining_waysout_colour_scheme():
    Colors = {'NPI':sns.color_palette()[3],
            'NDC':sns.color_palette()[2],
            'NZ':sns.color_palette()[0],
            'WO-NPi-ElecIndus':sns.color_palette()[3],
            'WO-NDCLTT-ElecIndus':sns.color_palette()[2],
            'WO-15C-ElecIndus':sns.color_palette()[0],
            'NPI_gem':sns.color_palette('pastel')[3],
            'NDC_gem':sns.color_palette('pastel')[2],
            'NZ_gem':sns.color_palette('pastel')[0]}
    return Colors


#=========================================================================================================
#    Stacked bars function
def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

    
# Functions used for bivariate plot 
def weighted_average(df, value, weight, country):
    """
    The weighted average function is used to when mapping the vulnerability of regions to the coal transition
    It allows to calculate the average of a variable in a country, using a weight such as the workforce
    """
    return (df[df['Region'] == country][value] * df[df['Region'] == country][weight]).sum() / df[df['Region'] == country][weight].sum()


def interpol(x, xlim, ylim):
    y = ylim[0] + (ylim[1]-ylim[0])/(xlim[1]-xlim[0])*(x-xlim[0])
    return y


#=========================================================================================================
def plot_vulnerability_bivariate(scenario, ax, Regions, Result_data, T, t0, t1, Asia, s_index, key_data, Scenarios_names):
    """
    This function maps the exposure (ratio of coal job destruction to labour force)
    and vulnerability (share of laid off workers going to unemployment) 
    of regions of China and India to the coal transition
    """
    # Defining the colormap
    b_xlim = [0.3,0.85]
    b_ylim = [np.log(3e-4),np.log(0.065)]


    n = (10, 10)  # x, y

    corner_colors = ("#e8e8e8",   "#C85a5a", "#64acbe", "#574249")
    cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
    cmap_zip = [cmap, b_xlim, b_ylim, n]
    
    
    # Initializing the lists
    share_n_finding = []
    cntry = []
    Dc=[]
    DI=[]
    share_destruction =[]
    destruction = []
    Ls = []

    # Iterating over regions
    for region in list(Regions):

        L = Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Employment|Coal|Downscaled') &
                        (Result_data['Scenario'] == scenario)].values[0][6:]
        LF0 = float(
            Result_data[(Result_data['Downscaled Region'] == region)
                        & (Result_data['Variable'] == 'Labour Force|Downscaled')
                        & (Result_data['Scenario'] == scenario)].values[0][6])
        Ls.append(L[0])
        y = float((L[T == 2020] - L[T == 2035]) / LF0)
        if y == 0:
            y = np.nan
        else:
            y = np.log(y)
        share_destruction.append(y)
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
        I = np.array(Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                                 & (Result_data['Scenario'] == scenario) &
                                 (Result_data['Downscaled Region']
                                  == region)].values.flat[6:][(T < t1)
                                                              & (T > t0)])

        if sum((U + D + I)) != 0:
            share_n_finding.append(1-sum(D+I) / sum((U + D + I)))
        else:
            share_n_finding.append(np.nan)

        cntry.append(Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0])
        if Result_data[Result_data['Downscaled Region'] == region]['Region'].values[0] == 'CHN':	
            Dc.append(share_n_finding[-1])
        else:
            DI.append(share_n_finding[-1])
    
       

    Regions = list(Regions)
    share_n_finding = pd.DataFrame(data=np.array([list(Regions), share_n_finding,cntry,share_destruction,destruction,Ls]).transpose(),
                           columns=['Region_Nam', 'share_n_finding','Region','share_destruction','destruction','Workforce'])

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
    
    Asia_Data_with_colors = Asia_Data.merge(cmapi, on='Region_Nam', how='left').fillna('lightgrey')
    Asia_Data_with_colors.plot(ax=ax, color=Asia_Data_with_colors['colors'],
        edgecolor='black',linewidth=0.5,rasterized=True,alpha=1)

    Asia[Asia['region']=='Asia'].plot(ax=ax, color='whitesmoke', edgecolor='black',linewidth=0.5)
    Asia[Asia['region']=='Disputed'].plot(ax=ax, color="whitesmoke", edgecolor='black', linestyle='--',linewidth=0.5)
    
    for region in ['Shanxi','Inner Mongolia','Jharkhand','Odisha','Chhattisgarh']:
        key_data[region+str(s_index)] = {'Downscaled Region':region,
                                         'Scenario':scenario,
                                        'coordinates':[float(share_n_finding[share_n_finding['Region_Nam']==region]['share_n_finding'].values[0]),
                                    float(share_n_finding[share_n_finding['Region_Nam']==region]['share_destruction'].values[0])],
                                    'destruction':float(share_n_finding[share_n_finding['Region_Nam']==region]['destruction'].values[0])}

    for region in ['China','India']:  
        key_data[region+str(s_index)] = {'Downscaled Region':'Average \n'+region,
                                         'Scenario':scenario,
                                        'coordinates':[weighted_average(share_n_finding, 'share_n_finding','Workforce', ['IND' if region =='India' else 'CHN'][0]),
                                                       weighted_average(share_n_finding, 'share_destruction','Workforce', ['IND' if region =='India' else 'CHN'][0])],
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

        R= sum(Result_data[
            (Result_data['Variable'] == 'Coal Worker Destination|Retire')
            & (Result_data['Scenario'] == Scenario) &
            (Result_data['Downscaled Region']
                == province)].values.flat[6:][(T < t1) & (T > t0)])
        D= sum(Result_data[
            (Result_data['Variable'] == 'Coal Worker Destination|Instant Match')
            & (Result_data['Scenario'] == Scenario) &
            (Result_data['Downscaled Region']
                == province)].values.flat[6:][(T < t1) & (T > t0)])
        I= sum(
            Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Delayed Match')
                        & (Result_data['Scenario'] == Scenario) &
                        (Result_data['Downscaled Region']
                            == province)].values.flat[6:][(T < t1)
                                                        & (T > t0)])
        U= sum(
            Result_data[(Result_data['Variable'] == 'Coal Worker Destination|Unemployment')
                        & (Result_data['Scenario'] == Scenario) &
                        (Result_data['Downscaled Region']
                            == province)].values.flat[6:][(T < t1)
                                                        & (T > t0)])
        H= sum(-Result_data[
            (Result_data['Variable'] == 'Coal Worker Destination|Hire')
            & (Result_data['Scenario'] == Scenario)
            & (Result_data['Downscaled Region'] == province)].values.flat[6:][
                (T < t1) & (T > t0)])
        
        data_save[(province, Scenario,t1)] = np.array([R, D, I, U, H]) / 1e6
        Retirement += R
        Direct += D
        Indirect += I
        Unemployed += U
        Hire += H

    data = np.array([Retirement, Direct, Indirect, Unemployed, Hire
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