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
def nnf(a):
    return np.array(a)


#    Stacked bars function
def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

    
# Functions used for bivariate plot 
def weighted_average(df, value, weight, country):
    return (df[df['Region'] == country][value] * df[df['Region'] == country][weight]).sum() / df[df['Region'] == country][weight].sum()


def interpol(x, xlim, ylim):
    y = ylim[0] + (ylim[1]-ylim[0])/(xlim[1]-xlim[0])*(x-xlim[0])
    return y


def plot_bivariate(scenario, ax, Regions, Result_data, cmap, b_xlim, b_ylim, T, t0, t1, Asia, s_index, key_data, Scenarios_names):
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

    return key_data