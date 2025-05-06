# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xycmap
import seaborn as sns
import os
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
import re
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import string
import pyam

Region_indexes = pd.read_csv('../coal.labour.nexus/data/Coal_labour/Downscaling/Indexes.csv')



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


def plot_national_employment_trajectories(T,Result_data,Historical_data,Step=5,Show_alternatives=False,Show_supply=False,Show_uncertainty=False):
    Colors = defining_waysout_colour_scheme()
    Countries = ['China','India']
    Regions = [
        Result_data[(Result_data['Region'] == ['CHN','IND'][x])&(~Result_data['Downscaled Region'].isin(['China', 'India']))]['Downscaled Region'].unique()
        for x in [0, 1]
    ]
    Scen_type = ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']

    if Show_alternatives:
        Alt_type = [
        '','_PG0','_C0I0_P0','_C0I0_P100','_C20I5_P0','_C40I10_P100']
        Ralpha = [1, 0.75, 1,1,1,1] * 5
        Rlinestyle = ['-',':','--','--','--','--']*5
        Rlinewidth = [1, 0.75, 0.75, 0.75, 0.75, 0.75]* 5
        Rmarker = ['', '', '^', 'o', 'x','s'] * 5
        Scen_list = [0,6,12,18,24]
    elif Show_supply:
        Alt_type = [
        '','_tra','_PG0']
        Ralpha = [1,1,1] * 5
        Rlinestyle = ['-','--',':']*5
        Rlinewidth = [1,0.75,0.75]* 5
        Rmarker = ['','',''] * 5
        Scen_list = [0,3,6,9,12]
    else:
        Alt_type = [
        '','_PG0']
        Ralpha = [1,1] * 5
        Rlinestyle = ['-',':']*5
        Rlinewidth = [1,0.75]* 5
        Rmarker = ['',''] * 5
        Scen_list = [4,2,0,8,6]
    Scenarios = [x + y for x in Scen_type for y in Alt_type]

    Variable = "Employment|Coal|Downscaled"
    fig, axs = plt.subplots(1, 2, figsize=standard_figure_size())
    for c_index, country in enumerate(Countries):
        
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

            if (Scenarios[j] in  ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']) & ~(('_CCS1' in Scenarios[j])&(country=='China')):
                if COAL_emp.values[0][6:][
                        T < 2070][-1] < COAL_emp.values[0][6:][5] / 2:
                    
                    ax.scatter(T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] /2][0],
                            -0.125,
                            color=Colors[scenario.split('_PG0')[0]],
                            marker='o',
                            s=20)
                    
                    if COAL_emp.values[0][6:][
                        T < 2070][-1] < COAL_emp.values[0][6:][5] *0.05:
                        t95 = T[COAL_emp.values[0][6:] < COAL_emp.values[0][6:][5] *0.05][0]
                        ax.scatter(t95,
                                -0.125,
                                color=Colors[scenario.split('_PG0')[0]],
                                marker='^',
                                s=20)
                        
                        # Outputing some results for text
                        if Scenarios[j] in ['NPI','NDC','NZ']:
                            print(f'In {country} under {Scenarios[j]}, 95% of jobs disappear by {t95} \n')
                        

        Scen_y = pd.DataFrame(Scen_y, index=Scenarios)
        S_maxy = pd.DataFrame(S_maxy, index=Scen_type)
        S_miny = pd.DataFrame(S_miny, index=Scen_type)

        for Scen_type_ind,scenario in enumerate(Scenarios):
            if not ((country=="China") & ('_CCS1' in scenario)): #Not plotting CCS scenarios in China for clarity (overlap with non CCS scenario)
                ax.plot(T[T < 2070][0:-1:Step],
                        Scen_y.loc[scenario].values[T < 2070][0:-1:Step],
                        color=Colors[split_by_any_substring(scenario, Alt_type[1:])],
                        linestyle=Rlinestyle[Scen_type_ind],
                        linewidth=2.2*Rlinewidth[Scen_type_ind],
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
        if Show_alternatives:
            for ind_scen_type, scen_type in enumerate(Scen_type):
                if not ((country=="China") & ('_CCS1' in scen_type)): #Not plotting CCS scenarios in China for clarity (overlap with non CCS scenario)
                
                    all_scenario = [scen_type+x for x in ['','_C0I0_P0','_C0I0_P100','_C20I5_P0','_C40I10_P100']]
                    y2030 = Scen_y.loc[all_scenario,15].values
                    y2050 = Scen_y.loc[all_scenario,35].values

                    ax.add_patch(Rectangle((2070+2*ind_scen_type, min(y2030)),0.75, max(y2030)-min(y2030),
                                        facecolor= Colors[scen_type], edgecolor='none',alpha=0.55))
                    ax.add_patch(Rectangle((2071+2*ind_scen_type, min(y2050)),0.75, max(y2050)-min(y2050),
                                        facecolor= Colors[scen_type], edgecolor='none',alpha=0.55))
                    ax.plot([2070+2*ind_scen_type,2070.75+2*ind_scen_type],[y2030[0],y2030[0]],color=Colors[scen_type])
                    ax.plot([2071+2*ind_scen_type,2071.75+2*ind_scen_type],[y2050[0],y2050[0]],color=Colors[scen_type])
                    if ind_scen_type==0:
                        ax.text(2070-0.5+2*ind_scen_type, max(y2030)+0.1, '2030', fontsize=5, color='k',rotation=45,ha='left',va='bottom')
                        ax.text(2071+1+2*ind_scen_type, min(y2050)+0.1, '2050', fontsize=5, color='k',rotation=45,ha='right',va='top')
                        
        # Formatting axes
        ax.set_title(country)
        ax.set_ylabel('Million workers')
        ax.set_ylim([-0.25, 5])
        ax.axvline(x=2020, color='k', linestyle='--', linewidth=0.8)


        # Outputing some results for text
        #2020-2030 Job cuts
        jc = (Scen_y.loc[['NPI','NDC','NZ'],5]-Scen_y.loc[['NPI','NDC','NZ'],15])*1e-1*1e3
        print(f'In {country}, {jc} thousand jobs are cut annually between 2020 and 2030')

        #Peak destruction by productivity
        for scenario in ['NDC','NZ']:
            dif = (Scen_y.loc[scenario+'_PG0']-Scen_y.loc[scenario])*1e3
            tdif = T[dif==max(dif)][0]
            print(f'In {country} under {scenario}, {max(dif)} jobs ({0.1*max(dif)/Scen_y.loc[scenario,5]:0.1f}%) are destroyed by productivity in {tdif}')

        #Destruction by productivity in NPI 2070
        dif = (Scen_y.loc['NPI_PG0',55]-Scen_y.loc['NPI',55])*1e3
        print(f'In {country} under NPI, {dif} jobs are destroyed by productivity')


    # LEGENDS)
    alines = []
    for ind, Scen_type_ind in enumerate(Scen_list):
        scenario = Scenarios[Scen_type_ind]
        alines.append(axs[0].plot([], [],
                                color=Colors[scenario.split('_PG0')[0]],
                                label=['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS'][ind],
                    linewidth=2.2*Rlinewidth[Scen_type_ind],
                    alpha=Ralpha[Scen_type_ind] )[0])
        
    alines.append(axs[0].plot([],[],color='k',linestyle='-',label='Productivity growth')[0])
    alines.append(axs[0].plot([],[],color='k',linestyle=':',label='No productivity growth')[0])
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

    # additional legend for alternatives
    if Show_alternatives:

        alines.append(axs[0].plot([], [],
                                color='k',
                                label='C20I5_P100')[0])
        alines.append(axs[0].plot([], [],
                                color='k', marker = '^',
                                label='C0I0_P0')[0])
        alines.append(axs[0].plot([], [],
                                color='k', marker = 'o',
                                label='C0I0_P100')[0])
        alines.append(axs[0].plot([], [],
                                color='k', marker = 'x',
                                label='C20I5_P0')[0])
        alines.append(axs[0].plot([], [],
                                color='k', marker = 's',
                                label='C40I10_P100')[0])
    # additional legend for supply-demand uncertainty
    if Show_uncertainty:
        if ~Show_alternatives:
            alines.append(axs[0].plot([], [],
                                    color='k',
                                    label='Central estimate')[0])
        alines.append(axs[0].fill_between([], [], [],
                                color='darkgrey',
                                alpha = 0.5,
                                label='Calibration uncertainty'))

    [ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)'])]

    #plot legend
    labels = [la.get_label() for la in alines]
    
    fig.legend(handles=alines,
            labels=labels,
            loc='lower center',
            ncol=4+Show_alternatives+Show_uncertainty,
            bbox_to_anchor=(0.5, -0.18),
            frameon=False)
    return fig


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


def destination_colors():

    colors = {
        'Retirement':sns.color_palette("pastel")[1],
        'Instantaneous matches':sns.color_palette("pastel")[2],
        'Delayed matches':sns.color_palette("pastel")[4],
        'Unemployed':sns.color_palette("pastel")[3],
        'Hire':sns.color_palette("pastel")[6],
    }

    return colors

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
    clrs = [destination_colors()[x.replace('\n','')] for x in labelz]
    for i in np.arange(0, data_shape[0]):
        alines.append([
            ax.bar(X[s_index],
                    data[i],
                    bottom=data_stack[i],
                    width=0.8,
                    # alpha=0.6,
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
    non_Producing = False
    if non_Producing:
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
    else:
        provincesChina = {
        'Heilongjiang': (0, 4),
        'Xinjiang': (0, 0),
        'Qinghai': (1, 0),
        'Ningxia': (1, 1),
        'Inner Mongolia': (1, 2),
        'Liaoning': (1, 4),
        'Jilin': (0, 3),
        'Gansu': (2, 0),
        'Shaanxi': (2, 1),
        'Shanxi': (2, 2),
        'Hebei': (1, 3),
        'Sichuan': (3, 0),
        'Henan': (2, 3),
        'Shandong': (2, 4),
        'Yunnan': (4, 0),
        'Guizhou': (3, 1),
        'Hunan': (3, 2),
        'Anhui': (3, 3),
        'Jiangsu': (3, 4),
        'Guangxi': (4, 1),
        'Jiangxi': (4, 2),
        'Fujian': (4, 3)
    }

    provincesIndia = {
        'Rajasthan': (0, 0),
        'Uttar Pradesh': (1, 1),
        'Assam': (0, 4),
        'Gujarat': (1, 0),
        'Madhya Pradesh': (2, 0),
        'Chhattisgarh': (1, 2),
        'Jharkhand': (1, 3),
        'Maharashtra': (3, 0),
        'Telangana': (2, 1),
        'Odisha': (2, 3),
        'West Bengal': (1, 4),
        'Tamil Nadu': (3, 2)
    }
    provincesIndia = {
        'Rajasthan': (0, 0),
        'Uttar Pradesh': (0, 1),
        'Assam': (0, 4),
        'Gujarat': (1, 0),
        'Madhya Pradesh': (1, 1),
        'Chhattisgarh': (1, 2),
        'Jharkhand': (1, 3),
        'Maharashtra': (2, 0),
        'Telangana': (2, 1),
        'Odisha': (2, 3),
        'West Bengal': (1, 4),
        'Tamil Nadu': (2, 2)
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
            if i_c <= 1:
                y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output.replace("CCS0",'CCS1'))&(Imaclim_data['Variables'] == var_imaclim)].values[0][5:]*Imaclim_unit_convert
            
                ax.plot(T, y, color='k',linestyle='--',linewidth=0.75)

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
            if var == "Capacity|Electricity|Coal":
                ax.set_ylim([0,[3000,1500,500][i_r]])


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



def grid_scale_treatment(region,ax,ax2,ind_country,remove_splin=False):
    if region == 'Shanxi':
        ax2.set_ylim([-20,900])
        if not remove_splin:
            ax2.spines['left'].set_color('red') 
            ax.tick_params(axis='y', colors='red')
    elif region =='Jharkhand':
        ax2.set_ylim([-7,375])
        if not remove_splin:
            ax2.spines['left'].set_color('red') 
            ax.tick_params(axis='y', colors='red')
    else:
        ax2.set_ylim([[-10,350],[-7,200]][ind_country])
    return region, ax, ax2



def regional_grid(representation):
    if representation<2:
        provincesChina, provincesIndia = defining_province_grid()
    else:
        provincesChina = {
            'Shanxi':(0,0),
            'Inner Mongolia':(0,2),
            # 'Shaanxi':(1,0),
            'Shandong':(0,1),
        }
        provincesIndia = {
            # 'Chhattisgarh':(0,0),
            'Jharkhand':(0,0),
            'Odisha':(0,1),
            'West Bengal':(0,2),
        }
    grid_size = [[[8,6],[8,8]],
                 [[5,5],[3,5]],
                 [[1,3],[1,3]]][representation]
    return provincesChina,provincesIndia, grid_size


def Grid_Employment_Destruction(fig,axs,T,Result_data,ind_country,U,Show_alternatives=False,grid_scale_same = True,representation=1,remove_splin=False):
    fig_width = fig.get_figwidth()
    fig_scale = fig_width/26
    Step = 5
    [ax.axis('off') for ax in axs.flatten()]
    if not Show_alternatives:
        axs[0,0].text(0.05,0.8,['a)','b)'][ind_country],transform=axs[0,0].transAxes, fontsize= 40*fig_scale, fontweight='bold')
    regions = regional_grid(representation)[ind_country]
    grid_size = regional_grid(representation)[2][ind_country]
    Scenarios = ['NZ_CCS1','NDC_CCS1','NZ','NDC','NPI']

    t0 = 2020
    t1 = 2060

    # Remove splin
    if remove_splin:
        [ax.axhline(y=100,color='k',linewidth=0.25,zorder=-1) for ax in axs.flatten()]
        [ax.spines['top'].set_visible(False) for ax in axs.flatten()]
        [ax.spines['right'].set_visible(False) for ax in axs.flatten()]
        axs.flatten()[-1].spines['left'].set_visible(False)
        axs.flatten()[-1].tick_params(left=False,  labelleft=False)
    

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
                    linewidth=6*fig_scale,
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
                    color=defining_waysout_colour_scheme()[scenario],
                    linestyle = ':',
                    linewidth=6*fig_scale
                )

                if Show_alternatives: # If show_Alternative is True, we also plot the alternative demand-driven scenario
                    y = Result_data[
                        (Result_data['Downscaled Region']==region)&
                        (Result_data.Scenario==scenario+'_tra')&
                        (Result_data.Variable==variable)
                    ].apply(pd.to_numeric,errors='coerce')*1e-3

                    ax2.plot(
                        T[(t0<=T)&(T<=t1)][0:-1:Step],
                        y.values[0][6:][(t0<=T)&(T<=t1)][0:-1:Step],
                        color=defining_waysout_colour_scheme()[scenario.split('_')[0]],
                        linestyle = '--',
                        linewidth=6*fig_scale
                    )

                if grid_scale_same:
                    grid_scale_treatment(region, ax, ax2, ind_country, remove_splin)

                
        
        ax2.text(0.05,0.05,region,transform=ax.transAxes, fontsize= 11, fontweight='bold')
    
    if representation == 2:
        [ax.set_ylabel('Thousand workers') for ax in axs[:,0]]
    
    # Legend in bottom right corner
    if ((ind_country == 1) | (Show_alternatives)):
        print('Adding legend')
        ax = axs[-1,grid_size[1]//2]#[2,-2][ind_country]]
        alines = []
        for ind_scenario, scenario in enumerate(Scenarios[:5]):
            alines.append(ax.plot([],[],linewidth=10*fig_scale,color=defining_waysout_colour_scheme()[scenario.split('_PG0')[0]], label=['1.5°C-CCS','NDC-LTT-CCS','1.5°C','NDC-LTT','NPi'][ind_scenario]))
        alines.append(ax.plot([],[], color='k',linewidth=10*fig_scale, linestyle=':',label='No productivity growth'))
        if Show_alternatives:
            alines.append(ax.plot([],[], color='k',linewidth=10*fig_scale, linestyle='--',label='Demand-driven scenario'))
        
        ax.legend(fontsize=25*fig_scale,ncol=3,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25))
        

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


def double_axis(ax, region):

    
    JEJ={
        'World':174.09/8821.3,
        'CHN':91.32/4558.6,
        'IND':15.05/910.8,
        'USA':12.07/539,
        'AFR':6.15/257.2,
        'EUR':5.92/552.6
    }

    ax3 = ax.twinx()
    ax3.set_ylim([lim/JEJ[region] for lim in ax.get_ylim()])
    

    return ax,ax3

def plot_AR6_range_production(var,var_imaclim,regions_ar6,categories,df,cols,Imaclim_data,T,compare_thresholds):
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
    if var_imaclim == 'Output|Coal':
        JEJ_w = 174.09/8821.3
        JEJ_c = 91.32/4558.6
        JEJ_i = 15.05/910.8


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
            # (
            #     df.filter(variable=var, region=reg, Category=cat).plot.line(
            #         color="Category", ax=ax,
            #         alpha=0, fill_between=True 
            #     )
            # )

            tq = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().columns
            q25 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[0]
            q75 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.25,0.75]).timeseries().values[1]
            q50 = df.filter(variable=var, region=reg, Category=cat).compute.quantiles([0.5]).timeseries()

            filter=(~np.isnan(q75))&(~np.isnan(q25))
            tq = tq[filter]
            q25 = q25[filter]
            q75 = q75[filter]


            # ax.fill_between(tq, q25, q75, color=cols[cat], alpha=0.5)
            df.filter(variable=var, region=reg, Category=cat).plot.line(
                    color="Category", ax=ax,
                    alpha=0.5,linewidth=0.2, fill_between=False, rasterized = True 
                )
            ax.plot(q50.columns, q50.values[0], color=cols[cat])

            n= len(df.filter(variable=var, region=reg, Category=cat)['scenario'])
            ax.text(2098,0, f'n={n}', fontsize=8, ha='right')

            # Plotting Imaclim results
            Region = ['World','CHN','IND'][i_r]
            Outputs = [['WO-15C-ElecIndus-CCS0','WO-15C-ElecIndus-CCS1'],['WO-NDCLTT-ElecIndus-CCS0','WO-NDCLTT-ElecIndus-CCS1'],['WO-NPi-ElecIndus-CCS0']][i_c]
            for Output in Outputs:
                y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output)&(Imaclim_data['Variables'] == var_imaclim)].values[0][5:]*Imaclim_unit_convert
                
                ax.plot(T[::5], y[::5], color=defining_waysout_colour_scheme()[Output],linewidth=1.5)

            if compare_thresholds:
                ax2 = cax[i_r][i_c]
                plot_max_range(tq,q25,q50,q75,cols,cat,ax2)
                plot_threshold_range(tq,q25,q50,q75,cols,cat,ax2)

                for Output in Outputs:
                    y = Imaclim_data[(Imaclim_data['Region'] == Region)&(Imaclim_data['Scenario'] == Output)&(Imaclim_data['Variables'] == var_imaclim)].values[0][5:]*Imaclim_unit_convert
                    
                    # Max
                    t_m = T[y==max(y)]
                    t_m = t_m[0] if len(t_m)>0 else 2106
                    ax2.scatter(t_m, 0.3, color=defining_waysout_colour_scheme()[Output], marker='x', s=6, zorder=5)
                    [cax[i_r][ni_c].scatter(t_m, 0.3, color='k', alpha=0.3, marker='x', s=6, zorder=5) for ni_c in range(3) if ni_c != i_c]

                    # Threshold
                    for ind_box, q in enumerate([0.5,0.05]):
                        t_m = T[y<=q*y[T==2020]]
                        t_m = t_m[0] if len(t_m)>0 else 2106
                        ax2.scatter(t_m, [0.2,0.1][ind_box], color=defining_waysout_colour_scheme()[Output], marker=['o','^'][ind_box], s=6, zorder=5)
                        [cax[i_r][ni_c].scatter(t_m, [0.2,0.1][ind_box], color='k', alpha=0.3, marker=['o','^'][ind_box], s=6, zorder=5) for ni_c in range(3) if ni_c != i_c]



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

            ax, ax3 = double_axis(ax, Region)
            if i_c != len(categories)-1:
                ax3.set_yticklabels([])
            else:
                ax3.set_ylabel('Mt') 


    # Adding extra reference trajectories if they are available for the given variable
    add_coaloutput_comparisons(axs,regions_ar6,categories) 


    # fig.suptitle(var)
    return fig


def split_by_any_substring(string, substrings):
    # Create a regex pattern that matches any of the substrings
    pattern = '|'.join(map(re.escape, substrings))

    # Split the string by the pattern
    split_list = re.split(pattern, string)

    # Return the first element of the split list
    return split_list[0]


def find_destination_data(T,Result_data,with_cntry=True):
    t0=2020
    # - Heatmap

    Scenarios = ['NPI','NDC','NDC_CCS1','NZ','NZ_CCS1']
    Scenarios_names = ['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS']
    scenario = Scenarios[0]
    var = 'Workforce'
    T1 = 2035
    Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)

    CNTRY = [Region_indexes.loc[Region_indexes["Subregion_name"]==region,"Region_name"].values[0] for region in Regions]
    if with_cntry:
        Region_with_cntr = [region+' ('+Region_indexes.loc[Region_indexes["Subregion_name"]==region,"Region_name"].values[0]+')' for region in Regions]
    else:   
        Region_with_cntr = Regions
    zlim, norm, colormap = semisymlognorm()
    Ls = []
    share_n_finding = []
    All_data = pd.DataFrame(index=Region_with_cntr,columns=['country','Workforce'])
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

    Calced_data = pd.DataFrame(data=np.array([Region_with_cntr, Calced_data]).transpose(),
                        columns=['Region_Nam', 'Calced_data']).set_index('Region_Nam')


    All_data['Workforce'] = Calced_data['Calced_data'].astype(float)
    All_data['country'] = CNTRY

    var = "share_destruction"
    for T1 in [2035,2050]:
        for ind_scenario, scenario in enumerate(Scenarios):
            Calced_data = []
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

            Calced_data = pd.DataFrame(data=np.array([Region_with_cntr, Calced_data]).transpose(),
                                columns=['Region_Nam', 'Calced_data']).set_index('Region_Nam')

            All_data[Scenarios_names[ind_scenario]+'\n'+str(T1)] = Calced_data['Calced_data'].astype(float)




    All_data.sort_values(['country','Workforce'],ascending=False,inplace=True)
    All_data.drop(All_data[All_data['Workforce']==0].index,inplace=True)
    All_data.drop(['country'],axis=1,inplace=True)
    return All_data

def Exposure_heatmap(T,Result_data):
    zlim, norm, colormap = semisymlognorm()
    All_data = find_destination_data(T,Result_data)
    # Add a blank column to create a wider gap between 2030 and 2050 scenarios
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(All_data.astype('float'), linewidth = 0.5, cmap=colormap, ax=ax,  norm=norm, cbar=False)

    ax.axvline(x=1,color='white',linewidth = 5)
    ax.axvline(x=6,color='white',linewidth = 5)

    cax2 = ax.inset_axes([1.07, 0, 0.05, 1])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax2, orientation='vertical')
    cax2.set_yscale('linear')
    cax2.set_yticks([-0.004,0,0.02,0.04])
    cax2.set_yticklabels(['-0.4%','0%','2%','4%'])
    cax2.spines[:].set_visible(False)
    cax2.set_ylabel('Share of labour force')
    return fig


#==================================================================================================================================================================================================================
#                  PLOTTING FUNCTIONS
#==================================================================================================================================================================================================================

#%% Main 1-  National employment trajectories
#%% Main 2 - Subnational employment trajectories


def print_subnational_employment_results(T,Result_data):
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
def exposure_scatter(T,Result_data,shade=False):
    Scenarios = ['WO-15C-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS0','WO-NPi-ElecIndus-CCS0',
        'WO-15C-ElecIndus-CCS1','WO-NDCLTT-ElecIndus-CCS1']


    Scenarios_names = ['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS']
    All_data = find_destination_data(T,Result_data,with_cntry=False)
    T1 = 2035
    fig, (ax, ax2) = plt.subplots(1,2,figsize=(7,6),width_ratios=[1, 0.15])
    ax.axvline(x=0, color='k', linestyle='-',linewidth = 0.5,zorder=0)
    for ind_region, region in enumerate(All_data.index):
        ax.scatter(All_data.loc[region, 'Workforce'],len(All_data.index)-ind_region-1, s=25,color='k', label=region)
        ax2.scatter(All_data.loc[region, 'Workforce'],len(All_data.index)-ind_region-1, s=25,color='k', label=region)
        for ind_scenario, scenario in enumerate(Scenarios_names):
            ax.scatter(All_data.loc[region, scenario+'\n'+str(T1)],len(All_data.index)-ind_region-1, s=25, label=region, color=defining_waysout_colour_scheme()[Scenarios[ind_scenario]])
        ax.plot([min([All_data.loc[region, x+'\n'+str(T1)] for x in Scenarios_names]),max([All_data.loc[region, x+'\n'+str(T1)] for x in Scenarios_names])],
                [len(All_data.index)-ind_region-1]*2,color='gray',zorder=0,alpha=0.4,linewidth=3.75)


    ax.set_yticks(range(len(All_data.index)))
    ax.set_yticklabels(All_data.index[::-1])
    ax.set_ylim([-0.5,33.5])
    ax2.set_ylim([-0.5,33.5])
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
                                color=defining_waysout_colour_scheme()[scenario],
                                label=['NPi','NDC-LTT','NDC-LTT w/CCS','1.5°C','1.5°C w/CCS'][ind_scenario]))

    labels = [la.get_label() for la in alines]
    
    if shade:
        ax.fill_between( [ax.get_xticks()[0],4.9e-2], [-1,-1], [21.5, 21.5], color='gray', edgecolor=None, alpha=0.2, zorder=-1)
        ax2.fill_between([5.1e-2,8e-2],               [-1,-1], [21.5, 21.5], color='gray', edgecolor=None, alpha=0.2, zorder=-1)
        ax.text(0.03,10,'China',fontsize=14,fontweight='bold')
        ax.text(0.03,25,'India',fontsize=14,fontweight='bold')

    fig.legend(handles=alines,
                labels=labels,
                    loc='upper center',
                bbox_to_anchor=(0.5, 0.05),
                frameon=False,
                ncol=2)
    return fig
#%% Main 4 - Boxplot of share not finding per scenario

def boxplot_share_not_finding(Result_data,T):
    all_region_names = False

    region_indices = def_region_indices()
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
                    T1 = finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==exclude_downscaled_regions[ind_country])&(Result_data.Scenario==scenario)],threshold,T)

                else:
                    T1 = t1

                Share_U, Share_R, TotDestination = destination_share(Result_data,country,scenario,t0,T1)

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
                        box.set_facecolor(defining_waysout_colour_scheme()[scenario])
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
                        ax.scatter(y,xs,s=4.5e-5*TotDestination[ind],color=defining_waysout_colour_scheme()[scenario])
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
                                color=defining_waysout_colour_scheme()[scenario],
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

#%% EXTENDED DATA =============================================================================
# ===========================================================================================================================

#%% ED1 - Coal production and AR6 comparison
# ===========================================================================================================================

def initiate_ar6():
    
    chn_name = 'Countries of centrally-planned Asia; primarily China'
    ind_name = 'Countries of South Asia; primarily India'
    regions_ar6 = ['World', chn_name, ind_name]

    variables_ar6 = ['Primary Energy|Coal','Trade|Primary Energy|Coal|Volume',
                'Final Energy|Industry|Solids|Coal','Emissions|CO2|Energy and Industrial Processes',
                'Carbon Sequestration|CCS|Biomass','Secondary Energy|Electricity|Coal',
                'Primary Energy|Oil','Primary Energy|Gas','Unemployment|Rate','Employment|Industry|Mining',
                "Investment|Energy Supply|Extraction|Coal",'Capacity|Electricity|Coal','Capacity|Electricity|Coal']
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
    return df, regions_ar6, cols


#%% ED2 - Use of coal primary energy
# ===========================================================================================================================



def Stacked_decomposition_coal_demand(Imaclim_data, T, Result_data, mtoe2ej):
    time_gap = 10

    Ts = range(2015,2095,time_gap)
    Regions = ['CHN','IND']
    Scenarios =['WO-NPi-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS1', 'WO-15C-ElecIndus-CCS0', 'WO-15C-ElecIndus-CCS1']
    Outputs_name = ['NPi','NDC-LTT','NDC-LTT-CCS','1.5°C','1.5°C-CCS']

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

    fig, axs = plt.subplots(len(Regions), len(Scenarios), figsize=standard_figure_size())

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
            cumulated_data = get_cumulated_array(data, min=0)
            cumulated_data_neg = get_cumulated_array(data, max=0)

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
            ax, ax2 = double_axis(ax, region)
            if ind_output == len(Scenarios)-1:
                ax2.set_ylabel('Mt')
            else:
                ax2.set_yticklabels([])


    [ax.set_yticklabels([]) for ax in axs[:,1:].flatten()]
    handles = [aline[0] for aline in alines]
    labelz = [aline[0].get_label() for aline in alines]
    fig.legend(handles=handles,
            labels=labelz,
            loc='upper center',
            ncol=4,
            bbox_to_anchor=(0.5, -0),
            frameon=False)
    return fig


#%% ED3 - Mobility of laid-off coal workers
# ===========================================================================================================================

def main_regions_destination_bars(provincesChina, provincesIndia, Result_data, T):
        
    Provinces = [provincesChina, provincesIndia]
    Countries = ['China', 'India','Shanxi','Jharkhand','Chhattisgarh']

    nls = [6, 8]

    Scenarioss = [ ['NPI','NDC','NDC_CCS1','NZ','NZ_CCS1']]*3
    Scenarioss_name = [[ 'NPi', 'NDC\nLTT', 'NDC\nLTT-CCS', '1.5°C', '1.5°C\nCCS']]*3

    T0s = [2020]*3
    T1s = [2030,2050,'80%']

    Xs = [
        list(range(len(Scenarioss[0])))
    ]*3
    data_save = {}
    fig, axs = plt.subplots(len(Countries),
                            len(T1s),
                            figsize=(20/2.54,13/2.54),
                            )
    for c_index in range(len(Countries)):
        x = 0
        
        region = Countries[c_index]
        if c_index <2:
            provinces = Provinces[c_index]
        else:
            provinces = {x:0 for x in [region]}
        
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
                    
                    t1 =finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==region)&
                                    (Result_data.Scenario==Scenario)],threshold,T)

                else:
                    t1=T1

                data_save, alines = destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
                if (stype_index == 2)&(t1!=2100):
                    ax.text(X[s_index],data_save[(region,Scenario,t1)][:-1].sum(),t1+1,style='italic',fontsize=6)


                x += 1


                if (Scenario == 'NZ')&(stype_index == 1):
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

    letters = [x+')' for x in string.ascii_lowercase]
    [ax.set_ylim([-0.5, 3.8]) for ax in axs[[0,1],:].flatten()]
    [ax.set_ylim([-0.3, 1]) for ax in axs[2:,:].flatten()]

    [ax.set_yticklabels([]) for ax in axs[:,[1,2]].flatten()]
    [ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),letters[1:])]

    fig.legend(handles=alines,
            labels=labs,
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.5, -0.2),
            frameon=False)

    fig.subplots_adjust(hspace=0.02, wspace=0.1)
    return fig, data_save


def print_destination_results(data_save):
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

    # Share of lay-offs not finding new employment
    scenario = 'NPI'
    t1 = 2050
    chn0 = 100-(ds.loc[('China',scenario,t1),'D'])/sum(ds.loc[('China',scenario,t1),['D','I','U']])*100
    ind0 = 100-(ds.loc[('India',scenario,t1),'D'])/sum(ds.loc[('India',scenario,t1),['D','I','U']])*100
    scenario = 'NZ'
    chn1 = 100-(ds.loc[('China',scenario,t1),'D'])/sum(ds.loc[('China',scenario,t1),['D','I','U']])*100
    ind1 = 100-(ds.loc[('India',scenario,t1),'D'])/sum(ds.loc[('India',scenario,t1),['D','I','U']])*100

    print(f'- In the long run, the 1.5°C scenario leads to a significant share of laid-off workers not finding employment by 2050, \n with {chn1:.1f}% of Chinese workers and {ind1:.1f}% of Indian workers not finding new employment \n against {chn0:.1f}% and {ind0:.1f}% respectively in the NPI scenario.')


#%% ED4 - Drivers of job cuts 
# ===========================================================================================================================
# This code is outdated : proper graph was not commited and needs to be redone

def def_destruction_denominator(divide_by,df,scenario,region):
    if divide_by == "2020":
        numerator = df.loc[(scenario,region),"2020"]
    elif divide_by == "Total Destruction":
        numerator = (df.loc[('NPI_PG0',region)].values-df.loc[(scenario,region)].values) 
    else:
        return "Invalid numerator"
    return numerator

def fix_xticks(ax,indices):
    # Fix the tick placement issue
    tick_positions = range(0, len(indices), 10)  # Positions based on categorical indexing
    tick_labels = indices[::10]  # Corresponding year labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)  # Rotate for readability
    return ax

def plot_drivers_destruction(Result_data):
    # Plotting the drivers of destruction in China and India
    regions = ["China","India","Shanxi","Jharkhand","Inner Mongolia","Henan"]
    divide_by = "2020"




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

    legend_handles = [mpatches.Patch(color=color_dict[var], label=var, alpha=0.5) for var in df_plot.columns.get_level_values("Variable")]
    ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1, 1))
    return fig

#%% SUPPLEMENTARY INFORMATION ============================================================================= 
# ===========================================================================================================================

#%% SI3 - Employment trajectories in China and India: calibration uncertainty
# ===========================================================================================================================

#%% SI4 - Labour sectoral mobility: calibration uncertainty
# ===========================================================================================================================
def bar_calibration(Result_data, T, provincesChina, provincesIndia):
    Provinces = [provincesChina, provincesIndia]
    Countries = ['China', 'India']
    nls = [6, 8]

    Scenarioss = [['NPI_min','NPI','NPI_max'],
                ['NDC_min','NDC','NDC_max'],
                ['NDC_CCS1_min','NDC_CCS1','NDC_CCS1_max'],
                ['NZ_min','NZ','NZ_max'],
                ['NZ_CCS1_min','NZ_CCS1','NZ_CCS1_max']]

    Scenarioss_name = [['Min','Central','Max']]*5

    t0 = 2020
    t1 = 2040

    Xs = [
        [x for x in range(len(Scenarioss[0]))],
        [x for x in range(len(Scenarioss[1]))],
        [x for x in range(len(Scenarioss[2]))],
        [x for x in range(len(Scenarioss[3]))],
        [x for x in range(len(Scenarioss[4]))]
        ]

    data_save = {}
    fig, axs = plt.subplots(2,
                            5,
                            figsize=standard_figure_size(),
        )

    for c_index in [0, 1]:
        x = 0
        provinces = Provinces[c_index]
        region = ['China', 'India'][c_index]
        for stype_index in range(5):
            ax = axs[c_index][stype_index]
            if stype_index == 0:
                ax.set_ylabel(region + '\nMillion workers')
            Scenarios = Scenarioss[stype_index]
            Sc = Scenarios
            X = Xs[stype_index]
            for s_index, Scenario in enumerate(Scenarios):
                data_save, alines = destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
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
                ax.set_title(['NPI', 'NDC','NDC w/CCS',
                            '1.5°C','1.5°C w/CCS'][stype_index])
            if stype_index != 0:
                ax.set_yticks([])
            ax.set_ylim([-0.6, 2.38])
            ax.set_ylim([-1, 3.8])

    [ax.set_yticklabels([]) for ax in axs[:,[1,2]].flatten()]
    [ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)'])]


    fig.legend(handles=alines,
            labels=labs,
            loc='center left',
            ncol=1,
            bbox_to_anchor=(0.9, 0.5),
            frameon=False)

    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    
    return fig, data_save

#%% SI5 - Labour sectoral mobility: retirement age sensitivity
# ===========================================================================================================================
def bar_retirement_age(Result_data, T, provincesChina, provincesIndia):
    Provinces = [provincesChina, provincesIndia]
    Countries = ['China', 'India']
    nls = [6, 8]

    Scenarioss = [['NPI',  'NPI_R55'],
                ['NDC', 'NDC_R55'],
                ['NDC_CCS1', 'NDC_CCS1_R55'],
                ['NZ','NZ_R55'],
                ['NZ_CCS1','NZ_CCS1_R55']]

    Scenarioss_name = [['NPI','Retirement \n55'],
                    ['NDC', 'Retirement \n55'],
                    ['NDC w/ CCS', 'Retirement \n55'],
                    ['1.5°C','Retirement \n55'],
                    ['1.5°C w/ CCS','Retirement \n55']]

    t0 = 2020
    t1 = 2050

    Xs = [
        [x for x in range(len(Scenarioss[0]))],
        [x for x in range(len(Scenarioss[1]))],
        [x for x in range(len(Scenarioss[2]))],
        [x for x in range(len(Scenarioss[3]))],
        [x for x in range(len(Scenarioss[4]))]
        ]

    data_save = {}
    fig, axs = plt.subplots(2,
                            5,
                            figsize=standard_figure_size(),
        )

    for c_index in [0, 1]:
        x = 0
        provinces = Provinces[c_index]
        region = ['China', 'India'][c_index]
        for stype_index in range(5):
            ax = axs[c_index][stype_index]
            if stype_index == 0:
                ax.set_ylabel(region + '\nMillion workers')
            Scenarios = Scenarioss[stype_index]
            Sc = Scenarios
            X = Xs[stype_index]
            for s_index, Scenario in enumerate(Scenarios):
                data_save, alines = destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
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
                ax.set_title(['NPI', 'NDC','NDC w/CCS',
                            '1.5°C','1.5°C w/CCS'][stype_index])
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
    return fig, data_save

def print_destination_results_retirement(data_save,t1=2050):
    ds = pd.DataFrame(data_save).T
    ds.columns = ['R', 'D', 'I', 'U', 'H']
    ch60 = ds.loc[('China','NZ',t1),'U']
    ch55= ds.loc[('China','NZ_R55',t1),'U']

    lo_reduc = (1-sum(ds.loc[('China','NZ_R55',t1),['U','D','I']])/sum(ds.loc[('China','NZ',t1),['U','D','I']]))*100

    print(f'Sensitivity analysis shown in the annex where retirement age is moved from 60 to 55 show that such a policy would reduce the number of layoffs by {lo_reduc:0.1f}% and hence the number of workers leaving into unemployment from {ch60} to {ch55}.')


    lo_reduc = (1-sum(ds.loc[('China','NDC_R55',t1),['U','D','I']])/sum(ds.loc[('China','NDC',t1),['U','D','I']]))*100
    in_reduc = (1-sum(ds.loc[('India','NDC_R55',t1),['U','D','I']])/sum(ds.loc[('India','NDC',t1),['U','D','I']]))*100
    print(f'This policy is more efficient for lower ambition scenarios with {lo_reduc:0.1f}% under NDC-LTT ({in_reduc:0.1f}% in India)')

#%% SI6 - Employment trajectories: productivity growth sensitivity
# ===========================================================================================================================

#%% SI7 - Labour sectoral mobility: productivity growth sensitivity
# ===========================================================================================================================
def bar_productivity(Result_data, T, provincesChina, provincesIndia):
    Provinces = [provincesChina, provincesIndia]
    Countries = ['China', 'India']
    nls = [6, 8]


    Scenarioss = [["NPI", "NPI_C0I0_P0", "NPI_C0I0_P100", "NPI_C20I5_P0", "NPI_C40I10_P100"],
                ["NDC", "NDC_C0I0_P0", "NDC_C0I0_P100", "NDC_C20I5_P0", "NDC_C40I10_P100"],
                ["NDC_CCS1","NDC_CCS1_C0I0_P0", "NDC_CCS1_C0I0_P100", "NDC_CCS1_C20I5_P0", "NDC_CCS1_C40I10_P100"],
                    ["NZ", "NZ_C0I0_P0", "NZ_C0I0_P100", "NZ_C20I5_P0", "NZ_C40I10_P100"],
                    ["NZ_CCS1","NZ_CCS1_C0I0_P0", "NZ_CCS1_C0I0_P100", "NZ_CCS1_C20I5_P0", "NZ_CCS1_C40I10_P100"]]

    Scenarioss_name = [["NPI","C0I0_P0", "C0I0_P100", "C20I5_P0", "C40I10_P100"],
                    ["NDC", "C0I0_P0", "C0I0_P100", "C20I5_P0", "C40I10_P100"],
                    ["NDC w/ CCS", "C0I0_P0", "C0I0_P100", "C20I5_P0", "C40I10_P100"],
                    ["1.5°C","C0I0_P0", "C0I0_P100", "C20I5_P0", "C40I10_P100"],
                    ["1.5°C w/ CCS","C0I0_P0", "C0I0_P100", "C20I5_P0", "C40I10_P100"]]

    Scenarioss_name = [["Central","No decapacity Lo prod", "No decapacity Hi prod", "Lo decapacity Lo prod", "Hi decapacity Hi prod"],
                    ["Central", "No decapacity Lo prod", "No decapacity Hi prod", "Lo decapacity Lo prod", "Hi decapacity Hi prod"],
                    ["Central", "No decapacity Lo prod", "No decapacity Hi prod", "Lo decapacity Lo prod", "Hi decapacity Hi prod"],
                    ["Central","No decapacity Lo prod", "No decapacity Hi prod", "Lo decapacity Lo prod", "Hi decapacity Hi prod"],
                    ["Central","No decapacity Lo prod", "No decapacity Hi prod", "Lo decapacity Lo prod", "Hi decapacity Hi prod"]]



    t0 = 2020
    t1 = 2050

    Xs = [
        [x for x in range(len(Scenarioss[0]))],
        [x for x in range(len(Scenarioss[1]))],
        [x for x in range(len(Scenarioss[2]))],
        [x for x in range(len(Scenarioss[3]))],
        [x for x in range(len(Scenarioss[4]))]
        ]

    data_save = {}
    fig, axs = plt.subplots(2,
                            5,
                            figsize=standard_figure_size(),
        )

    for c_index in [0, 1]:
        x = 0
        provinces = Provinces[c_index]
        region = ['China', 'India'][c_index]
        for stype_index in range(5):
            ax = axs[c_index][stype_index]
            if stype_index == 0:
                ax.set_ylabel(region + '\nMillion workers')
            Scenarios = Scenarioss[stype_index]
            Sc = Scenarios
            X = Xs[stype_index]
            for s_index, Scenario in enumerate(Scenarios):
                data_save, alines = destination_bar(Result_data, X, T, t0, t1, Scenario, ax, region, provinces, data_save, s_index)
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
                ax.set_title(['NPI', 'NDC','NDC w/CCS',
                            '1.5°C','1.5°C w/CCS'][stype_index])
            if stype_index != 0:
                ax.set_yticks([])
            ax.set_ylim([-0.6, 2.38])
            ax.set_ylim([-1, 3.8])

    [ax.set_yticklabels([]) for ax in axs[:,[1,2]].flatten()]
    [ax.text(0.02,0.96, label, transform=ax.transAxes, fontsize= 11, fontweight='bold', va='top', ha='left') for ax, label in zip(axs.flatten(),['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)'])]


    fig.legend(handles=alines,
            labels=labs,
            loc='center left',
            ncol=1,
            bbox_to_anchor=(0.9, 0.5),
            frameon=False)

    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    return fig, data_save

#%% SI9 - National employment trajectories: supply vs demand-driven sensitivity
# ===========================================================================================================================


#%% SI10 - Subnational employment trajectories: supply vs demand-driven sensitivity
# ===========================================================================================================================

#%% SI11 - Vulnerability of regions to coal transition: supply vs demand-driven sensitivity
# ===========================================================================================================================

def boxplot_share_not_finding_demand(Result_data, T):

    all_region_names = False
    region_indices = def_region_indices()
    destination_variables = [x for x in Result_data.Variable.unique() if 'Coal Worker Destination' in x if 'Retire' not in x and 'Hire' not in x]

    Countries = ["CHN",'IND']
    exclude_downscaled_regions = ['China','India']

    Scenarios = [
                'NPI_tra',
                'NPI',
                'NDC_CCS1_tra',
                'NDC_CCS1',
                'NDC_tra',
                'NDC',
                'NZ_CCS1_tra',
                'NZ_CCS1',
                'NZ_tra',
                'NZ',
                ][::-1]


    loc = {"Shanxi":(0.75,-0.1),
        "Jharkhand":(0.7,1.4),
        "Odisha":(0.85,0.8),
        'Chhattisgarh':(0.1,1)}



    t0s = [2020, 2020]
    t1s = [2030,2050] 

    for_abstract = True
    if for_abstract:
        t0s = [2020, 2020]
        t1s = [2035,2050]

    fig, axs = plt.subplots(2,2,figsize=(16/2.54,9/2.54))



    for ind_t,(t0,t1) in enumerate(zip(t0s,t1s)):
        
        x=0
        
        
        for ind_country, country in enumerate(Countries):
            for ind_scenario, scenario in enumerate(Scenarios):
                if "_tra" in scenario:
                    alpha = 0.75
                else:
                    alpha = 0.3

                if type(t1) is str:
                    threshold = 0.8
                    T1 = finding_emp_threshold_date(Result_data[(Result_data['Downscaled Region']==exclude_downscaled_regions[ind_country])&(Result_data.Scenario==scenario)],threshold,T)

                else:
                    T1 = t1

                Share_U, Share_R, TotDestination = destination_share(Result_data,country,scenario,t0,T1)

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
                if (scenario=="NZ") & (t1==2050):
                    u_national = sum((TotDestination*Share_U).dropna())/sum(TotDestination.dropna())
                    r_national = ((TotDestination*Share_R/(1-Share_R))).dropna()
                    r_national = sum(r_national.dropna())/(sum(r_national.dropna())+sum(TotDestination.dropna()))
                    print(f'Under {scenario}, between {t0}-{t1} in {country}, the share of workers leaving into unemployment is {u_national:0.2f} ({min(Share_U):0.2f}-{max(Share_U):0.2f}), that\'s {sum((TotDestination*Share_U).dropna()):0.0f} workers' )
                    print(f'Under {scenario}, between {t0}-{t1} in {country}, the share of workers leaving into retirement is {r_national:0.2f} ({min(Share_R):0.2f}-{max(Share_R):0.2f})' )
                    region = ['Shanxi','Jharkhand'][ind_country]
                    su = Share_U[region]
                    tu = (TotDestination*Share_U)[region]
                    print(f'Under {scenario} in {region}, between {t0}-{t1}, {su:0.2f}% (ie {tu:0.0f} workers) may not find employment')
                
                pos = x+[-0.87,-0.7,-0.52,-0.35,-0.17,0,0.17,0.35,0.52,0.7,0.87][ind_scenario]/2

                for ind_var, var in enumerate([Share_U,Share_R]):
                    
                    ax = axs[ind_t,ind_var]
                
                    if ind_var==0:
                        ax.arrow(0.8,0.5,0.1,0,head_width=0.03,color='k')
                        ax.text(0.85,0.24,"increased\n vulnerability",fontsize=5,fontweight='normal')
                    else:
                        ax.arrow(0.2,0.5,-0.1,0,head_width=0.03,color='k')
                        ax.text(0.11,0.24,"increased\n vulnerability",fontsize=5,fontweight='normal')


                    bxplot=ax.boxplot(var,
                            positions=[pos],
                            vert=False,
                            patch_artist=True,
                            showfliers=False,
                            whiskerprops={'linestyle': 'none'},
                            capprops={'linestyle':'none'},
                            zorder=1,
                            widths=0.09/2 )
                    
                    # Customizing the appearance
                    for box in bxplot['boxes']:
                        box.set_facecolor(defining_waysout_colour_scheme()[scenario.split('_tra')[0]])
                        box.set_alpha(alpha)  
                        box.set_edgecolor('black')
                        box.set_linewidth(0.5)  # Set the edge linewidth here
                        
                    # Set the median line color to black
                    for median in bxplot['medians']:
                        median.set_color('black')


                    ax.scatter(np.mean(var),pos,s=3,color='k',zorder=2)
                    # Data points
                    for ind in var.index:
                        y = var.loc[ind]
                        xs = np.random.normal(pos, 0.05/2, size=1)
                        ax.scatter(y,xs,s=4.5e-5*TotDestination[ind]/2,color=defining_waysout_colour_scheme()[scenario.split('_tra')[0]],alpha=alpha)
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
    for ind,ind_scenario in enumerate(range(8,-1,-2)):
        scenario = Scenarios[ind_scenario]
        alines.append(ax.scatter([],[],
                                color=defining_waysout_colour_scheme()[scenario],
                                label=['NPi','NDC-LTT','NDC-LTT w/CCS','1.5°C','1.5°C w/CCS'][ind]))
        
    alines.append(ax.scatter([],[],s=0,label='Number of Workers'))
    for size in [5000,50000,500000]:
        alines.append(ax.scatter([],[],s=4.5e-5*size,color='grey',label=size))
    alines.append(mpatches.Patch(color='grey', alpha=1, label='Demand-driven'))
    alines.append(mpatches.Patch(color='grey', alpha=0.5, label='Supply-driven'))
    labels = [la.get_label() for la in alines]
    handles = [label for label in alines]

    fig.legend(handles=alines,
            labels=labels,
                loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False)



    fig.set_tight_layout('tight')
    return fig


#%% P1 - Scenarios descriptions
# ===========================================================================================================================


def plot_scenario_description(Imaclim_data, T, Step=5):
    Colors = defining_waysout_colour_scheme()
    Countries = ['World','CHN','IND']
    Scenarios = ['WO-15C-ElecIndus-CCS0', 'WO-NDCLTT-ElecIndus-CCS0','WO-NPi-ElecIndus-CCS0',
            'WO-15C-ElecIndus-CCS1','WO-NDCLTT-ElecIndus-CCS1']


    Scenarios_names = ['1.5°C','NDC-LTT','NPi','1.5°C-CCS','NDC-LTT-CCS']
    Variables = ['Emissions|CO2|Energy and Industrial Processes',
                'Resource|Extraction|Coal',]

    fig, axs = plt.subplots(len(Variables),len(Countries),figsize=standard_figure_size())

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

            

    add_coaloutput_comparisons([[axs[1,0]],[axs[1,1]],[axs[1,2]]],Countries,[0])

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
    return fig