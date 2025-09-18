# ===========================================================================================================================
# ===========================================================================================================================  
# Regional employment vulnerability to rapid coal transition in China and India, an integrated and downscaled assessment
#                                                    Conference graphs
# ===========================================================================================================================
# ===========================================================================================================================
"""

This file is a companion to `Plots.ipynb` aimed at plotting additional graphs 
not meant to be published but useful for presentation with bigger format and maps to help audience identify key regions.

"""

#%%
# Importing libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import seaborn as sns
from cartopy import crs as ccrs
from shapely.geometry import box
from shapely.geometry import Point
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# Plotting functions

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


file_name = list(np.array([['../coal.labour.nexus/output/Downscaled_coal_labour_' + x + y + '.csv' for x in ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1']] for y in ['','_PG0','_R55','_min','_max']]).flatten())

Result_data = []
for file in file_name:
    data = pd.read_csv(file)
    data.Scenario = data.Scenario
    Result_data.append(data)


#======
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



region_indices = pd.read_csv('..\coal.labour.nexus\data\Coal_labour\Downscaling\Indexes.csv')

# ===========================================================================================================================
#Unit
exa2giga        =                 1e9 # G / E
tep2gj          =              41.855 # GJ/tep
mtoe2gj         =        1e6 * tep2gj # GJ/Mtep
mtoe2ej         =  mtoe2gj / exa2giga # EJ/Mtep


#%%
GEM = pd.read_excel("C:/Users/augus/Data/GEM/Global-Coal-Mine-Tracker-April-2024.xlsx",sheet_name="Global Coal Mine Tracker (Non-C")
GEM['Productivity'] = GEM['Capacity (Mtpa)'].apply(pd.to_numeric,errors="coerce")/GEM['Workforce Size'].apply(pd.to_numeric,errors="coerce")

#%%



GEM['Longitude'] = pd.to_numeric(GEM['Longitude'], errors='coerce')
GEM['Latitude'] = pd.to_numeric(GEM['Latitude'], errors='coerce')
GEM = GEM.dropna(subset=['Longitude', 'Latitude'])
GEM['geometry'] = GEM.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
GEM = gpd.GeoDataFrame(GEM, geometry=GEM.geometry, crs="EPSG:4326")

GEM['Capacity (Mtpa)'] = pd.to_numeric(GEM['Capacity (Mtpa)'], errors='coerce')

#%%

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
region = 'Shanxi'

def format_side_axis(ax,region,remove_splin=True,position='right'):


    ax.axhline(y=100,color='k',linewidth=2,zorder=-1) 
    ax.axhline(y=0,color='k',linewidth=2,zorder=-1) 
    if remove_splin:
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
    
    lims = {
        'China': [-200,4000],
        'Shanxi': [-50, 1260],
        'India': [-150,1900],
        'Jharkhand': [-40, 750]
    }
    if region in ['West Bengal', 'Odisha','Shandong','Inner Mongolia']:
        ax.set_ylim(-50,325)
    elif region in lims.keys():
        ax.set_ylim(lims[region])

    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    # Set tick label size
    ax.tick_params(axis='both', which='both', labelsize=24)
    ax.set_ylabel('[Thousand workers]',fontsize=24)
    if position == 'right':
        ax.text(0.95,0.87,region,transform=ax.transAxes, fontsize= 40, fontweight='bold', ha='right')
    elif position == 'left':
        ax.text(0.1,0.87,region,transform=ax.transAxes, fontsize= 40, fontweight='bold', ha='left')
    return ax


def plot_labour_wedges(ax,region,scenario,threshold_percentage=95,step=5,position='right',T1=None):
    T = np.array(range(2015,2101))
    start = 2015-2015
    
    end = 2070-2015
    # Get trajectories
    L_NPI = Result_data.loc[(Result_data.Scenario == 'NPI')&(Result_data['Downscaled Region'] == region)&(Result_data.Variable=='Employment|Coal|Downscaled'),[str(x) for x in T]].values[0].astype(float)*1e-3
    L_NPI0= Result_data.loc[(Result_data.Scenario == 'NPI_PG0')&(Result_data['Downscaled Region'] == region)&(Result_data.Variable=='Employment|Coal|Downscaled'),[str(x) for x in T]].values[0].astype(float)*1e-3
    L     = Result_data.loc[(Result_data.Scenario == scenario)&(Result_data['Downscaled Region'] == region)&(Result_data.Variable=='Employment|Coal|Downscaled'),[str(x) for x in T]].values[0].astype(float)*1e-3
    L0    = Result_data.loc[(Result_data.Scenario == scenario+'_PG0')&(Result_data['Downscaled Region'] == region)&(Result_data.Variable=='Employment|Coal|Downscaled'),[str(x) for x in T]].values[0].astype(float)*1e-3
    print(f'L is there {np.shape(L)}')
    # Calculate the stop date of the wedges based on when coal labour "phase out" is achieved
    threshold = (100-threshold_percentage)/100
    if T1:
        stop_date = T1-2015+1 
    elif len(T[L<L[5]*threshold])>0:
        stop_date = min(T[L<L[5]*threshold][0]-2015,end)
    else:
        stop_date = end

    # Caluclate the wedges
    productivityNPI = L_NPI0-L_NPI
    production      = L_NPI0-L0
    addiproductivity=  L_NPI-L_NPI0-L+L0
    # ... Split positive and negative additional productivity in mitigation scenario wrt NPi
    addiproductivity_positive = addiproductivity.copy()
    addiproductivity_positive[addiproductivity_positive<0]=0
    
    addiproductivity_negative = addiproductivity.copy()
    addiproductivity_negative[addiproductivity_negative>0]=0

    # PLOT
    ax.fill_between(T[start:stop_date:step],
                    (L_NPI0)[start:stop_date:step],
                    (L_NPI0-production)[start:stop_date:step],
                    color=sns.color_palette('Set2')[1],alpha=0.7,edgecolor=None)
    ax.fill_between(T[start:stop_date:step],
                    (L_NPI0-production)[start:stop_date:step],
                    (L_NPI0-production-productivityNPI)[start:stop_date:step],
                    color=sns.color_palette('Set2')[2],alpha=0.7,edgecolor=None)
    ax.fill_between(T[start:stop_date:step],
                    L_NPI0[start:stop_date:step],
                    (L_NPI0-addiproductivity_negative)[start:stop_date:step],
                    color=sns.color_palette('Set2')[3],alpha=0.7,edgecolor=None)
    ax.fill_between(T[start:stop_date:step],
                    (L_NPI0-production-productivityNPI)[start:stop_date:step],
                    (L_NPI0-production-productivityNPI-addiproductivity_positive)[start:stop_date:step],
                    color=sns.color_palette('Set2')[4],alpha=0.7,edgecolor=None)
    print(T[start:end:step])

    ax.plot(T[start:end:step],L[start:end:step],color='k',linewidth=5)
    ax.plot(T[start:stop_date:step],L_NPI0[start:stop_date:step],color='k',linestyle=':',linewidth=2)


    if T1:
        print(f'T1 is {T1}')
        ax.plot([2020,T1+20],[L[5],L[5]],color='grey',linewidth=2,linestyle='--')
        ax.plot([T1,T1+20],[L[stop_date],L[stop_date]],color='grey',linewidth=2,linestyle='--')
        # ax.plot([T1+5,T1+5],[L[5],L[end]],color='lightgrey',linewidth=20)
        ax.add_patch(matplotlib.patches.Rectangle(
            (T1+20, L[stop_date]),  # (x, y) position
            5,             # width
            L[5] - L[stop_date],   # height
            color='grey',
            alpha=0.5,
            zorder=-1
        ))

    ax = format_side_axis(ax,region,position=position)
    return ax 


def plot_labour_graph(ax,region,Scenarios):
    
    for scenario in Scenarios:
        t = range(2015,2070,5)
        y0 = Result_data.loc[
            (Result_data.Scenario==scenario+'_PG0')&
            (Result_data.Variable=='Employment|Coal|Downscaled')&
            (Result_data['Downscaled Region']==region),
            [str(x) for x in t]
        ].values[0]*1e-3
        ax.plot(t,y0,
                color = defining_waysout_colour_scheme()[scenario],
                linewidth = 4, linestyle = '--',zorder=-1)

        
        y = Result_data.loc[
            (Result_data.Scenario==scenario)&
            (Result_data.Variable=='Employment|Coal|Downscaled')&
            (Result_data['Downscaled Region']==region),
            [str(x) for x in t]
        ].values[0]*1e-3
        ax.plot(t,y,
                color = defining_waysout_colour_scheme()[scenario],
                linewidth = 5)

    ax = format_side_axis(ax,region)
    return ax


def plot_productivity_graph(ax,region,Scenarios,normalise=True,remove_splin=True):
    country = {
        'CHN':'China',
        'IND':'India'
    }[Result_data[Result_data['Downscaled Region']==region].Region.values[0]]
    for scenario in Scenarios:
        t = range(2015,2070,5)

        
        y = Result_data.loc[
            (Result_data.Scenario==scenario)&
            (Result_data.Variable=='Labour Productivity|Coal|Downscaled')&
            (Result_data['Downscaled Region']==region),
            [str(x) for x in t]
        ].values[0]*1e3

        if normalise:
            denom = Result_data.loc[
                (Result_data.Scenario==scenario)&
                (Result_data.Variable=='Labour Productivity|Coal|Downscaled')&
                (Result_data['Downscaled Region']==country),
                str(t[0])
            ].values[0]*1e3
            y = y/denom

        ax.plot(t,y,
                color = defining_waysout_colour_scheme()[scenario],
                linewidth = 5)

    
    ax.axhline(y=0,color='k',linewidth=2,zorder=-1) 
    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    if remove_splin:
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)

    # Set tick label size
    ax.tick_params(axis='both', which='both', labelsize=24)
    ax.set_ylabel('[EJ/Yr/Thousand workers]',fontsize=24)
    # ax.set_ylim([-0.01,0.07527])

    if normalise:
        ax.set_ylim([0,11])
        ax.axhline(y=1,color='k',linewidth=1,)
    
    ax.text(0.85,0.87,region,transform=ax.transAxes, fontsize= 40, fontweight='bold', ha='right')
    return ax



def map_employment(ax,var,t1,scenario,zlim,colormap,crs,norm,highlight_regions=[]):

    clean_layout = True
    if clean_layout:
        row_width = 0
    else:
        row_width = 0.75

    # Define bounding box in degrees
    bbox = box(60, 3, 140, 53)  # lon_min, lat_min, lon_max, lat_max
    gdf_bbox = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    
    crs_proj4 = crs.proj4_init


    Regions = np.unique(Result_data[~Result_data['Downscaled Region'].isin(
    ['China', 'India', 'Rest of Asia', 'Asia without Indonesia', 'Indonesia'])]
                    ['Downscaled Region'].values)
    
    Calced_data = []
    for region in Regions:
            Calced_data.append(Result_data.loc[
                (Result_data['Downscaled Region']==region)&
                (Result_data.Scenario==scenario)&
                (Result_data.Variable==var),
                str(t1)
            ].values[0]*1e-3)
    
    Calced_data =pd.DataFrame({
        'Region_Nam': list(Regions), 
        'Calced_data': Calced_data,
    })

    Asia_Data = Asia.merge(Calced_data, on='Region_Nam').to_crs(crs_proj4)

    Asia_Data['Calced_data'] = pd.to_numeric(Asia_Data['Calced_data'], errors='coerce')
    
    color_not_plotted = '#EAEAEA'

    if clean_layout:
        threshold = 2
        empty_regions = Asia_Data[Asia_Data['Calced_data']<threshold]
        Asia_Data = Asia_Data[Asia_Data['Calced_data']>threshold]
        empty_regions.plot(ax=ax, color=color_not_plotted, edgecolor='none', linewidth=row_width, rasterized=True)
        edgecolor = 'grey'
    else:
        edgecolor = 'none'


    cbar = Asia_Data.plot(column='Calced_data',
                        cmap=colormap,
                        legend=False,
                        ax=ax,
                        edgecolor=edgecolor,
                        missing_kwds={
                            "color": color_not_plotted,
                            "label": "Missing values",
                        },
                        vmin=zlim[0],
                        vmax=zlim[1],
                        linewidth=0.75,
                        rasterized=True,
                        norm=norm)
    
    # Split the GeoDataFrame into two: normal and highlighted
    highlighted = Asia_Data[Asia_Data['Region_Nam'].isin(highlight_regions)].to_crs(crs_proj4)
    others = Asia_Data[~Asia_Data['Region_Nam'].isin(highlight_regions)].to_crs(crs_proj4)
    # Plot all regions with thin grey borders
    if not clean_layout:
        others.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5,
                        rasterized=True)

    # Plot the 3 highlighted regions with thick black borders

    rest_Asia =  gpd.clip(Asia[Asia['region']=='Asia'], gdf_bbox).to_crs(crs_proj4)

    rest_Asia.plot(ax=ax, color=color_not_plotted, edgecolor='grey',linewidth=row_width,rasterized=True)
    Asia[Asia['region']=='Disputed'].to_crs(crs_proj4).plot(ax=ax, color=color_not_plotted, edgecolor='grey', linestyle='--',linewidth=row_width,rasterized=True)
    
    GEM_proj = gpd.clip(GEM, gdf_bbox).to_crs(crs_proj4)
    GEM_proj['x'] = GEM_proj.geometry.x
    GEM_proj['y'] = GEM_proj.geometry.y
    ax.scatter(
        GEM_proj['x'],
        GEM_proj['y'],
        transform=crs,                  # projection matches map
        color='dimgrey',
        s=GEM_proj['Capacity (Mtpa)']*0.75,  # scale size (adjust multiplier as needed)
        alpha=0.7,
        rasterized=True
    )
    

    for ind, buffer_size in enumerate([-4,-2,0]):
        region_buffer = highlighted.buffer(buffer_size*1e4)  # adjust as needed
        buffer_gdf = gpd.GeoDataFrame(geometry=region_buffer, crs=crs_proj4)
        ax.add_geometries(buffer_gdf.geometry, crs=crs, facecolor='none',  edgecolor='black',linewidth=(ind+1)/4,
                        rasterized=True)


    Coordinates = {
        'Shanxi': (544805, 500634),
        'Shandong': (2504440, 835039),
        'Inner Mongolia': (-100000, 1571131),
        'Odisha': (-1447408, -1345884),
        'Jharkhand': (-2002751, -317525),
        'West Bengal': (-308100, -731698)
    }
    


    for region in highlight_regions:
        x, y = Coordinates[region]
        ax.text(
            x, y,
            region,
            fontsize=20,
            fontweight='bold',
            ha='center',
            va='center',
            color='black'  # You can change this for contrast
        )

    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels = False
    ax.set_extent([67, 130, 7, 55], crs=ccrs.PlateCarree())


    return ax, cbar


color_dict = {
    ('Operating', 0.0): sns.color_palette('Reds',n_colors=8)[7],
    ('Operating', 5.0): sns.color_palette('Reds',n_colors=8)[6],
    ('Operating', 10.0): sns.color_palette('Reds',n_colors=8)[5],
    ('Operating', 15.0): sns.color_palette('Reds',n_colors=8)[4],
    ('Operating', 20.0): sns.color_palette('Reds',n_colors=8)[3],
    ('Operating', '25+'): sns.color_palette('Reds',n_colors=8)[2],
    ('Operating', 'Unknown'): sns.color_palette('Oranges',n_colors=7)[1],
    ('Proposed', 'Proposed'):  sns.color_palette('Blues',n_colors=7)[2],
}


def plot_mines_capacity(region,ax,color_dict=color_dict,dropna=True,xlabel=True):

    status = ['Operating', 'Proposed']
    types = ['Underground', 'Surface', 'Underground & Surface']

    # Filter the data
    if region in ['China','India']:
        local_mines = GEM[(GEM.Country == region) &
                        (GEM['Status'].isin(status)) &
                        (GEM['Mine Type'].isin(types))]
    else:
        # Filter the data
        local_mines = GEM[(GEM['State, Province'] == region) &
                        (GEM['Status'].isin(status)) &
                        (GEM['Mine Type'].isin(types))]

    # Clean and bin the R/P ratio
    local_mines['Reserve to Production Ratio (R/P)'] = 5 * (pd.to_numeric(local_mines['Reserve to Production Ratio (R/P)'], errors='coerce') // 5)
    local_mines.loc[local_mines['Reserve to Production Ratio (R/P)'] > 20, 'Reserve to Production Ratio (R/P)'] = '25+'
    if dropna:
        local_mines = local_mines.dropna(subset=['Reserve to Production Ratio (R/P)'])
    else:
        local_mines['Reserve to Production Ratio (R/P)'] = local_mines['Reserve to Production Ratio (R/P)'].fillna('Unknown')
    local_mines.loc[local_mines.Status=='Proposed','Reserve to Production Ratio (R/P)' ]='Proposed'

    # Group and pivot
    local_mines = (local_mines
        .loc[:, ['Mine Type', 'Status', 'Capacity (Mtpa)', 'Reserve to Production Ratio (R/P)']]
        .groupby(['Mine Type', 'Status', 'Reserve to Production Ratio (R/P)'])
        .sum()
        .reset_index()
        .pivot(index='Mine Type', columns=['Status', 'Reserve to Production Ratio (R/P)'], values='Capacity (Mtpa)')
    )
    
    def rp_sort_key(rp):
        try:
            # Try to convert to float
            return float(rp)
        except ValueError:
            # Place '25+' just after 25, 'unknown' last
            if rp == '25+':
                return 25.1
            elif rp == 'unknown':
                return float('inf')
            else:
                return float('inf')  # Fallback for any unexpected strings

    
    local_mines = local_mines[sorted(local_mines.columns, key=lambda col: col[0])]
    local_mines = local_mines[sorted(local_mines.columns, key=lambda col: rp_sort_key(col[1]))]


    # Reindex to include all desired types and keep order
    local_mines = local_mines.reindex(types)
    local_mines = local_mines.rename(index={'Underground & Surface': 'Both'})

    columns = local_mines.columns  # MultiIndex with (Status, R/P)
    colors = [color_dict.get(col, '#cccccc') for col in columns]  # fallback to grey if not in dict


    # Plot
    local_mines.plot.bar(stacked=True, color=colors,ax=ax,legend=False)

    # Make remaining spines thicker

    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    # Set tick label size
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.set_ylabel('Capacity\n[Mtpa]',fontsize=20)
    ax.set_xlabel('')
    if not xlabel:
        ax.set_xticklabels('')

    return ax

#%%
def plot_main(plot_type,country,scenario='NPI',ind='0',plot_capacity=True,savefig=True,show_uncertainty=False):
    Scenarios = ['NDC_CCS1','NZ_CCS1','NPI','NDC','NZ']
    Scenarios_names = ['NDC-LTT w/CCS','1.5°C w/CCS','NPi','NDC-LTT','1.5°C']


    crs = ccrs.AlbersEqualArea(central_longitude=100, central_latitude=30)



    T1 = 2035
    fig, ax = plt.subplots(1,1,subplot_kw={"projection": crs}, figsize=(10, 8))
                            

    zlim = [0,300]




    vmin, vmax = zlim[0], zlim[1]
    colormap = plt.cm.OrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    if country == 'China':
        highlight_regions = ['Shanxi','Inner Mongolia','Shandong']
    elif country == 'India':    
        
        highlight_regions = ['Jharkhand','West Bengal','Odisha']


    var = 'Employment|Coal|Downscaled'
    t1= 2020
    ax, cbar = map_employment(ax,var,t1,scenario,zlim,colormap,crs,norm,highlight_regions=highlight_regions)


    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])  # Required for colorbar creation

    # Create an axis for the colorbar (customize position [left, bottom, width, height])
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])  # Adjust position as needed
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='max')  # Arrow on right
    cbar_ax.set_xlabel('Coal workforce [thousand workers]')

    ax.axis('off')


    side_reg  = [country] + highlight_regions

    if plot_capacity:
        r_axis1 = ax.inset_axes([-0.68, 0.55, 0.60, 0.7])
        r_axis2 = ax.inset_axes([-0.68, -0.3, 0.60, 0.7])
        r_axis3 = ax.inset_axes([1.45, 0.55, 0.60, 0.7])
        r_axis4 = ax.inset_axes([1.45, -0.3, 0.60, 0.7])

        c_axis1 = ax.inset_axes([-1, 0.55   , 0.15, 0.7])
        c_axis2 = ax.inset_axes([-1, -0.3  , 0.15, 0.7])
        c_axis3 = ax.inset_axes([1.15, 0.55 , 0.15, 0.7])
        c_axis4 = ax.inset_axes([1.15, -0.3, 0.15, 0.7])

    else:
        r_axis1 = ax.inset_axes([-1, 0.7, 0.9, 0.7])
        r_axis2 = ax.inset_axes([-1, -0.2, 0.9, 0.7])
        r_axis3 = ax.inset_axes([1.15, 0.7, 0.9, 0.7])
        r_axis4 = ax.inset_axes([1.15, -0.2, 0.9, 0.7])

    side_axes = [r_axis1,r_axis2,r_axis3,r_axis4]
    for axis, region in zip(side_axes,side_reg):
        if plot_type == 'Line':
            axis = plot_labour_graph(axis,region,Scenarios)
        elif plot_type == 'Wedge':
            axis = plot_labour_wedges(axis,region,scenario)
        if plot_type == 'Productivity':
            axis = plot_productivity_graph(axis,region,Scenarios)

    if plot_capacity:
        c_axes = [c_axis1,c_axis2,c_axis3,c_axis4]
        xlabels = [False,True,False,True]
        for axis, region,xlabel in zip(c_axes,side_reg,xlabels):
            axis = plot_mines_capacity(region,axis,xlabel=xlabel)

    if show_uncertainty:
        bound_2023 = {'China':[1.87e3,2.80e3],
                      'India':[0.47e3,1e3]}[country]
        r_axis1.plot([2020,2020],bound_2023,color='grey',alpha=0.5,linewidth=10,zorder=-1)

    if savefig:
        fig.savefig(f'figures\P{ind}_Map_{plot_type}_{country}_{scenario}.svg', bbox_inches='tight',dpi=150)

    return ax

#%%
def plot_summary(plot_type,scenario='NZ_CCS1',plot_capacity=False,show_uncertainty=False):
    Scenarios = ['NDC_CCS1','NZ_CCS1','NPI','NDC','NZ']
    Scenarios_names = ['NDC-LTT w/CCS','1.5°C w/CCS','NPi','NDC-LTT','1.5°C']


    crs = ccrs.AlbersEqualArea(central_longitude=100, central_latitude=30)



    T1 = 2050
    fig, ax = plt.subplots(1,1,subplot_kw={"projection": crs}, figsize=(15, 12))
                            

    zlim = [0,300]




    vmin, vmax = zlim[0], zlim[1]
    colormap = plt.cm.OrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
        
    highlight_regions = ['Jharkhand','Shanxi']


    var = 'Employment|Coal|Downscaled'
    t1= 2020
    ax, cbar = map_employment(ax,var,t1,scenario,zlim,colormap,crs,norm,highlight_regions=highlight_regions)


    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])  # Required for colorbar creation

    # Create an axis for the colorbar (customize position [left, bottom, width, height])
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])  # Adjust position as needed
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='max')  # Arrow on right
    cbar_ax.set_xlabel('Coal workforce [thousand workers]', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax.axis('off')


    side_reg  = ['China','India']

    if plot_capacity:
        r_axis1 = ax.inset_axes([1.45, 0.55, 0.50, 0.6])
        r_axis2 = ax.inset_axes([1.45, -0.3, 0.50, 0.6])

        c_axis1 = ax.inset_axes([1.15, 0.55 , 0.15, 0.7])
        c_axis2 = ax.inset_axes([1.15, -0.3, 0.15, 0.7])

    else:
        r_axis1 = ax.inset_axes([1.15, 0.7, 0.7, 0.6])
        r_axis2 = ax.inset_axes([1.15, -0.2, 0.7, 0.6])

    side_axes = [r_axis1,r_axis2]
    for axis, region in zip(side_axes,side_reg):
        if plot_type == 'Line':
            axis = plot_labour_graph(axis,region,Scenarios)
        elif plot_type == 'Wedge':
            axis = plot_labour_wedges(axis,region,scenario,position='left',T1=T1)
            axis.set_ylim([-200,4000])

        if plot_type == 'Productivity':
            axis = plot_productivity_graph(axis,region,Scenarios)

    if plot_capacity:
        c_axes = [c_axis1,c_axis2]
        xlabels = [False,True]
        for axis, region,xlabel in zip(c_axes,side_reg,xlabels):
            axis = plot_mines_capacity(region,axis,xlabel=xlabel)


    return ax

plot_summary('Wedge')

#%%

if __name__ == "__main__":

    plot_main('Line','China',ind='1')
    plot_main('Line','India',ind='2')

    plot_main('Wedge','China',scenario='NPI',ind='3')
    plot_main('Wedge','India',scenario='NPI',ind='4')

    plot_main('Wedge','China',scenario='NZ',ind='5')
    plot_main('Wedge','India',scenario='NZ',ind='6')
    plot_main('Wedge','India',scenario='NZ_CCS1',ind='7')


# %%
