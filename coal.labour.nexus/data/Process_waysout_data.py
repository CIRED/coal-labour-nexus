#%%     Load libraries

import pandas as pd
import numpy as np
from datetime import date
import os

#%%     Load data
input_path = os.path.join('/data','public_data','WaysOut_China_and_India_downscaling_data')
input_file = os.path.join(input_path,'Database.csv')


Data = pd.read_csv(input_file)
output_path = os.path.join("Coal_labour","Downscaling")
os.makedirs(output_path, exist_ok=True)
#%%
China_regions = [x for x in Data[Data['Country']=="China"]['Region'].unique() if x != 'China']
India_regions = [x for x in Data[Data['Country']=="India"]['Region'].unique() if x not in ['India','Ladakh','Daman & Diu','Dadra & Nagar Haveli','Lakshadweep']] #Neglecting a number of smaller states/cities with inconsistent data and no coal production, this will not affect results

#%% Share function
def calculate_share(database, variable_type, country):
    if country == "China":
        regions = China_regions
    elif country == "India":
        regions = India_regions
    else:
        return pd.DataFrame()  # Return an empty dataframe if the country is not supported

    filtered_data = database[(database['Type'] == variable_type) & (database['Country'] == country) & (~database['Region'].isin(['China','India']))]
    region_data = []

    for region in regions:
        # Calculate the total value for the current region
        region_value = filtered_data[filtered_data['Region'] == region]['Value'].sum()
        region_share = region_value / filtered_data['Value'].sum()
        region_data.append({'Region': region, 'Share': region_share})
        

    total_value = filtered_data['Value'].sum()
    total_share = total_value / filtered_data['Value'].sum()
    region_data.append({'Region': country, 'Share': total_share}) #'Value': total_value, 

    return pd.DataFrame(region_data)

def calc_productivity(df,Prod,productivity_rate):
    year = df.Year.values[-1]
    value= df.Value.values[-1]
    # Start by calculating production 
    calc = pd.DataFrame(columns=[x for x in range(2003,2024)])
    calc.loc['product',year] = Prod[year].values[0]/value
    calc.loc['Emp',year]=value
    
    for t in range(2003,2023): #Forward productivity calculations
        calc.loc['product',t] = calc.loc['product',year] * (1+productivity_rate*1e-2)**(t-year)
        calc.loc['Emp',t] = Prod[t].values[0]/calc.loc['product',t]
    return calc

#%% Unemployment China
# Assume that the unemployment rate is the same for urban and rural areas
Urban_unemployment_rate = Data[(Data['Type']=='Urban unemployed rate') & (Data['Year']==2021)& (Data['Region']!='China')]
Total_employed_people = Data[(Data['Type']=='Employed Persons') & (Data['Year']==2021)& (Data['Region']!='China')]


Unemployed_people_china = Total_employed_people.merge(Urban_unemployment_rate, on='Region', suffixes=('_employed', '_unemployed'))
Unemployed_people_china['Value'] = Unemployed_people_china['Value_employed'] * (1 / (1-Unemployed_people_china['Value_unemployed']/100)-1)
Unemployed_people_china = Unemployed_people_china[['Region', 'Value']]

total_row = pd.DataFrame({'Region': ['China'], 'Value': [Unemployed_people_china['Value'].sum()]})

Unemployed_people_china = Unemployed_people_china.append(total_row, ignore_index=True)

Unemployed_people_china['Share'] = Unemployed_people_china['Value'] / Unemployed_people_china[Unemployed_people_china['Region']=='China']['Value'].values[0]

# %% Unemployment India force participation rate of those regions with unavailable data is equal to the average of the other regions
# Here we assume that the labo
LFPR = Data[(Data['Type']=='Labour force participation rate') & (Data['Year']==2019)& (Data['Region']!='India')]
UR = Data[(Data['Type']=='Unemployment rate') & (Data['Year']==2019)& (Data['Region']!='India')]

# For population, note that population Telangana which left Andhra Pradesh after the 2011 census, has been separated
Pop = Data[(Data['Type']=='Population') & (Data['Year']==2011)& (Data['Region']!='India')]

# Drop every column from LFPR except Region and Value
LFPR = LFPR.drop(list(set(LFPR.columns)-set(['Region','Value'])),axis=1)

mean_LFPR = LFPR['Value'].mean()
for reg in list(set(Pop['Region'])-set(LFPR['Region'])):
    LFPR = LFPR.append(pd.DataFrame({'Region': reg, 'Value':[mean_LFPR]}), ignore_index=True)


Unemployed_people_india = LFPR.merge(UR, on='Region', suffixes=('_lfpr', '_ur'))
Unemployed_people_india= Unemployed_people_india.merge(Pop, on='Region')


Unemployed_people_india['Value'] = Unemployed_people_india['Value'] * Unemployed_people_india['Value_lfpr'] * Unemployed_people_india['Value_ur'] / 100

Unemployed_people_india = Unemployed_people_india[['Region', 'Value']]

total_row = pd.DataFrame({'Region': ['India'], 'Value': [Unemployed_people_india['Value'].sum()]})

Unemployed_people_india = Unemployed_people_india.append(total_row, ignore_index=True)

Unemployed_people_india['Share'] = Unemployed_people_india['Value'] / Unemployed_people_india[Unemployed_people_india['Region']=='India']['Value'].values[0]
#%% ====================== Productivity increase China ======================

CE_SY = Data[(Data['Type']=='Coal Employment')&(Data['Source'].apply(lambda x: ' '.join(x.split()[:3]) == 'China statistical yearbook'))&(Data['Region']=='China')]
CE_SLY = Data[(Data['Type']=='Coal Employment')&(Data['Source'].apply(lambda x: ' '.join(x.split()[:4]) == 'China Labour Statistical Yearbook'))&(Data['Region']=='China')]
CQ_SY = Data[(Data['Type']=='Coal Production (Mt)')&(Data['Region']=='China')]

prod = CE_SY.merge(CQ_SY, on='Year', suffixes=('_L', '_Q'))
prod['Value'] = np.log(prod['Value_Q'] / prod['Value_L'])

# calculating 2015-2022 increase rate - what we want is just CAGR
CAGR = (np.exp(prod['Value'][prod['Year']==2022].values[0]) / np.exp(prod['Value'][prod['Year']==2015].values[0])) ** (1/7) - 1



# %%
# Data econ_structure
# Coal - coal production
# Oil  - crude oil production
# Gas  - natural gas production
# ET   - oil and gas
# Electricity - electricity production in China, GDP from electricity in India
# Construction - construction
# Services - in China: GDP from tertiary industry, in India: sum: administration, agriculture, finance, real estate, trade and hotels
# Air transport - GDP - for India, we do not consider 'transport' directly as we do not have information for its composition
# Water transport - GDP
# Other transport - GDP
# Agriculture - agriculture
# Industry - industry


# China
Column_names = ['Coal', 'Coal_GEM', 'Oil', 'Gas', 'Electricity', 'Construction', 'Services', 'Air transport', 'Water transport', 'Terrestrial transport', 'Agriculture', 'Industry']
type_china = ['Coal output','Coal output','Crude Oil output','Natural gas output','Electricity Generation','GDP from construction','GDP from tertiary industry','GDP','GDP','GDP','GDP from agriculture','GDP from industry']
sources = ['China statistical yearbook 2023','Global Energy Monitor (2023) Global Coal Mine Tracker']+['China statistical yearbook 2017']+['China statistical yearbook 2023']*9 #Oil output last accessible in 2017



data_china = pd.DataFrame()
data_china['Region'] = ['China'] +China_regions 

for ind_col in range(len(Column_names)):
    data_china = data_china.merge(calculate_share(Data[Data['Source']==sources[ind_col]], type_china[ind_col], 'China'), on='Region', suffixes=('', Column_names[ind_col])) 

data_china.columns = ['Region']+Column_names

data_china['ET'] = (data_china['Oil'] + data_china['Gas'])/2
# %%
# India
Column_names = ['Coal', 'Coal_GEM', 'Oil', 'Gas', 'Electricity', 'Construction', 'Agriculture', 'Industry']
type_india = ['Coal output','Coal output','Oil output','Gas output','GDP from electricity','GDP from construction','GDP from agriculture','GDP from manufacturing']
sources = ['Pai and Zerriffi (2021), "Indian coal mine location and production - December 2020"','Global Energy Monitor (2023) Global Coal Mine Tracker']+['India Ministry of Petroleum and Natural Gas - Annual Report 2022-23']*2+['Handbook Of Statistics On Indian Economy 2023']*4
years = [2019,2022]+[2021]*6

data_india = pd.DataFrame()
data_india['Region'] = ['India'] +India_regions

for ind_col in range(len(Column_names)):
    data_india = data_india.merge(calculate_share(Data[(Data['Year']==years[ind_col])&(Data['Source']==sources[ind_col])], type_india[ind_col], 'India'), on='Region', suffixes=('', Column_names[ind_col])) 
    
data_india.columns = ['Region']+Column_names

#%% calculating total GDP by regions
variables = ['GDP from administration',
'GDP from agriculture',
'GDP from construction',
'GDP from electricity',
'GDP from finance',
'GDP from manufacturing',
'GDP from mining',
'GDP from other',
'GDP from real estate',
'GDP from trade and hotels',
'GDP from transport']


variables_services = ['GDP from administration',
'GDP from finance',
'GDP from other',
'GDP from real estate',
'GDP from trade and hotels',
'GDP from transport']
                      

year = 2021

for reg in India_regions:
    data_india.loc[data_india['Region']==reg, 'GDP'] = Data[(Data['Type'].isin(variables)) & (Data['Region']==reg)& (Data['Year']==year)]['Value'].sum()
    data_india.loc[data_india['Region']==reg, 'GDP_services'] = Data[(Data['Type'].isin(variables_services)) & (Data['Region']==reg)& (Data['Year']==year)]['Value'].sum()
# data_china['ShareOil']=data_china['Share']

data_india.loc[data_india['Region']=='India', 'GDP'] = data_india['GDP'].sum()
data_india.loc[data_india['Region']=='India', 'GDP_services'] = data_india['GDP_services'].sum()

data_india['Services'] = data_india['GDP_services'] / data_india[data_india['Region']!='India']['GDP_services'].sum()

data_india['Air transport'] = data_india['GDP'] / data_india[data_india['Region']!='India']['GDP'].sum() 
data_india['Water transport'] = data_india['GDP'] / data_india[data_india['Region']!='India']['GDP'].sum() 
data_india['Terrestrial transport'] = data_india['GDP'] / data_india[data_india['Region']!='India']['GDP'].sum() 


data_india['ET'] = (data_india['Oil'] + data_india['Gas'])/2

#%% Aggregating data
data_standard = pd.DataFrame(columns = ['Region_code','Subregion_code','Region_name','Subregion_name','Coal','Oil','Gas','ET','Electricity','Construction','Services','Air transport','Water transport','Terrestrial transport','Agriculture','Industry','Unemployment'])
data_standard['Subregion_name'] = ['China'] + China_regions + ['India'] + India_regions

data_GEM = data_standard.copy()

countries = ['CHN', 'IND']
for c_index in [0,1]:
    country = countries[c_index]
    cdata = [data_china, data_india][c_index]

    udata = [Unemployed_people_china, Unemployed_people_india][c_index]

    regs = [['China']+China_regions,['India']+India_regions][c_index]
    for ind_reg in range(len(regs)):
        reg = regs[ind_reg]
        data_standard.loc[data_standard['Subregion_name']==reg,'Region_name'] = country
        for sec in data_standard.columns[4:-1]:
            data_standard.loc[data_standard['Subregion_name']==reg,sec] = cdata[cdata['Region']==reg][sec].values[0]
        data_standard.loc[data_standard['Subregion_name']==reg,'Unemployment'] = udata[udata['Region']==reg]['Share'].values[0]
        data_standard.loc[data_standard['Subregion_name']==reg,'Region_code'] = [6,7][c_index]
        data_standard.loc[data_standard['Subregion_name']==reg,'Subregion_code'] = ind_reg

        data_GEM.loc[data_GEM['Subregion_name']==reg,'Region_name'] = country
        data_GEM.loc[data_GEM['Subregion_name']==reg,'Coal'] = cdata[cdata['Region']==reg]['Coal_GEM'].values[0]
        for sec in data_GEM.columns[4:-1]:
            data_GEM.loc[data_GEM['Subregion_name']==reg,sec] = cdata[cdata['Region']==reg][sec].values[0]
        data_GEM.loc[data_GEM['Subregion_name']==reg,'Unemployment'] = udata[udata['Region']==reg]['Share'].values[0]
        data_GEM.loc[data_GEM['Subregion_name']==reg,'Region_code'] = [6,7][c_index]
        data_GEM.loc[data_GEM['Subregion_name']==reg,'Subregion_code'] = ind_reg

data_standard.to_csv(os.path.join(output_path,'Econ_structure.csv'),index=False)
data_GEM.to_csv(os.path.join(output_path,'Econ_structure_GEM.csv'),index=False)
data_standard.drop(
    data_standard.columns[~data_standard.columns.isin(
        ['Region_code','Subregion_code','Region_name','Subregion_name']
        )],axis=1).to_csv(os.path.join(output_path,'Indexes.csv'),index=False)

# %%===============================================================================================
# ===============================================================================================
#                             Coal employment
# ===============================================================================================
# ===============================================================================================



# China
# For China, we take 2015 total coal employment from China Statistical Yearbook
# We assume it is distributed in the same way as given in the China Labour Statistical Yearbook
# We use the absolute value of the former source as they are not constrained to urban workers

# taking 2015 sum

#=============== Mid
tot_china_mid = Data[(Data['Type']=='Coal Employment') & 
                 (Data['Year']==2015) & 
                 (Data['Region']=='China')&
                 (Data['Source']=='China Labour Statistical Yearbook 2016')]['Value'].values[0]

#=============== High
tot_china_max = Data[(Data['Type']=='Coal Employment') & 
                 (Data['Year']==2015) & 
                 (Data['Region']=='China')&
                 (Data['Source']=='China statistical yearbook 2016')]['Value'].values[0]

#=============== Low
tot_china_min = Data[(Data['Type']=='Coal Employment') & 
                 (Data['Region']=='China')&
                 (Data['Source'].str.contains('IEA'))&
                 (Data['Note'].str.contains("Direct calculated"))]
Prod=Data[(Data.Region=='China')&(Data.Type=="Coal Production (Mt)")].pivot_table(values='Value',columns='Year')
calc= calc_productivity(tot_china_min,Prod,10.78)
tot_china_min = calc.loc['Emp',2015]

l_china = calculate_share(Data[Data['Year']==2018],'Coal Employment','China')
l_china = l_china.merge(data_china[['Region','Coal']], on='Region')
l_china.loc[l_china['Coal']==0,'Share'] = 0

for ind_scenario,scenario in enumerate(['mid','min','max']):
    l_china.loc[:,scenario] = l_china.loc[:,'Share'] * [tot_china_mid,tot_china_min,tot_china_max][ind_scenario]
    # Removing regions with no coal production
    l_china.loc[l_china['Region']=='China',scenario] = l_china[l_china['Region']!='China'][scenario].sum()



#%% India
# For India, we take 2015 value from previous Imaclim|GTAP|ILO calibration
# We then distribute this according to the values found by Pai and Zerriffi (2020)
# This assumes the 1/3 direct to indirect job ratio is constant throughout the country

Prod=Data[(Data.Region=='India')&(Data.Type=="Coal Production (Mt)")].pivot_table(values='Value',columns='Year')

tot_india_mid = Data[(Data['Source']=='Imaclim-GTAP-ILO')&
                 (Data['Region']=='India')]['Value'].values[0]

tot_india_min = Data[(Data['Source'].str.contains('Dsouza and Singhal'))&
                 (Data['Note'].str.contains('Scenario 1'))]

tot_india_max = Data[(Data['Source'].str.contains('Dsouza and Singhal'))&
                 (Data['Note'].str.contains('Scenario 3'))]

tot_india_min= calc_productivity(tot_india_min,Prod,10.78).loc['Emp',2015]
tot_india_max= calc_productivity(tot_india_max,Prod,10.78).loc['Emp',2015]


# Taking distribution from Pai and Zerriffi (2020)
l_india = calculate_share(Data[Data['Year']==2021],'Coal employment','India')

for ind_scenario,scenario in enumerate(['mid','min','max']):
    l_india[scenario] = l_india['Share'] * [tot_india_mid,tot_india_min,tot_india_max][ind_scenario]



# %%
# For both both countries, estimating the 2015 employment from the GEM database involves the following steps
# 1. Take the 2022 sum of employment over all subnational regions from the database
# 2. Calculate the 2022 labour productivity by comparing the 2022 coal production in Mt from the Energy Institute
# 3. Find the 2015 labour productivity assuming a growth rate of 10.59% per year in China and 5.8% in India
# 4. Calculate the 2015 employment by dividing the 2015 coal production by the 2015 labour productivity
# 5. Distribute the 2015 employment according to the 2022 GEM distribution in both countries

l_GEM = pd.DataFrame(columns = ['Region_code','Subregion_code','Region_name','Subregion_name','2015'])
l_GEM['Subregion_name'] = ['China'] + China_regions + ['India'] + India_regions

countries = ['China', 'India']
for c_index in [0,1]:
    country = countries[c_index]

    emp_2022   = Data[(Data['Type']=='Coal employment') & (Data['Year']==2022) & (Data['Region']==country) & (Data['Source']=='Global Energy Monitor (2023) Global Coal Mine Tracker')]['Value'].values[0]
    prod_2022  = Data[(Data['Type']=='Coal Production (Mt)') & (Data['Year']==2022) & (Data['Region']==country) & (Data['Source']=='Energy Institute Statistical Review of World Energy')]['Value'].values[0]
    produ_2022 = prod_2022 / emp_2022 # Mt/worker
    produ_2015 = produ_2022 / ((1 + [10.49e-2,5.8e-2][c_index])**(2022-2015)) # Mt/worker 
    
    emp_2015 = Data[(Data['Type']=='Coal Production (Mt)') & (Data['Year']==2015) & (Data['Region']==country) & (Data['Source']=='Energy Institute Statistical Review of World Energy')]['Value'].values[0] / produ_2015
    
    share_l = calculate_share(Data[(Data['Year']==2022)&(Data['Source']=='Global Energy Monitor (2023) Global Coal Mine Tracker')],'Coal employment',country)

    share_l['Value'] = share_l['Share'] * emp_2015

    regs = [['China']+China_regions,['India']+India_regions][c_index]
    for ind_reg in range(len(regs)):
        reg = regs[ind_reg]
        l_GEM.loc[l_GEM['Subregion_name']==reg,'Region_code'] = [6,7][c_index]
        l_GEM.loc[l_GEM['Subregion_name']==reg,'Subregion_code'] = ind_reg
        l_GEM.loc[l_GEM['Subregion_name']==reg,'Region_name'] = ['CHN', 'IND'][c_index]
        l_GEM.loc[l_GEM['Subregion_name']==reg,'2015'] = share_l[share_l['Region']==reg]['Value'].values[0]

l_GEM.to_csv(os.path.join(output_path,'Coal_jobs_2015_2021_GEM.csv'),index=False)
#%% Aggregating coal employment data
l_standard = pd.DataFrame(columns = ['Region_code','Subregion_code','Region_name','Subregion_name','mid','min','max'])
l_standard['Subregion_name'] = ['China'] + China_regions + ['India'] + India_regions

for ind_scenario, scenario in enumerate(['mid','min','max']):
    countries = ['CHN', 'IND']
    for c_index in [0,1]:
        country = countries[c_index]
        cdata = [l_china, l_india][c_index]

        regs = [['China']+China_regions,['India']+India_regions][c_index]
        for ind_reg in range(len(regs)):
            reg = regs[ind_reg]
            l_standard.loc[l_standard['Subregion_name']==reg,'Region_code'] = [6,7][c_index]
            l_standard.loc[l_standard['Subregion_name']==reg,'Subregion_code'] = ind_reg
            l_standard.loc[l_standard['Subregion_name']==reg,'Region_name'] = country
            l_standard.loc[l_standard['Subregion_name']==reg,scenario] = cdata[cdata['Region']==reg][scenario].values[0]

l_standard.to_csv(os.path.join(output_path,'Coal_jobs_2015.csv'),index=False)
#%% Population distribution

pop_india = pd.DataFrame()
pop_india['Region'] = ['India'] +India_regions
pop_india = pop_india.merge(calculate_share(Data[(Data['Year']==2011)],'Population','India'), on='Region', suffixes=('', '2011'))

pop_china = pd.DataFrame()
pop_china['Region'] = ['China'] +China_regions
pop_china = pop_china.merge(calculate_share(Data[(Data['Year']==2022)],'Population','China'), on='Region', suffixes=('', '2010'))


sectors = ['Electricity','Construction','Services','Air transport','Water transport','Terrestrial transport']

data_pop = data_standard.copy()

countries = ['CHN', 'IND']
for c_index in [0,1]:
    country = countries[c_index]
    cdata = [pop_china, pop_india][c_index]

    regs = [['China']+China_regions,['India']+India_regions][c_index]
    for ind_reg in range(len(regs)):
        reg = regs[ind_reg]
        for sec in sectors:
            data_pop.loc[data_standard['Subregion_name']==reg,sec] = cdata[cdata['Region']==reg]['Share'].values[0]

data_pop.to_csv(os.path.join(output_path,'Econ_structure_pop.csv'),index=False)
# %%
