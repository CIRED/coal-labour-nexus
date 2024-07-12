# This is a treatment code to build the necessary demographic data necessary to run the Imaclim coal labour nexus 
#=================================================================================================================================================================================================================
#%% Importing libraries
import pandas as pd
import numpy as np
import os
# %%
#=================================================================================================================================================================================================================
regions = ['China','India']
LF_age = [20,59]

output_path = os.path.join("Coal_labour","SSP")

# %% Reading UN data
#=================================================================================================================================================================================================================
UN_data = pd.read_csv( "/data/public_data/UNO_world_population_prospect/normalized/WPP2022_PopulationExposureBySingleAgeSex_Medium_1950-2021.csv", delimiter='|', encoding="utf-8")

UN_data = UN_data[(UN_data['Location'].isin(regions))&(UN_data['Time'] ==2015)]

UN_data_china = UN_data[UN_data['Location']=='China'].drop(columns=[x for x in UN_data.columns if x not in ['AgeGrp','PopTotal']])
UN_data_china.columns = ['Year',2015]

UN_data_china.to_csv(os.path.join(output_path,'Age_pyramid_China.csv'),index=False)


UN_data_india = UN_data[UN_data['Location']=='India'].drop(columns=[x for x in UN_data.columns if x not in ['AgeGrp','PopTotal']])
UN_data_india.columns = ['Year',2015]

UN_data_india.to_csv(os.path.join(output_path,'Age_pyramid_India.csv'),index=False)


dataPath='/data/public_data/IIASA_scenario_explorer_public/download/'
ssp_pop = pd.read_csv(dataPath + 'ssp__all.csv', delimiter=',', encoding="utf-8")
ssp_pop = ssp_pop[ssp_pop.Region.isin(regions)]


# %%
#=================================================================================================================================================================================================================
scenario = 'SSP2'
region = regions[0]
age_range =  ['Age 0-4',
              'Age 5-9',
              'Age 10-14',
              'Age 15-19',
              'Age 20-24',
              'Age 25-29',
              'Age 30-34',
              'Age 35-39',
              'Age 40-44',
              'Age 45-49',
              'Age 50-54',
              'Age 55-59',
              'Age 60-64',
              'Age 65-69',
              'Age 70-74',
              'Age 75-79',
              'Age 80-84',
              'Age 85-89',
              'Age 90-94',
              'Age 95-99',
              'Age 100+']

#%% Linear interpolation of SSP projections
#=================================================================================================================================================================================================================
ages = range(0,101)
pop_age = pd.DataFrame(columns=['Country','Scenario']+[str(i) for i in range(2015,2101)])
for scenario in ['SSP'+str(x) for x in range(1,6)]:
    for region in regions:
        pop_china = {}
        pop_china
        for i in range(0, len(age_range)):
            pop_china['Male|'+age_range[i]] = [ssp_pop[(ssp_pop.Region==region)&(ssp_pop.Scenario==scenario)&(ssp_pop.Variable=='Population|Male|'+age_range[i])][str(x)].values[0] for x  in range(2020,2101,5)]
            pop_china['Female|'+age_range[i]] = [ssp_pop[(ssp_pop.Region==region)&(ssp_pop.Scenario==scenario)&(ssp_pop.Variable=='Population|Female|'+age_range[i])][str(x)].values[0] for x  in range(2020,2101,5)]
            pop_china['Total|'+age_range[i]] = np.array(pop_china['Male|'+age_range[i]]) + np.array(pop_china['Female|'+age_range[i]])

        pop_china = pd.DataFrame.from_dict(data=pop_china,columns=[str(i) for i in range(2020,2101,5)], orient='index')

        for i in range(0, len(age_range)):
            pop_china.loc['Male|'+age_range[i],'2015'] = ssp_pop[(ssp_pop.Region==region)&(ssp_pop.Scenario=='Historical Reference')&(ssp_pop.Variable=='Population|Male|'+age_range[i])]['2015'].values[0]
            pop_china.loc['Female|'+age_range[i],'2015'] = ssp_pop[(ssp_pop.Region==region)&(ssp_pop.Scenario=='Historical Reference')&(ssp_pop.Variable=='Population|Female|'+age_range[i])]['2015'].values[0]
            pop_china.loc['Total|'+age_range[i],'2015'] =pop_china.loc['Male|'+age_range[i],'2015']+pop_china.loc['Female|'+age_range[i],'2015']
            

        pop_china.loc['Male'] = pop_china.loc[['Male' in x for x in pop_china.index]].sum()
        pop_china.loc['Female'] = pop_china.loc[['Female' in x for x in pop_china.index]].sum()
        pop_china.loc['Total'] = pop_china.loc[['Total' in x for x in pop_china.index]].sum()


        interpolated_years = [i for i in range(2015,2101) if not str(i) in pop_china.columns]

        # Reusing Imaclim data treatment code to linearly interpolate missing years
        pop_china.columns = pop_china.columns.astype(int)

        for year in interpolated_years:
            pop_china[year] = np.nan
        pop_china = pop_china.reindex(sorted(pop_china.columns), axis=1)

        pop_china = pop_china.interpolate(method='linear', axis=1, limit_area=None, limit_direction='both', columns=interpolated_years)

        
        pop_china_age = pd.DataFrame(index=ages, columns=pop_age.columns)

        for age in ages[:-1]:
            pop_china_age.loc[age].iloc[2:] = pop_china.loc['Total|Age '+str(int(np.floor(age/5)*5))+'-'+str(int(np.floor(age/5)*5+4))].values/5
        pop_china_age.loc[100].iloc[2:] = pop_china.loc['Total|Age 100+'].values

        pop_china_age.loc['LF',2:] = pop_china_age.loc[[x for x in range(LF_age[0],LF_age[1]+1)]].sum() 
        pop_china_age.loc['tx_entry',2:] = pop_china_age.loc[LF_age[0]]/pop_china_age.loc['LF']
        pop_china_age.loc['tx_exit',2:] = pop_china_age.loc[LF_age[1]+1]/pop_china_age.loc['LF']
        pop_china_age.loc[:,'Country'] = region
        pop_china_age.loc[:,'Scenario'] = scenario
        pop_age = pop_age.append(pop_china_age)


for i_s, scenario in enumerate(['SSP'+str(x) for x in range(1,6)]):
    txEntry = pd.DataFrame(index=range(0,12),columns=range(2015,2101))
    txEntry[txEntry.isna()]=1

    txExit = txEntry.copy()

    for i_r, region in enumerate(regions):
        txEntry.loc[5+i_r] = pop_age.loc[(pop_age.Country==region)&(pop_age.Scenario==scenario)].loc["tx_entry"].values[2:]
        txExit.loc[5+i_r]  = pop_age.loc[(pop_age.Country==region)&(pop_age.Scenario==scenario)].loc["tx_exit"].values[2:]

    txEntry.to_csv(os.path.join(output_path,'txEntry_'+scenario+'.csv'),index=False,header=False)
    txExit.to_csv(os.path.join(output_path,'txExit_'+scenario+'.csv'),index=False,header=False)
# %%
