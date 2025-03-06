// This file is used to run the analysis of the evolution of coal labour in several IAM scenarios
// It initializes key parameters for the runs and treats output data

//===============================================================================
// Preamble

date=getdate();
date= string(date(1))+string(date(2))+string(date(6));

SAVEDIR = '../output/'+date+'_Coal_labour_results'; mkdir(SAVEDIR);

diary(SAVEDIR+"/"+date+"_summary.log");
fileID = mopen(SAVEDIR+'/'+date+"_log.txt",'w') 
time=getdate();
mfprintf(fileID, "Time: "+string(time(7))+":"+string(time(8))+":"+string(time(9))+"\n");


//======================================================================================================================================================
// Import data (including coal jobs)

Region_code = csvRead('../data/Coal_labour/Downscaling/Indexes.csv',',',[],'string');
Country_names = Region_code(2:$,3);
Region_code = Region_code(2:$,4);
NRegions = size(Region_code,1);

age_pyramid.china = csvRead('../data/Coal_labour/SSP/Age_pyramid_China.csv');
age_pyramid.india = csvRead('../data/Coal_labour/SSP/Age_pyramid_India.csv');

//===============================================================================
// Import functions
exec("functions.sci",-1)


//===============================================================================
// Copying Imaclim variables to ease transition
exec("imaclim.variables.sce");

txEntr = csvRead('../data/Coal_labour/SSP/txEntry_SSP2.csv',',','.',[]);
txExit = csvRead('../data/Coal_labour/SSP/txExit_SSP2.csv',',','.',[]);
txLact = csvRead('../data/Coal_labour/SSP/txLact_SSP2.csv',',','.',[])

//===============================================================================
// Defining Scenarios

// IAM trajectory source - pathways are assumed to be stored in *.csv file containing a single scenario from single model with header of the form: [Model, Scenario, Region, Variable, Unit, 2015, ...] 
NPI_CCS0 = 'IMACLIM_waysout_outputs_WO-NPi-ElecIndus-CCS0';
NDC_CCS0 = 'IMACLIM_waysout_outputs_WO-NDCLTT-ElecIndus-CCS0';
NZ_CCS0  = 'IMACLIM_waysout_outputs_WO-15C-ElecIndus-CCS0';
NDC_CCS1 = 'IMACLIM_waysout_outputs_WO-NDCLTT-ElecIndus-CCS1';
NZ_CCS1  = 'IMACLIM_waysout_outputs_WO-15C-ElecIndus-CCS1';



// 2025-03-05 Productivity SI
scenario_names = ['NPI','NDC','NZ','NDC_CCS1','NZ_CCS1',...
                  'NPI_PG0','NDC_PG0','NZ_PG0','NDC_CCS1_PG0','NZ_CCS1_PG0',...
                  'NPI_R55','NDC_R55','NZ_R55','NDC_CCS1_R55','NZ_CCS1_R55',...
                  'NPI_min','NDC_min','NZ_min','NDC_CCS1_min','NZ_CCS1_min',...
                  'NPI_max','NDC_max','NZ_max','NDC_CCS1_max','NZ_CCS1_max',...
                  'NPI_C0I0_P0','NDC_C0I0_P0','NZ_C0I0_P0','NDC_CCS1_C0I0_P0','NZ_CCS1_C0I0_P0',...
                  'NPI_C0I0_P100','NDC_C0I0_P100','NZ_C0I0_P100','NDC_CCS1_C0I0_P100','NZ_CCS1_C0I0_P100',...
                  'NPI_C20I5_P0','NDC_C20I5_P0','NZ_C20I5_P0','NDC_CCS1_C20I5_P0','NZ_CCS1_C20I5_P0',...
                  'NPI_C20I5_P100','NDC_C20I5_P100','NZ_C20I5_P100','NDC_CCS1_C20I5_P100','NZ_CCS1_C20I5_P100',...
                  'NPI_C40I10_P0','NDC_C40I10_P0','NZ_C40I10_P0','NDC_CCS1_C40I10_P0','NZ_CCS1_C40I10_P0',...
                  'NPI_C40I10_P100','NDC_C40I10_P100','NZ_C40I10_P100','NDC_CCS1_C40I10_P100','NZ_CCS1_C40I10_P100']

scenario_source= repmat([NPI_CCS0,NDC_CCS0,NZ_CCS0,NDC_CCS1,NZ_CCS1],1,11);
Scenario_types = repmat(['NPi_CCS0','NDC_CCS0','1.5C_CCS0','NDC_CCS1','1.5C_CCS1'],1,11);
productivity_source = ['C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100',...
                        'C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100',...
                        'C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100',...
                        'C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100',...
                        'C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100',...
                        'C0I0_P0','C0I0_P0','C0I0_P0','C0I0_P0','C0I0_P0',...
                        'C0I0_P100','C0I0_P100','C0I0_P100','C0I0_P100','C0I0_P100',...
                        'C20I5_P0','C20I5_P0','C20I5_P0','C20I5_P0','C20I5_P0',...
                        'C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100','C20I5_P100',...
                        'C40I10_P0','C40I10_P0','C40I10_P0','C40I10_P0','C40I10_P0',...
                        'C40I10_P100','C40I10_P100','C40I10_P100','C40I10_P100','C40I10_P100',]



// Initialising parameters with default values
// -  Retirement age
S_retirement_age = 60*ones(size(scenario_names,1),size(scenario_names,2));
// -  Productivity growth
S_productivity_growth = 5.8*ones(size(scenario_names,1),size(scenario_names,2));
// -  Data source - 1: default 2:GEM
S_ind_gem = ones(size(scenario_names,1),size(scenario_names,2));
// -  Structural change - 1: default, 2: Convergence to population share, 3: Structural Change, 4: Constant with DOSE calibration, 5: Convergence with DOSE calibration
S_ind_struct = ones(size(scenario_names,1),size(scenario_names,2)); 
// -  Additional parameters
S_ind_unemployment = ones(size(scenario_names,1),size(scenario_names,2)); // 1: competition against other unemployed people for vacancies, 2: unemployment not considered
S_ind_entry_leave = ones(size(scenario_names,1),size(scenario_names,2)); // 1: Retiring workers create new vacancies, 2: they do not
// -  Ouputing treated variables to infill IAM scenarios with missing variables
S_ind_preparing_infilling = zeros(size(scenario_names,1),size(scenario_names,2));
// -  Workforce - 1: mid, 2: min, 3: max
S_ind_workforce = ones(size(scenario_names,1),size(scenario_names,2));


// Alternative parameters
// S_ind_gem(4:6) = 2;
// S_productivity_growth(7:9) = 0;
// S_retirement_age(10:12) = 55;
// S_ind_workforce(16:18) = 2;
// S_ind_workforce(19:21) = 3;

// S_ind_struct(4:6)=2;
// S_ind_struct(7:9)=3;
// S_ind_struct(10:12)=4;
// S_ind_struct(13:15)=5;

//1- S_ind_preparing_infilling(1:3)=1;

// Basic SI
S_productivity_growth(6:10) = 0;
S_retirement_age(11:15) = 55;
S_ind_workforce(16:20) = 2;
S_ind_workforce(21:25) = 3;


// Load Productivity path calculated with the productivity nexus

// Inputing productivity path
Prod_path = csvRead('../productivity_trajectories/coalregion_SensitivityAnalysis_V1.csv",',','[]',"string");
Prod_path = strsubst(Prod_path, "NA", "0"); 

// separating from metadata
Prod_header  = Prod_path(1,:);
Prod_T = Prod_header(10:$);
Produ  = evstr(Prod_path(2:$,10:$));
Pr_scenario  = Prod_path(2:$,1);
Pr_prscenario  = Prod_path(2:$,2)+'_'+Prod_path(2:$,3);
Pr_regi_c    = evstr(Prod_path(2:$,5));
Pr_regi_dow_c= evstr(Prod_path(2:$,6));  
Pr_regi      = Prod_path(2:$,7);
Pr_regi_dow  = Prod_path(2:$,8);  
Pr_variabs   = Prod_path(2:$,4);



//===============================================================================
for scenario = [1:size(scenario_names,2)];

      disp(scenario_names(scenario))

      scenario_type = Scenario_types(scenario) 
      productivity_scenario = productivity_source(scenario) 
      
      runname = scenario_names(scenario);
      mfprintf(fileID,runname+"\n");

      // Parameter
      retirement_age = S_retirement_age(scenario);
      productivity_growth = S_productivity_growth(scenario);
      ind_gem = S_ind_gem(scenario);
      ind_struct = S_ind_struct(scenario);
      ind_unemployment = S_ind_unemployment(scenario);
      ind_entry_leave = S_ind_entry_leave(scenario);
      ind_preparing_infilling = S_ind_preparing_infilling(scenario);
      ind_workforce = S_ind_workforce(scenario);
      ind_smooth = 1 ;      


      
      // Inputing data
      Data = csvRead('../input/'+scenario_source(scenario)+".csv",',','[]',"string");
      
      // Separating values from metadata
      Data_header = Data(1,:);
      Data_T = Data_header(6:$);
      Data = Data(2:$,:);
      Scenari_up = Data(:,2);
      Regions_up = Data(:,3);
      Variabs_up = Data(:,4);
      Model = Data(1,1); 
      Data=InterpolatingYearlyData(Data);


      // Running Analysis
      exec('V2_nexus.labour.core.sce');
      
      // Saving Results
      All_Result = ['Model','Scenario','Region','Downscaled Region','Variable','Unit',string([2015:2100])];
      Scenario_list = repmat(runname,NRegions,1);
      Model_list = repmat(Model+' - Coal Labour Nexus Downscaling',NRegions,1);
            
      //=============================================
      //=============================================  
      // Core results
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Coal Worker Destination|Unemployment",NRegions,1),repmat("People",NRegions,1),string(Unemployed_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Resource|Extraction|Coal|Downscaled",NRegions,1),repmat("EJ/yr",NRegions,1),string(Qcoal_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Coal|Downscaled",NRegions,1),repmat("People",NRegions,1),string(Lcoal_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Coal Worker Destination|Hire",NRegions,1),repmat("People",NRegions,1),string(Hire_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Coal Worker Destination|Retire",NRegions,1),repmat("People",NRegions,1),string(Retire_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Coal Worker Destination|Instant Match",NRegions,1),repmat("People",NRegions,1),string(Instant_M_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Coal Worker Destination|Delayed Match",NRegions,1),repmat("People",NRegions,1),string(Delayed_M_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Labour Productivity|Coal|Downscaled",NRegions,1),repmat("EJ/yr/person",NRegions,1),string(Produ_pr)]];


      // Additional results
      All_Result= [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Coal Worker Searching|Downscaled",NRegions,1),repmat("People",NRegions,1),string(CS_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Available Vacancies|Downscaled",NRegions,1),repmat("People",NRegions,1),string(Openings_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Coal|Destruction|Downscaled",NRegions,1),repmat("People",NRegions,1),string(CD_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Unemployment|Downscaled",NRegions,1),repmat("People",NRegions,1),string(Unemployement_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Total Vacancies|Downscaled",NRegions,1),repmat("People",NRegions,1),string(TOpenings_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Labour Force|Downscaled",NRegions,1),repmat("People",NRegions,1),string(LF_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Non Coal|Downscaled",NRegions,1),repmat("People",NRegions,1),string(NC_pr)]];


      // Structural change results
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Agriculture|Share|Downscaled",NRegions,1),repmat("-",NRegions,1),string(SA_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Industry|Share|Downscaled",NRegions,1),repmat("-",NRegions,1),string(SM_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Services|Share|Downscaled",NRegions,1),repmat("-",NRegions,1),string(SS_pr)]];

                        
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Agriculture|Downscaled",NRegions,1),repmat("People",NRegions,1),string(TA_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Industry|Downscaled",NRegions,1),repmat("People",NRegions,1),string(TM_pr)]];
      All_Result = [All_Result;
                        [Model_list,Scenario_list,Country_names,Region_code,repmat("Employment|Services|Downscaled",NRegions,1),repmat("People",NRegions,1),string(TS_pr)]];


      //=============================================
      // Saving output                
      csvWrite(All_Result, SAVEDIR+'/Downscaled_coal_labour_'+runname+'.csv');
      mfprintf(fileID, "Written :"+runname+"\n");
      
end


//=============================================
// Terminate
time=getdate();
mfprintf(fileID, "Terminatig \n Time: "+string(time(7))+":"+string(time(8))+":"+string(time(9)));
mclose(fileID);

