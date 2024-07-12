// This file is used to run the analysis of the evolution of coal labour in several IAM scenarios
// It initializes key parameters for the runs and treats output data

//===============================================================================
// Preamble

date=getdate();
date= string(date(1))+string(date(2))+string(date(6));

SAVEDIR = '../output/'+date+'_Coal_labour_results'; mkdir(SAVEDIR);

diary(SAVEDIR+"/"+date+"_summary.log");
fileID = mopen(SAVEDIR+'/'+date+"_log.txt",'w') 
mfprintf(fileID, "File created which is very good \n");
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
// Copying Imaclim variables to ease transition
exec("imaclim.variables.sce");

txLact = csvRead('../data/Coal_labour/SSP/active_population_growth_rate__ssp_2023__SSP2.csv','|',[],[],[],'/\/\//');
txEntr = csvRead('../data/Coal_labour/SSP/txEntry_SSP2.csv',',','.',[]);
txExit = csvRead('../data/Coal_labour/SSP/txExit_SSP2.csv',',','.',[]);
txLact_header = txLact(1,:);
ind_first_year = find(txLact_header == base_year_simulation);
txLact = txLact(2:$, (ind_first_year+1):$);

//===============================================================================
// Defining Scenarios

// IAM trajectory source - pathways are assumed to be stored in *.csv file containing a single scenario from single model with header of the form: [Model, Scenario, Region, Variable, Unit, 2015, ...] 
NPI = 'IMACLIM_waysout_outputs_WO-NPi-ElecIndus';
NDC = 'IMACLIM_waysout_outputs_WO-NDCLTT-ElecIndus';
NZ  = 'IMACLIM_waysout_outputs_WO-15C-ElecIndus';

NPI = 'IMACLIM_waysout_outputs_WO-NPi-ElecIndus_EW';
NDC = 'IMACLIM_waysout_outputs_WO-NDCLTT-ElecIndus_EW';
NZ  = 'IMACLIM_waysout_outputs_WO-15C-ElecIndus_EW';

// List of scenarios and associated 
scenario_names = ['NPI','NDC','NZ','NPI_gem','NDC_gem','NZ_gem','NPI_PG0','NDC_PG0','NZ_PG0','NPI_R55','NDC_R55','NZ_R55','NPI_EW','NDC_EW','NZ_EW'];
scenario_source = [NPI,NDC,NZ,NPI,NDC,NZ,NPI,NDC,NZ,NPI,NDC,NZ,NPI,NDC,NZ];
// scenario_names = ['NPI','NDC','NZ','NPI_Pop','NDC_Pop','NZ_Pop','NPI_sc','NDC_sc','NZ_sc','NPI_Dose','NDC_Dose','NZ_Dose','NPI_dpop','NDC_dpop','NZ_dpop'];
// scenario_source = [NPI,NDC,NZ,NPI,NDC,NZ,NPI,NDC,NZ,NPI,NDC,NZ,NPI,NDC,NZ];


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



// Alternative parameters
S_ind_gem(4:6) = 2;
S_productivity_growth(7:9) = 0;
S_retirement_age(10:12) = 55;

// S_ind_struct(4:6)=2;
// S_ind_struct(7:9)=3;
// S_ind_struct(10:12)=4;
// S_ind_struct(13:15)=5;

S_ind_preparing_infilling(1:3)=1;

//===============================================================================
for scenario = [1:size(scenario_names,2)]


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
      ind_smooth = 1 ;      

      
      // Inputing data
      Data = csvRead('../input/'+scenario_source(scenario)+".csv",',','[]',"string");
      
      // Separating values from metadata
      Data_header = Data(1,:);
      Data = Data(2:$,:);
      Regions_up = Data(:,3);
      Variabs_up = Data(:,4);
      Model = Data(1,1); 
      Data = strtod(Data(:,6:$));


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

