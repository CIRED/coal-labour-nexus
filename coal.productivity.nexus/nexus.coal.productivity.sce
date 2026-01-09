// ============================================================================
// NEXUS Coal productivity module
// ============================================================================ 

// This module aims to introduce a data-rich approach to tracking the dynamics of labour productivity change in the coal mining sector. 
// It exploits the Gloal Energy Monitor coal mine database (April 2024 version + September 2024 supplement) and complementary data (e.g. sub-national wage data, coal production price data by grade and coal type) to rank mines according to an estimate of their economic performance. 
// This module was coded in Scilab to facilitate its integration into the imaclim framework. It could be translated into other languages.

// Input: This module takes national coal production trajectories on an annual basis as input. In the study, these are derived from the simulation of several climate change mitigation scenarios using the Imaclim model. Several trajectories corresponding to several scenarios can be taken into account.
// Output: The module calculates the average productivity and production of coal mining at the provincial or state level. It can therefore anticipate changes in the distribution of production areas over time,  as well as the resulting changes in productivity at national and subnational levels.

// Organisation of the nexus
// I. Loading and preprocessing of national coal production trajectories
// II. Calibration: consisting in creating the variables and loading the coal-mine data
// III. Dynamics of Mine Operation and Mine Stock Evolution


// I. LOADING AND PREPROCESSING OF NATIONAL COAL PRODUCTION TRAJECTORIES ///////////////////////////////////////////

// Load coal output trajectories from the Ways-Out scenarios (final scenarios used in the paper).
// Two options are available to load the trajectories: 

ind_output_extract = 1; // Trajectory input format option: 1 = pre-extracted CSV; 2 = standard Imaclim scenario output files

if ind_output_extract == 1 
    traj_coaloutput_scename = csvRead(DATA+'Coal_Productivity/input/IMACLIM_waysout_outputs_03032025.csv',',',[],"string",[],'/\/\//',[2 2 38000 2]);
    traj_coaloutput_country = csvRead(DATA+'Coal_Productivity/input/IMACLIM_waysout_outputs_03032025.csv',',',[],"string",[],'/\/\//',[2 3 38000 3]);
    traj_coaloutput_variable = csvRead(DATA+'Coal_Productivity/input/IMACLIM_waysout_outputs_03032025.csv',',',[],"string",[],'/\/\//',[2 4 38000 4]);
    traj_coaloutput_unit = csvRead(DATA+'Coal_Productivity/input/IMACLIM_waysout_outputs_03032025.csv',',',[],"string",[],'/\/\//',[2 5 38000 5]);
    traj_coaloutput_matrix = csvRead(DATA+'Coal_Productivity/input/IMACLIM_waysout_outputs_03032025.csv',',',[],[],[],'/\/\//',[2 6 38000 100]);
    Q_traj_NPi = traj_coaloutput_matrix(find((traj_coaloutput_scename == "WO-NPi-ElecIndus-CCS0")&(traj_coaloutput_country=="CHN"|traj_coaloutput_country=="IND")&(traj_coaloutput_unit=="Mtoe")&(traj_coaloutput_variable == "Output|Coal")),:);
    Q_traj_NDC = traj_coaloutput_matrix(find((traj_coaloutput_scename == "WO-NDCLTT-ElecIndus-CCS0")&(traj_coaloutput_country=="CHN"|traj_coaloutput_country=="IND")&(traj_coaloutput_unit=="Mtoe")&(traj_coaloutput_variable == "Output|Coal")),:);
    Q_traj_NDC_CCS = traj_coaloutput_matrix(find((traj_coaloutput_scename == "WO-NDCLTT-ElecIndus-CCS1")&(traj_coaloutput_country=="CHN"|traj_coaloutput_country=="IND")&(traj_coaloutput_unit=="Mtoe")&(traj_coaloutput_variable == "Output|Coal")),:);
    Q_traj_15C = traj_coaloutput_matrix(find((traj_coaloutput_scename == "WO-15C-ElecIndus-CCS0")&(traj_coaloutput_country=="CHN"|traj_coaloutput_country=="IND")&(traj_coaloutput_unit=="Mtoe")&(traj_coaloutput_variable == "Output|Coal")),:);
    Q_traj_15C_CCS = traj_coaloutput_matrix(find((traj_coaloutput_scename == "WO-15C-ElecIndus-CCS1")&(traj_coaloutput_country=="CHN"|traj_coaloutput_country=="IND")&(traj_coaloutput_unit=="Mtoe")&(traj_coaloutput_variable == "Output|Coal")),:);
    Q_traj = [Q_traj_NPi;Q_traj_NDC;Q_traj_NDC_CCS;Q_traj_15C;Q_traj_15C_CCS];
    Q_traj_name = ["NPi_CCS0";"NDC_CCS0";"NDC_CCS1";"1.5C_CCS0";"1.5C_CCS1"];

else // Alternative input format: standard Imaclim scenario output files (used in early runs).
    Q_traj_NPi = [csvRead(DATA + "Coal_Productivity/input/"+"outputs_base" + 112 + ".tsv","\t",".",[],[],'/\/\//',[3241 1 3241 87]) ; csvRead(DATA + "Coal_Productivity/input/"+"outputs_base" + 112 + ".tsv","\t",".",[],[],'/\/\//',[3788 1 3788 87])];
    Q_traj_NDC = [csvRead(DATA + "Coal_Productivity/input/"+"outputs_base" + 172 + ".tsv","\t",".",[],[],'/\/\//',[3241 1 3241 87]) ; csvRead(DATA + "Coal_Productivity/input/"+"outputs_base" + 172 + ".tsv","\t",".",[],[],'/\/\//',[3788 1 3788 87])];
    Q_traj_NZE = [csvRead(DATA + "Coal_Productivity/input/"+"outputs_base" + 4202 + ".tsv","\t",".",[],[],'/\/\//',[3241 1 3241 87]) ; csvRead(DATA + "Coal_Productivity/input/"+"outputs_base" + 4202 + ".tsv","\t",".",[],[],'/\/\//',[3788 1 3788 87])];
    Q_traj = [Q_traj_NPi;Q_traj_NDC;Q_traj_NZE];
end

// For integration within Imaclim, use Q (possibly using a smoothed moving average).
// At this stage the module can be run ex post (no added value from running simultaneously). A future extension could introduce feedback loops (e.g., productivity growth affecting wages, costs, prices, and possibly investment).

// Compute moving averages of Q (handling missing data).
Q_traj_mean = 0 * Q_traj;
for  j_Q_traj = 1:size(Q_traj,1)
    Q_traj_mean(j_Q_traj,1) = nanmean(Q_traj(j_Q_traj,1:2));
    Q_traj_mean(j_Q_traj,2) = nanmean(Q_traj(j_Q_traj,1:3));
    Q_traj_mean(j_Q_traj,$) = nanmean(Q_traj(j_Q_traj,($-2):$));
    for i_Q_traj_mean = 3:(size(Q_traj,2)-1)
        Q_traj_mean(j_Q_traj,i_Q_traj_mean) = nanmean(Q_traj(j_Q_traj,i_Q_traj_mean-2:i_Q_traj_mean+1));
    end
end

// Build a matrix of annual changes in Q based on the trajectories.
Q_traj_var = Q_traj_mean./repmat(Q_traj_mean(:,7),1,size(Q_traj_mean,2));

for i_Q_traj = 1:size(Q_traj_name,1) // Loop to run the module across scenarios. 
   
    for H_decap_China = -20  // Decapacity target (Mt/year). For reference, ~160 Mt/year was reached over 2011–2019.
    //for H_decap_China = [-20, 0, -40] // Sensitivity loop over decapacity assumptions (optional).
        H_decap_India = H_decap_China/4 ; // Decapacity target for India (Mt/year). 
        
        for H_newprod_opt = 1  // Assumption on productivity of newly opened mines: 1 = optimistic (+100% vs. the post-2018 average productivity in the province/state).
        //for H_newprod_opt = [1, 0] // Sensitivity loop over new-mine productivity assumptions (optional).



// II. INITIALISATION 	////////////////////////////////////////////////////////////////////////////////////////////////

            for current_time_im = 1:86 // In this version, the module is run ex post up from 2015 to 2100 (outside Imaclim).
                
                if current_time_im >= 8 // 
                    disp("Closure of Mining Capacity in China: " + SuiviCapacite_closed(coal_indexes == "China",current_time_im-1) + ", in Inde : " + SuiviCapacite_closed(coal_indexes == "India",current_time_im-1));
                if current_time_im == 2021-base_year_simulation // We start using this module after 2021 (earlier years rely on exogenous data; GEM mine data are roughly representative of the early-2020s).

                    // Loading an output database (standard format used in the model)
                    coal_indexes = csvRead(DATA+'Coal_Productivity/Indexes.csv',',',[],"string",[],'/\/\//',[2 4 67 4]);
                    coal_indexes_iso = csvRead(DATA+'Coal_Productivity/Indexes.csv',',',[],"string",[],'/\/\//',[2 5 67 5]);
                    coal_indexes_codecountry = csvRead(DATA+'Coal_Productivity/Indexes.csv',',',[],[],[],'/\/\//',[2 1 67 1]);
                    
                    coal_newprod_opt = H_newprod_opt; // Assumption on productivity of newly opened mines: 1 = optimistic (+100% vs. the post-2018 average productivity in the province/state).
                    if coal_newprod_opt == 1
                        GEM_data_suffixe = '_P100';
                    elseif coal_newprod_opt == 0 then
                        GEM_data_suffixe = '_P0';
                    end
                    
                    // Load the initial mine database. Scilab handles mixed-type tables poorly, so we load columns as separate vectors.
                    // As it is difficult (if not impossible) to manipulate a dataframe with Scilab, we construct vectors of the different variables in the database. So, this code is not very robust to changes in the structure of the database, but we don't see any other option with Scilab.
                    coalmine_ID = evstr(csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],"string",[],'/\/\//',[2 1 10000 1]));
                    coalmine_country = evstr(csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],'string',[],'/\/\//',[2 4 10000 4]));
                    coalmine_subnat = evstr(csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],'string',[],'/\/\//',[2 5 10000 5]));
                    coalmine_status = evstr(csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],'string',[],'/\/\//',[2 6 10000 6]));
                    coalmine_output = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 12 10000 12]);
                    coalmine_output_gj = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 13 10000 13]);
                    coalmine_workforcesize = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 9 10000 9]);
                    coalmine_minetype = evstr(csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],'string',[],'/\/\//',[2 10 10000 10]));
                    coalmine_miningmethod = evstr(csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],'string',[],'/\/\//',[2 11 10000 11]));
                    coalmine_reserve = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 14 10000 14]);
                    coalmine_reserve_gj = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 15 10000 15]);
                    coalmine_wage = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 16 10000 16]);
                    coalmine_productivity = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 17 10000 17]);
                    coalmine_productivity_gj = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 18 10000 18]);
                    coalmine_indexperf = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 19 10000 19]);
                    coalmine_rank = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 20 10000 20]);
                    coalmine_sharenewoutput = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 21 10000 21]);
                    coalmine_heatcontent = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 22 10000 22]);
                    coalmine_outputprice = csvRead(DATA+'Coal_Productivity/input/CoalEconomy_newdata_GEM_maxOutput_ResHigh'+GEM_data_suffixe+'.csv','|',[],[],[],'/\/\//',[2 23 10000 23]);
                    

                    // Create a mine-by-year activity tracking matrix (1 = used; 0 = not used). [test/debug]
                    coalmine_monitoring = [];

                    // Creation of vectors (no longer used?)
                    coal_country_unique = unique(coalmine_country);
                    coal_subnat_unique = unique(coalmine_subnat);
                    

                    // Initialization of the productivity, productivity growth rate and output matrixes, which are the main outputs of this nexus
                    CoalProdvty_growthrate = zeros(size(coal_indexes,1),TimeHorizon); // Note that for the 7 first years, the coal employment module use fixed rates, so we don't compute it.
                    CoalProdvty_subnat = zeros(size(coal_indexes,1),TimeHorizon);
                    CoalOutputYear_reg = zeros(size(coal_indexes,1),TimeHorizon);
                    CoalOutputYear_reg_ton = zeros(size(coal_indexes,1),TimeHorizon);
                    CoalOutputYear_reg_ejton = zeros(size(coal_indexes,1),TimeHorizon);


                    //Creation of monitoring files (used for the module development) :
                    OpenedMines = []; // List of opened mines (IDs).
                    SuiviMouvements_closed  = zeros(size(coal_indexes,1),TimeHorizon); // Track the number of mines closed each year.
                    SuiviMouvements_opened  = zeros(size(coal_indexes,1),TimeHorizon);
                    SuiviCapacite_closed    = zeros(size(coal_indexes,1),TimeHorizon); // Track the capacity closed each year.
                    SuiviCapacite_opened    = zeros(size(coal_indexes,1),TimeHorizon);
                

                    // Initialise production by country and province/state from the GEM database.
                    for i_subnat = 1:size(coal_indexes,1) // Loop over coal_indexes rows (one per province/state/country).
                        // Build the list of mines belonging to the given country/province/state.
                        if sum(coal_indexes(i_subnat)==coal_country_unique)==0 // If this row corresponds to a province/state (as opposed to a country aggregate). 
                            listmines_in = find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Operating"));
                        else // otherwise, the lines are for countries
                            listmines_in = find((coalmine_country == coal_indexes(i_subnat))&(coalmine_status == "Operating"));
                        end

                        // Compute totals (sum over mines) and labour productivity (sum output / sum workforce).
                        if length(listmines_in)==0
                            CoalProdvty_subnat(i_subnat,current_time_im) = 0; // Initialisation de la productivité à partir de la base de données GEM
                            CoalProdvty_growthrate(i_subnat,current_time_im) = %nan; // Always NA in the baseline year.
                            CoalOutputYear_reg(i_subnat,current_time_im) = 0; // Initialise production by country/region from the GEM database.
                            CoalOutputYear_reg_ton(i_subnat,current_time_im) = 0; // Initialise production by country/region from the GEM database., en tonnes

                        else
                            CoalProdvty_subnat(i_subnat,current_time_im) = nansum(coalmine_output_gj(listmines_in)) / nansum(coalmine_workforcesize(listmines_in)); // Initialisation de la productivité à partir de la base de données GEM
                            CoalProdvty_growthrate(i_subnat,current_time_im) = %nan; // NA in the first year.
                            CoalOutputYear_reg(i_subnat,current_time_im) = nansum(coalmine_output_gj(listmines_in));// Initialise production by country/region from the GEM database.
                            CoalOutputYear_reg_ton(i_subnat,current_time_im) = nansum(coalmine_output(listmines_in));

                        end
                    end
                    // Store the first-year bottom-up productivity as the reference.
                    CoalProdvty_subnat_ref  = CoalProdvty_subnat(:,current_time_im);
                    CoalOutputYear_reg_ref = CoalOutputYear_reg(:,current_time_im);

                    // Store the Imaclim reference production (including coal output).
                      
                end // End of initialisation.



        // III. DYNAMICS OF MINE OPERATION AND MINE STOCK EVOLUTION /////////////////////////////////////////////////

                // Start mine-level operations: production (and reserve depletion), closures, openings, and annual activity flag.

                total_mines_closed = 0; // Initialise annual closed-mine counter (debug/monitoring).

                if current_time_im >= 2022-base_year_simulation // We start using this module after 2021, as we have exogenous data before.

                    // Reset the 'used this year' flag for all mines before running annual operations (useful for debugging/validation). 
                    coalmine_usedthisyear = zeros(size(coalmine_country,1),1);

                    for i_region = ["China","India"] // Loop over the modelled countries (target output paths are country-level). 
                        coalmine_change = 0 ; // Bottom-up output tracking index (relative to the reference year); used to check when the target output is reached. 
                        ind_reg_imaclim = (i_region=="China")*ind_chn + (i_region=="India")*ind_ind; // Line to obtain the index number of the corresponding region.
                        minesofcountry = find(coalmine_country == i_region & ~isnan(coalmine_rank)); // Indices of mines in the country with sufficient data to be modelled (very few are excluded).
                        Q_change = Q_traj_var(2*i_Q_traj-((i_region=="China")*1 + (i_region=="India")*0),current_time_im); // Target output index (relative to the reference year; exogenous, from the Imaclim scenario path).
                        disp("Change in Q this year : " + round(1000*(Q_change-1))/10 + "% in " + i_region); // Displayed to check

                        [sorted_A, coalmine_rank_sorted] = gsort(coalmine_rank(minesofcountry), "g", "i"); // Sort mines by rank.

                        // 1. Prior application of a de-capacity policy that removes the lowest-performing capacity on an annual basis
                        // Inspired by decapacity policies that closed ~1600 Mt (~9000 mines) between 2011 and 2019.  
                        ind_coal_decapacity = 1 ; // index to activate the option (1 = active)
                        if ind_coal_decapacity == 1 
                            if i_region == "China" &  total_mines_closed <= 1000 // We cap the number of closures for tractability; the threshold corresponds roughly to closing the lowest-productivity quartile of mines.
                                coal_decapacity_target = H_decap_China ; // Decapacity target (Mt/year). 60 per year is the sustained pace reached during the 2011–2019 period
                                H_decap_China_text = "_C" + string(abs(coal_decapacity_target));
                            elseif i_region == "India" &  total_mines_closed <= 1000 // NB: probably never reached for India
                                coal_decapacity_target =  H_decap_India; // 
                                H_decap_India_text = "I" + string(abs(coal_decapacity_target));
                            end

                                coalmine_decap = 0; // Index used to track closed capacities (loop continues until the above target is reached)
                                nb_mines_closed = 0; // Counter used to enforce the maximum number of closures.
                            for i_rank = length(minesofcountry):-1:1 // Iterate mines from lowest to highest rank (worst performers first).
                                if  coalmine_decap > coal_decapacity_target
                                    if coalmine_status(minesofcountry(coalmine_rank_sorted(i_rank))) == "Operating"
                                        coalmine_status(minesofcountry(coalmine_rank_sorted(i_rank)),1) = "Closed"; // Decommissioning implemented by changing status from 'Operating' to 'Closed'.
                                        coalmine_decap = coalmine_decap - coalmine_output(minesofcountry(coalmine_rank_sorted(i_rank)));
                                        nb_mines_closed = nb_mines_closed+1;
                                        // The following variables are control variables, mainly useful for development
                                        SuiviMouvements_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = SuiviMouvements_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + 1;
                                        SuiviCapacite_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = SuiviCapacite_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                        SuiviMouvements_closed(coal_indexes == i_region,current_time_im) = SuiviMouvements_closed(coal_indexes == i_region,current_time_im) +1;
                                        SuiviCapacite_closed(coal_indexes == i_region,current_time_im) = SuiviCapacite_closed(coal_indexes == i_region,current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    end
                                end
                            end
                            disp("Fermeture de " + nb_mines_closed + " mines en " + (current_time_im+2014) + " in " + i_region); // Development monitoring line
                        end
                        total_mines_closed = total_mines_closed + nb_mines_closed;

                        // 2. Operation of existing mines to reach the expected production level
                        for i_rank = 1:length(minesofcountry) // Mines are processed in ascending rank order ...
                            if coalmine_change < Q_change // ... and the loop stops once the annual production target is reached
                        
                                if coalmine_status(minesofcountry(coalmine_rank_sorted(i_rank))) == "Operating" // If the mine has this status
                                    // adding the mine’s output to the country’s total output
                                    CoalOutputYear_reg(coal_indexes == i_region,current_time_im) = CoalOutputYear_reg(coal_indexes == i_region,current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    CoalOutputYear_reg_ton(coal_indexes == i_region,current_time_im) = CoalOutputYear_reg_ton(coal_indexes == i_region,current_time_im) + coalmine_output(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    // adding the mine’s output to the province’s output
                                    CoalOutputYear_reg(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = CoalOutputYear_reg(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    CoalOutputYear_reg_ton(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = CoalOutputYear_reg_ton(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    // flag indicating that the mine has been operated this year
                                    coalmine_usedthisyear(minesofcountry(coalmine_rank_sorted(i_rank))) = 1;
                                    // subtracting this year’s production from the mine’s reserves
                                    coalmine_reserve_gj(minesofcountry(coalmine_rank_sorted(i_rank))) = coalmine_reserve_gj(minesofcountry(coalmine_rank_sorted(i_rank))) - coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));

                                    // If reserves fall below annual production, the mine is closed and will no longer be used
                                    if coalmine_reserve_gj(minesofcountry(coalmine_rank_sorted(i_rank))) < coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)))
                                        coalmine_status(minesofcountry(coalmine_rank_sorted(i_rank)),1) = "Closed_Depletion";
                                        // The variables below are monitoring variables, mainly useful during development
                                        SuiviMouvements_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = SuiviMouvements_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + 1;
                                        SuiviCapacite_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = SuiviCapacite_closed(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                        SuiviMouvements_closed(coal_indexes == i_region,current_time_im) = SuiviMouvements_closed(coal_indexes == i_region,current_time_im) +1;
                                        SuiviCapacite_closed(coal_indexes == i_region,current_time_im) = SuiviCapacite_closed(coal_indexes == i_region,current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    end
                                end
                                // Update coalmine_change with the mine’s output before moving to the next mine
                                coalmine_change = CoalOutputYear_reg(coal_indexes == i_region,current_time_im)/CoalOutputYear_reg_ref(coal_indexes == i_region);
                            end
                        end // End of loop for operation of existing mines


                        
                        // 3. Opening project-stage mines to fill the shortfall
                        nb_mines_opened = 0; // Initialization of this control variable
                        capacity_opened = 0; // Initialization of this control variable
                        // If national production is still below the target, project-stage mines are opened first, in order of their rank
                        for i_rank = 1:length(minesofcountry)
                            if coalmine_change < Q_change
                                if coalmine_status(minesofcountry(coalmine_rank_sorted(i_rank))) == "Proposed" // NB: this could be refined with status and/or project-phase details
                                    // To be put into operation, the mine changes status and is then operated for the first time with the same procedures as an active mine
                                    coalmine_status(minesofcountry(coalmine_rank_sorted(i_rank)),1) = "Operating";
                                    // adding the mine’s output to the country’s total output
                                    CoalOutputYear_reg(coal_indexes == i_region,current_time_im) = CoalOutputYear_reg(coal_indexes == i_region,current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    CoalOutputYear_reg_ton(coal_indexes == i_region,current_time_im) = CoalOutputYear_reg_ton(coal_indexes == i_region,current_time_im) + coalmine_output(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    // adding the mine’s output to the province’s output
                                    CoalOutputYear_reg(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = CoalOutputYear_reg(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    CoalOutputYear_reg_ton(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = CoalOutputYear_reg_ton(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    // flag indicating that the mine has been operated this year
                                    coalmine_usedthisyear(minesofcountry(coalmine_rank_sorted(i_rank))) = 1;
                                    // subtracting this year’s production from the mine’s reserves
                                    coalmine_reserve_gj(minesofcountry(coalmine_rank_sorted(i_rank))) = coalmine_reserve_gj(minesofcountry(coalmine_rank_sorted(i_rank))) - coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    
                                    // The variables below are monitoring variables, mainly useful during development
                                    nb_mines_opened = 1 + nb_mines_opened;
                                    capacity_opened = coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank))) + capacity_opened;
                                    OpenedMines = [OpenedMines minesofcountry(coalmine_rank_sorted(i_rank))];                            
                                    SuiviMouvements_opened(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = SuiviMouvements_opened(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + 1;
                                    SuiviCapacite_opened(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) = SuiviCapacite_opened(coal_indexes == coalmine_subnat(minesofcountry(coalmine_rank_sorted(i_rank))),current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));
                                    SuiviMouvements_opened(coal_indexes == i_region,current_time_im) = SuiviMouvements_opened(coal_indexes == i_region,current_time_im) +1;
                                    SuiviCapacite_opened(coal_indexes == i_region,current_time_im) = SuiviCapacite_opened(coal_indexes == i_region,current_time_im) + coalmine_output_gj(minesofcountry(coalmine_rank_sorted(i_rank)));

                                end
                            end

                            // Update coalmine_change with the mine’s output before moving to the next mine                 
                            coalmine_change = CoalOutputYear_reg(coal_indexes == i_region,current_time_im)/CoalOutputYear_reg_ref(coal_indexes == i_region);         
                        end // End of loop for project-mine commissioning
                        
                        // Line used to track project-mine openings during execution (useful for development control)
                        if nb_mines_opened >=1
                            disp("Ouverture de projets en cours en " + (current_time_im+2014) + " en " + i_region + " : " + nb_mines_opened + " mines, " + capacity_opened + " Mt");
                        end


                        // 4. Creation of fictitious mines if opened mines do not allow the expected production to be reached
                        // Note: this follows a planning logic, where mines are opened to meet demand because it can be anticipated; in reality, higher imports could also occur  
                                     
                        // Chosen method: one generation of new mines is created each year, while keeping fictitious mine entries that define the attributes of newly installed capacity (by region, based on recent installations)
                        else 
                            if coalmine_change < Q_change // Condition for using fictitious mines: national output remains below the target
                                disp("Création de mines fictives en " + (current_time_im+2014) + " in " + i_region); // Display that new capacities are installed in the country this year (for monitoring)
                                coal_missingoutput = (Q_change - coalmine_change) * CoalOutputYear_reg_ref(coal_indexes == i_region); // Production gap
                                
                                // Preliminary step: regional shares are adjusted when a territory’s reserves are exhausted
                                for i_subnat = 1: size(coal_indexes,1) // Loop over territories; if expected output exceeds reserves, the regional share is set to zero
                                    if length(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive")))>0
                                        if coalmine_reserve_gj(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))) < (coal_missingoutput * coalmine_sharenewoutput(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive")))) // If reserves are below the desired output, the resource is considered fully depleted (approximation)
                                            coalmine_sharenewoutput(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))) = 0;
                                        end
                                    end
                                end
                                // Renormalise territorial shares if needed at the end of this process
                                coalmine_sharenewoutput(find((coalmine_country == i_region) & (coalmine_status == "Fictive"))) = coalmine_sharenewoutput(find((coalmine_country == i_region) & (coalmine_status == "Fictive")))/sum(coalmine_sharenewoutput(find((coalmine_country == i_region) & (coalmine_status == "Fictive"))));
                                

                                for i_subnat = 1: size(coal_indexes,1) // Loop over all producing (and non-producing) regions, adding fictitious mines proportional to recent output contributions
                                    if (coal_indexes_codecountry(i_subnat) == ind_reg_imaclim) & (length(coal_indexes_iso(i_subnat))>2) // Only for the country being processed; and only for territories (not for “country” rows)
                                        if sum(coal_indexes(i_subnat) == coal_subnat_unique) > 0 // Only for producing territories
                                            if length(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive")))>0 // Certain regions with initial mines may have had no recent output; no fictitious mine created there.

                                                // Define all attributes of the territory’s new mine so that all “mine database” arrays remain aligned
                                                newmine_index   = size(coalmine_ID,1)+1; // Index of the new mine
                                                coalmine_ID(newmine_index,1)                = "F_" + coal_indexes_iso(i_subnat) + "_" + (current_time_im + base_year_simulation); // Create an identifier
                                                coalmine_country(newmine_index,1)           = i_region; // Country
                                                coalmine_subnat(newmine_index,1)            = coal_indexes(i_subnat); // Territory
                                                coalmine_status(newmine_index,1)            = "Operating"; // The mine now belongs to existing mines and will be handled in step 2
                                                coalmine_output_gj(newmine_index,1)         = coal_missingoutput * coalmine_sharenewoutput(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))); // Assign installed capacity to the region based on its contribution to recent installations
                                                coalmine_heatcontent(newmine_index,1)       = coalmine_heatcontent(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))); // Heat content based on regional average
                                                coalmine_output(newmine_index,1)            = coalmine_output_gj(newmine_index,1) / coalmine_heatcontent(newmine_index,1);
                                                coalmine_minetype(newmine_index,1)          = "Mix";
                                                coalmine_miningmethod(newmine_index,1)      = "Mix";
                                                coalmine_reserve_gj(newmine_index,1)        = min(30 * coalmine_output_gj(newmine_index,1),coalmine_reserve_gj(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive")))); // Assume 30-year lifetime for new mines
                                                coalmine_reserve_gj(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))) = max(coalmine_reserve_gj(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))) - coalmine_reserve_gj(newmine_index,1),0); // Track territorial reserves using the fictitious-mine row
                                                coalmine_wage(newmine_index,1)              = coalmine_wage(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))); 
                                                growthrate_newcapacity                      = ((i_region=="China")*1.02 + (i_region=="India")*1.025)^max(current_time_im+base_year_simulation-2023,1); // Productivity growth rate of new capacity indexed to wage growth
                                                coalmine_productivity_gj(newmine_index,1)   = growthrate_newcapacity * coalmine_productivity_gj(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))); // Apply productivity growth; most recent capacities will be better ranked
                                                coalmine_productivity(newmine_index,1)      = coalmine_productivity_gj(newmine_index,1) / coalmine_heatcontent(newmine_index,1);
                                                coalmine_reserve(newmine_index,1)           = coalmine_reserve_gj(newmine_index,1) / coalmine_heatcontent(newmine_index,1);
                                                coalmine_workforcesize(newmine_index,1)     = coalmine_output_gj(newmine_index,1)/coalmine_productivity_gj(newmine_index,1)*10^6;
                                                coalmine_outputprice(newmine_index,1)       = coalmine_outputprice(find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_status == "Fictive"))); // Price based on regional price (coal quality assumed homogeneous)
                                                coalmine_indexperf(newmine_index,1)         = coalmine_outputprice(newmine_index,1) / coalmine_wage(newmine_index,1) * coalmine_productivity(newmine_index,1); // Compute the performance index of the new capacity
                                                coalmine_usedthisyear(newmine_index,1)      = 1;
                                                coalmine_sharenewoutput(newmine_index,1)    = 0; // No longer used for these new capacities
                                                coalmine_rank(newmine_index,1)              = 0; // Will be defined later
                                                
                                            end
                                        end
                                    end
                                end                       
                            end    
                        end // End of loop on creation of new capacity
                                    

                        // 5. Updating mine ranks (determined by a performance index)
                        list_indexperf = gsort(unique(coalmine_indexperf(find((coalmine_country == i_region)))), "g","d"); // List performance indices in descending order
                        list_indexperf = list_indexperf(~isnan(list_indexperf)); // Remove NA values
                        // 9/11: block below removed because it no longer appears to be used
                        rank_mine = 1; // Initialize variable rank_mine
                        coalmine_rank(find(((coalmine_status~="Operating")|(coalmine_status ~= "Proposed")) & (coalmine_country == i_region))) = %nan ; // A rank is assigned only to mines with status "Operating" or "Proposed"
                        for i_indexperf = 1 : size(list_indexperf,1) // Loop through performance indices in descending order
                            ranks_todefine = find((coalmine_indexperf==list_indexperf(i_indexperf))&(coalmine_country == i_region)); // Identify the mine whose rank will be assigned this round
                            if size(ranks_todefine,2) > 0 // Handle equal-rank cases
                                for i_rank_update = 1:size(ranks_todefine,2) // Assign ranks to mines associated with this performance index; ensure rank uniqueness
                                    rank_temp = ranks_todefine(i_rank_update);
                                    if (coalmine_status(rank_temp)=="Operating"|coalmine_status(rank_temp)=="Proposed") & (coalmine_country(rank_temp) == i_region)
                                        coalmine_rank(rank_temp) = rank_mine ;
                                        rank_mine = rank_mine + 1;
                                    end
                                end
                            end
                        end
                    end

        // III. Complete regional matrices for output, productivity, and productivity-growth

                    // Loop over territories (including countries) in the predefined output table
                    for i_subnat = 1:size(coal_indexes,1)
                        // List the mines in the territory (used for bottom-up computation of production and workforce)
                        if sum(coal_indexes(i_subnat)==coal_country_unique)==0 // If row corresponds to a province/state (as opposed to country aggregate). 
                            listmines_in = find((coalmine_subnat == coal_indexes(i_subnat))&(coalmine_usedthisyear == 1));
                        else // Otherwise, this corresponds to “country/region” rows
                            listmines_in = find((coalmine_country == coal_indexes(i_subnat))&(coalmine_usedthisyear == 1));
                        end

                        // Use this list to compute regional values; handle separately cases with no production
                        if length(listmines_in)==0 // Case where no mine has been operated in the region
                            CoalProdvty_subnat(i_subnat,current_time_im) = 0; 
                            CoalProdvty_growthrate(i_subnat,current_time_im) = %nan; 
                            CoalOutputYear_reg(i_subnat,current_time_im) = 0; 
                            CoalOutputYear_reg_ton(i_subnat,current_time_im) = 0; 
                        elseif nansum(coalmine_output_gj(listmines_in)) == 0 // Case of no production
                            CoalProdvty_subnat(i_subnat,current_time_im) = 0; 
                            CoalProdvty_growthrate(i_subnat,current_time_im) = %nan; 
                            CoalOutputYear_reg(i_subnat,current_time_im) = 0; 
                            CoalOutputYear_reg_ton(i_subnat,current_time_im) = 0; 
                        else // Case where some mines produced this year
                            CoalProdvty_subnat(i_subnat,current_time_im) = nansum(coalmine_output_gj(listmines_in)) / nansum(coalmine_workforcesize(listmines_in)); 
                            if CoalProdvty_subnat(i_subnat,current_time_im-1) == 0;
                                CoalProdvty_growthrate(i_subnat,current_time_im) = 100*(CoalProdvty_subnat(i_subnat,current_time_im)/CoalProdvty_subnat_ref(i_subnat)-1); // Alternative method for years before systematic computation; years 2015–2021 may need re-examination 
                            else
                                CoalProdvty_growthrate(i_subnat,current_time_im) = 100*(CoalProdvty_subnat(i_subnat,current_time_im)/CoalProdvty_subnat(i_subnat,current_time_im-1)-1); 
                            end
                            CoalOutputYear_reg(i_subnat,current_time_im) = nansum(coalmine_output_gj(listmines_in));
                            CoalOutputYear_reg_ton(i_subnat,current_time_im) = nansum(coalmine_output(listmines_in));
                            CoalOutputYear_reg_ejton(i_subnat,current_time_im) = nansum(coalmine_output_gj(listmines_in))/nansum(coalmine_output(listmines_in));
                        end
                    end


                    // Progressive creation of a matrix of used mines (useful for monitoring, especially during development)
                    coalmine_monitoring_prev = coalmine_monitoring;
                    coalmine_monitoring = zeros(size(coalmine_usedthisyear,1),size(coalmine_monitoring,2));
                    for i_mat = 1: size(coalmine_monitoring_prev,1)
                        for j_mat = 1: size(coalmine_monitoring_prev,2)
                            coalmine_monitoring(i_mat, j_mat) = coalmine_monitoring_prev(i_mat, j_mat);
                        end
                    end
                    coalmine_monitoring=[coalmine_monitoring coalmine_usedthisyear]; 
                        
                end // end "if time > 2021"

                // Writing/exporting the different output files: output, productivity, productivity growth, monitoring indicators
                if or(current_time_im + base_year_simulation == [2100])
                    coalmine_monitoring = [coalmine_ID coalmine_country coalmine_subnat coalmine_output coalmine_minetype coalmine_miningmethod coalmine_reserve coalmine_wage coalmine_productivity coalmine_workforcesize coalmine_indexperf coalmine_rank coalmine_monitoring];
                    date_heure = getdate();
                    date_heure_str = msprintf("%02d%02d%02d-%02d%02d%02d", date_heure(1), date_heure(2), date_heure(6), date_heure(7), date_heure(8),date_heure(9));
                    csvWrite(CoalOutputYear_reg,DATA + "Coal_Productivity/output/"+"CoalOutputYear_reg_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                    csvWrite(CoalOutputYear_reg_ton,DATA + "Coal_Productivity/output/"+"CoalOutputYear_reg_ton_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                    csvWrite(CoalOutputYear_reg_ejton,DATA + "Coal_Productivity/output/"+"CoalOutputYear_reg_ejton_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                    csvWrite(CoalProdvty_subnat,DATA + "Coal_Productivity/output/"+"CoalProdvty_subnat_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                    csvWrite(CoalProdvty_growthrate,DATA + "Coal_Productivity/output/"+"CoalProdvty_growthrate_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                    if writefichierssuivis == 1
                        csvWrite(coalmine_monitoring,DATA + "Coal_Productivity/output/"+"coalmine_monitoring_" + Q_traj_name(i_Q_traj) + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                        csvWrite(SuiviCapacite_closed,DATA + "Coal_Productivity/output/"+"SuiviCapacite_closed_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                        csvWrite(SuiviCapacite_opened,DATA + "Coal_Productivity/output/"+"SuiviCapacite_opened_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                        csvWrite(SuiviMouvements_closed,DATA + "Coal_Productivity/output/"+"SuiviMouvements_closed_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                        csvWrite(SuiviMouvements_opened,DATA + "Coal_Productivity/output/"+"SuiviMouvements_opened_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
                    end
                end

            end // end loop over simulation time

            //disp(OpenedMines); // Previously used to display opened mines (no longer needed)
        end // End of loop over productivity optimism hypothesis (sensitivity analysis)
    end // End of loop over de-capacity hypotheses (sensitivity analysis)
end // End of loop over Q(coal) trajectories



csvWrite(Q_traj_mean,DATA + "Coal_Productivity/output/"+"Q_traje_mean_" + Q_traj_name(i_Q_traj) + "_" + H_decap_China_text + H_decap_India_text + "_" + GEM_data_suffixe + "_" + (current_time_im+base_year_simulation) + "_" + date_heure_str + ".csv");
