// 22/11/2023 - Augustin Danneaux

// This file contains the core analysis of the coal labour nexus



//======================================================================================================================================================

Base_year = 2015;
End_year  = 2101;

//======================================================================================================================================================


if ind_gem == 1 then
    Emp_struct = csvRead('../data/Coal_labour/Downscaling/Econ_structure.csv');
else
    Emp_struct = csvRead('../data/Coal_labour/Downscaling/Econ_structure_GEM.csv');
end

// select I_struct
if (ind_struct==2|ind_struct==5) then

    Emp_struct_pop = csvRead('../data/Coal_labour/Downscaling/Econ_structure_pop.csv');
    t1 = 5;
    t2 = 75;
    Shift_struct = max(min(([1:TimeHorizon]-t1)./(t2-t1),1),0);
elseif (ind_struct==3|ind_struct==4) then

    if length(strindex(runname,"NPI"))~=0  then
        Emp_struct_chnge = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change_NPI.csv');
    elseif length(strindex(runname,"NDC_CCS1"))~=0  then
        Emp_struct_chnge = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change_NDC_CCS1.csv');
    elseif length(strindex(runname,"NZ_CCS1"))~=0  then
        Emp_struct_chnge = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change_NZ_CCS1.csv');
    elseif length(strindex(runname,"NDC"))~=0  then
        Emp_struct_chnge = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change_NDC_CCS0.csv');
    elseif length(strindex(runname,"NZ"))~=0  then
        Emp_struct_chnge = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change_NZ_CCS0.csv');
    else
        Emp_struct_chnge = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change.csv');
    end
    Emp_struct_chnge_index = csvRead('../data/Coal_labour/Downscaling/Econ_structural_change_index.csv', ",", [], "string");
end


      
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Loading the initial age pyramid

Entry_age = 18;


Base_pyramid_China = zeros(TimeHorizon, retirement_age-Entry_age+2);
Base_pyramid_India = zeros(TimeHorizon, retirement_age-Entry_age+2);

Base_pyramid_China(1,:) = age_pyramid.china((Entry_age+2):(retirement_age+3),2)'/sum(age_pyramid.china((Entry_age+2):(retirement_age+2),2)');
Base_pyramid_India(1,:) = age_pyramid.india((Entry_age+2):(retirement_age+3),2)'/sum(age_pyramid.india((Entry_age+2):(retirement_age+2),2)');

for k = [2:retirement_age-Entry_age+2]
    for j = [2:TimeHorizon]
        Base_pyramid_China(j,k)=Base_pyramid_China(j-1,k-1);
        Base_pyramid_India(j,k)=Base_pyramid_India(j-1,k-1);
    end
end

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Employment downscaling

downscaled_countries = [ind_chn,ind_ind];
n_downscaled_countries = size(downscaled_countries,2);


// Initializing arrays 
Z = zeros(reg,TimeHorizon); 
U = zeros(reg,TimeHorizon); 
Prod_coal = [];


// ==================================================================================================================================================================
if length(strindex(Model,"IMACLIM")) ==0
    disp(Model)
    disp("Importing format from other model");
    exec("Default.Labour.sce");
else
    // Post treatment of Imaclim variables to fit in nexus format
    exec("Imaclim.Labour.sce");
end
// ==================================================================================================================================================================




// Calculating the employment in coal
exec("V2_coal.productivity.sce");


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Initializing arrays to store results

Negative_unemployment =zeros(reg,TimeHorizon);
Negative_unemployment_pr = zeros(NRegions,TimeHorizon);

Qcoal_pr     = zeros(NRegions,TimeHorizon);
Lcoal_pr     = zeros(NRegions,TimeHorizon);
Produ_pr     = zeros(NRegions,TimeHorizon);

Hire_pr      = zeros(NRegions,TimeHorizon);
Retire_pr    = zeros(NRegions,TimeHorizon);
Instant_M_pr = zeros(NRegions,TimeHorizon);
Delayed_M_pr = zeros(NRegions,TimeHorizon);
Unemployed_pr= zeros(NRegions,TimeHorizon);


Openings_pr= zeros(NRegions,TimeHorizon);
Unemployement_pr= zeros(NRegions,TimeHorizon);
TOpenings_pr= zeros(NRegions,TimeHorizon);
LF_pr= zeros(NRegions,TimeHorizon);
NC_pr= zeros(NRegions,TimeHorizon);

SA_pr = zeros(NRegions,TimeHorizon);
SM_pr = zeros(NRegions,TimeHorizon);
SS_pr = zeros(NRegions,TimeHorizon);

TA_pr = zeros(NRegions,TimeHorizon);
TM_pr = zeros(NRegions,TimeHorizon);
TS_pr = zeros(NRegions,TimeHorizon);


// Iterating through all downscaled regions
for  k =1:n_downscaled_countries
    Regions = find(Emp_struct(:, 1) == downscaled_countries(k))-1;

    if downscaled_countries(k) == ind_chn then
        Base_pyramid = Base_pyramid_China;
    else
        Base_pyramid = Base_pyramid_India;
    end


    for ks =1:length(Regions)
        negative_unemployment_ck = 0; // negative_unemployment_check

        Region = Regions(ks);
        //Downscaling Non coal employement
        L_downscaled = zeros(sec,TimeHorizon);

        for j = 1:sec


            // ======================
            // Convergence for ind_pop
            
            select ind_struct
            case 1
                L_downscaled(j,:) = matrix(L_ILO_fix(downscaled_countries(k),j,:),[1,TimeHorizon]).*Emp_struct(Region+1, j+4)*1e6;

            case 2
                L_downscaled(j,:) = matrix(L_ILO_fix(downscaled_countries(k),j,:),[1,TimeHorizon]).*(Emp_struct(Region+1, j+4).*(1-Shift_struct)+Emp_struct_pop(Region+1, j+4).*Shift_struct)*1e6;
            case 3
                // 
                for t = 1:TimeHorizon
                    year = Base_year+t-1;
                    sector_mapping = [0,0,0,0,0,1,2,0,0,0,3,1];
                    sector_mapping = sector_mapping(j);
                    select sector_mapping // In this exercise, we assume GRP grows uniformaly, so contributions of regions to transport sectors scale accordingly
                    case 1 // Industry
                        col_name = 'indus'+ string(year);
                        col_index = find(Emp_struct_chnge_index==col_name);
                        Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);
                    case 2 // Services
                        col_name = 'servi'+ string(year);
                        col_index = find(Emp_struct_chnge_index==col_name);
                        Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);

                    case 3 // Agriculture
                        col_name = 'agric'+ string(year);
                        col_index = find(Emp_struct_chnge_index==col_name);
                        Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);
                    end

                    L_downscaled(j,t) = L_ILO_fix(downscaled_countries(k),j,t)*Emp_struct(Region+1, j+4)*1e6;
                end
            case 4
                // 
                for t = 1:TimeHorizon
                    year = 2015;
                    sector_mapping = [0,0,0,0,0,1,2,0,0,0,3,1];
                    sector_mapping = sector_mapping(j);
                    select sector_mapping // In this exercise, we assume GRP grows uniformaly, so contributions of regions to transport sectors scale accordingly
                    case 1 // Industry
                        col_name = 'indus'+ string(year);
                        col_index = find(Emp_struct_chnge_index==col_name);
                        Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);
                    case 2 // Services
                        col_name = 'servi'+ string(year);
                        col_index = find(Emp_struct_chnge_index==col_name);
                        Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);

                    case 3 // Agriculture
                        col_name = 'agric'+ string(year);
                        col_index = find(Emp_struct_chnge_index==col_name);
                        Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);
                    end

                    L_downscaled(j,t) = L_ILO_fix(downscaled_countries(k),j,t)*Emp_struct(Region+1, j+4)*1e6;

                end
            case 5 
                year = 2015;
                sector_mapping = [0,0,0,0,0,1,2,0,0,0,3,1];
                sector_mapping = sector_mapping(j);
                select sector_mapping // In this exercise, we assume GRP grows uniformaly, so contributions of regions to transport sectors scale accordingly
                case 1 // Industry
                    col_name = 'indus'+ string(year);
                    col_index = find(Emp_struct_chnge_index==col_name);
                    Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);
                case 2 // Services
                    col_name = 'servi'+ string(year);
                    col_index = find(Emp_struct_chnge_index==col_name);
                    Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);

                case 3 // Agriculture
                    col_name = 'agric'+ string(year);
                    col_index = find(Emp_struct_chnge_index==col_name);
                    Emp_struct(Region+1, j+4) = Emp_struct_chnge(Region+1, col_index);
                end
                    L_downscaled(j,:) = matrix(L_ILO_fix(downscaled_countries(k),j,:),[1,TimeHorizon]).*(Emp_struct(Region+1, j+4).*(1-Shift_struct)+Emp_struct_pop(Region+1, j+4).*Shift_struct)*1e6;
            end

            for t = 1:TimeHorizon
                if sum(L_downscaled([6,7,11,12],t))~=0 then
                    SA_pr(Region,t) = L_downscaled(11,t)/sum(L_downscaled([6,7,11,12],t));
                    SM_pr(Region,t) = sum(L_downscaled([6,12],t))/sum(L_downscaled([6,7,11,12],t));
                    SS_pr(Region,t) = L_downscaled(7,t)/sum(L_downscaled([6,7,11,12],t));
                end
                TA_pr(Region,t) = L_downscaled(11,t);
                TM_pr(Region,t) = sum(L_downscaled([6,12],t));
                TS_pr(Region,t) = L_downscaled(7,t);
            end
            
        end

        NC_jobs = sum(L_downscaled(2:$,:),1);

        //Smoothing non coal employment results
        NC_jobs_temp = NC_jobs;
        
        if ind_smooth ==1 then
            for l = 2:TimeHorizon-1
                NC_jobs(l) =mean(NC_jobs_temp(l-1:l+1));
            end
        end

        Coal_jobs = Emp_coal(Region,:);
        //Smoothing coal employment results
        Coal_jobs_temp = Coal_jobs;
        if ind_smooth ==1 then
            for t = 2:TimeHorizon-1
                Coal_jobs(t) =mean(Coal_jobs_temp(t-1:t+1));

                if Coal_jobs(t)<0
                    disp('Calculated negative employment in Region  : '+string(Region)+'   at time step: '+string(t));
                end

            end
        end


        if min(Coal_jobs)<0
            disp('!!!!! Negative employment  : '+string(Region))
        end

        // Calculating total openings
        Total_openings = zeros(1,TimeHorizon);
        for t = 2:TimeHorizon
            Total_openings(t) = NC_jobs(t)-NC_jobs(t-1)*(1-txExit(downscaled_countries(k),t));
        end

        // Initializing Subnational Unemployment and Labour Force
        Unemployement = ones(1,TimeHorizon)*U(downscaled_countries(k),1)*Emp_struct(Region+1,17)*LF_ILO(downscaled_countries(k),1)*1e6;
        Labour_force = [Coal_jobs(1)+NC_jobs(1)+Unemployement(1)];
        

        // Coal job balance
        Coal_retirement = Coal_jobs(1)*Base_pyramid(:,$)';
        Pyramid = Coal_jobs(1)*Base_pyramid(:,:);
        Hired_pyramid =zeros(TimeHorizon,retirement_age-Entry_age+2);
        Fired_pyramid =zeros(TimeHorizon,retirement_age-Entry_age+2);



        Coal_destruction = zeros(1, TimeHorizon);
        Coal_hire = zeros(1, TimeHorizon);
        Coal_searching= zeros(1, TimeHorizon);
        Coal_temp= zeros(1, TimeHorizon); // Workers who left the coal sector but are still looking for a job
        Openings = zeros(1, TimeHorizon);

        for t= [2:TimeHorizon]
    
            for aging = [2:retirement_age-Entry_age+2] //Advancing each age class to the next
                Pyramid(t,aging)=Pyramid(t-1,aging-1);
            end

    
            Coal_destruction(t)=Coal_jobs(t-1)-Coal_jobs(t);
            
            
            Coal_retirement(t) = Pyramid(t,$);
            Coal_hire(t) = max(Coal_retirement(t)-Coal_destruction(t),0);
    
            Pyramid(t,1:5) = Pyramid(t,1:5)+Coal_hire(t)/5; //Adding hired workers to the age pyramid
    
            Coal_searching(t) = Coal_destruction(t)-Coal_retirement(t)+Coal_hire(t);
    
            if sum(Pyramid(t,:))==0  //Removing laid off workers from the pyramid (assumed to be distributed like other workers)
                Pyramid(t,:) = 0;
            else
                Pyramid(t,:) = Pyramid(t,:) - Coal_searching(t)*Pyramid(t,:)/sum(Pyramid(t,:));
            end
            
        end

        // Now going through each time step

        for t = 1:TimeHorizon

            Unemployement(t) = Labour_force(t)-(Coal_jobs(t)+NC_jobs(t)+Coal_temp(t));

            // Dealing with negative values of unemployment
            // Under the present framework, negative unemployment if regional employment grows faster than the active population
            // This is possible in regions with levels of employment in dynamic sectors (eg services)
            // This would in practice be associated with immigration towards those regions
            // At the same time, we would observe emigration from regions with higher unemployment which is not considered here, providing a conservative estimate

            if Unemployement(t)<0 then
                Negative_unemployment(downscaled_countries(k),t) = Negative_unemployment(downscaled_countries(k),t)-Unemployement(t);
                Negative_unemployment_pr(Region,t) = -Unemployement(t);
                Unemployement(t) = 0;               
                negative_unemployment_ck = 1;
            end
        
            // Calculating the job seeker from the coal sector
            // Coal workers compete with those who did not find employment the year prior, those already unemployed and the share of new labour market entrants
            if (Coal_searching(t)+Unemployement(t)+NC_jobs(t)*txEntr(downscaled_countries(k),t)+Coal_temp(t)) < Total_openings(t) then 
                Share_coal_search(Region,t) = 1;
                Share_temp_search(Region,t) = 1;

            elseif (Coal_searching(t)+Unemployement(t)+NC_jobs(t)*txEntr(downscaled_countries(k),t)+Coal_temp(t))~=0 then
                Share_coal_search(Region,t) = Coal_searching(t)/(Coal_searching(t)+Unemployement(t)+NC_jobs(t)*txEntr(downscaled_countries(k),t)+Coal_temp(t));
                Share_temp_search(Region,t) = Coal_temp(t)/(Coal_searching(t)+Unemployement(t)+NC_jobs(t)*txEntr(downscaled_countries(k),t)+Coal_temp(t));
                
            else
                Share_coal_search(Region,t) = 0;
            end



            // Available openings: available to coal workers
            Openings(t) = Total_openings(t)*Share_coal_search(Region,t);

            Instant_matches(t) = max(0,min(Openings(t),Coal_searching(t)));
            Delayed_matches(t) = max(0,min(sum(Total_openings(t))*Share_temp_search(Region,t),Coal_temp(t)));

            // Updating worker groups based on matches
            Coal_temp(t+1) = Coal_searching(t) - Instant_matches(t);

            Unemployed_workers = Coal_searching(t)-Instant_matches(t);
            Unfilled_openings  = max(Openings(t) - Instant_matches(t),0);

            
            if t ~=TimeHorizon then 
                Labour_force = [Labour_force,Labour_force($)*(1+txLact(downscaled_countries(k),t))];
            end
        end
        

        // Calculating the number of coal workers leaving into unemployment (of more than 1 year)
        Unemployed = [];
        for t = 1:TimeHorizon
            if t~=TimeHorizon then
                LeftU = Coal_searching(t)-Instant_matches(t)-Delayed_matches(t+1);
            else
                LeftU = Coal_searching(t)-Instant_matches(t);
            end
            Unemployed = [Unemployed,LeftU];
        end
       
        // Collecting data

        Qcoal_pr(Region,:) = Prod_coal(Region,:);
        Lcoal_pr(Region,:) = Coal_jobs;
        if min(Lcoal_pr)<0
            disp('found negative Lcoal_pr')
        end
        Produ_pr(Region,:) = Productivity(Region,:);

        Hire_pr(Region,:) = Coal_hire;
        Retire_pr(Region,:) = Coal_retirement;
        Instant_M_pr(Region,:) = Instant_matches';
        Delayed_M_pr(Region,:) = Delayed_matches';
        Unemployed_pr(Region,:) = Unemployed;

        Ct_pr(Region,:) = Coal_temp;
        CS_pr(Region,:) = Coal_searching;
        CD_pr(Region,:) = Coal_destruction;
        Openings_pr(Region,:) = Openings;
        Unemployement_pr(Region,:) = Unemployement;
        TOpenings_pr(Region,:) = Total_openings;
        LF_pr(Region,:) = Labour_force;
        NC_pr(Region,:) = NC_jobs;
      
        if negative_unemployment_ck ~= 0 then
            mfprintf(fileID,'Negative unemployment: '+string(Region_code(Region))+"\n");
        end
    end
end



t = [2015:2101];
if ind_unemployment ==1
    mfprintf(fileID,"Maximum unaccounted workers in China: "+"\n");
    mfprintf(fileID,string(max(Negative_unemployment(6,1:70)))+"\n");
    mfprintf(fileID,"Maximum unaccounted share of labour force in China:"+"\n");
    mfprintf(fileID,string(...
                round(max(Negative_unemployment(6,1:70)./LF_pr(1,1:70))*1e4)/1e2)+...
                "percent"+"\n");
    mfprintf(fileID,"Maximum unaccounted workers in India:"+"\n");
    mfprintf(fileID,string(max(Negative_unemployment(7,1:70)))+"\n");
    mfprintf(fileID,"Maximum unaccounted share of labour force in India:"+"\n");
    mfprintf(fileID,string(...
                round(max(Negative_unemployment(7,1:70)./LF_pr(33,1:70))*1e4)/1e2)+...
                "percent"+"\n");
end
mfprintf(fileID,"\n\n\n\n"); 