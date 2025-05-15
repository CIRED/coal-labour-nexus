// Changing region names for downscale regions to correspond to region name from model source
China_name = "China";
India_name = "India";


regnames(ind_chn)=China_name;
regnames(ind_ind)=India_name;


if ~isempty(strindex(convstr(Scenari_up(1),'l'),'npi'))
    infill_source = "../Infilling_trajectories/Infilling_coal_labour_NPI.csv";
elseif ~isempty(strindex(convstr(Scenari_up(1),'l'),'ndc'))
    infill_source = "../Infilling_trajectories/Infilling_coal_labour_NDC.csv";
elseif ~isempty(strindex(convstr(Scenari_up(1),'l'),'1.5'))
    infill_source = "../Infilling_trajectories/Infilling_coal_labour_NZ.csv";
else
    disp("No infilling scenarios found")
end

// Searching for employment variables

emp_variables =['Employment|Agriculture','Employment|Service','Employment|Industry'];
sector_indices = [indice_agriculture,indice_composite,indice_industries];
lv=length(sector_indices)
for ind_var = 1:length(sector_indices)

    variable = emp_variables(ind_var);
    j = sector_indices(ind_var);

    if length(find(Variabs_up==variable))~=0
        disp(variable+' available')

        for k =1:n_downscaled_countries
            Cname = regnames(downscaled_countries(k));
            index = find((Variabs_up==variable)&(Regions_up==Cname));
            L_ILO_fix(downscaled_countries(k),j,:) =  Data(index,find(Data_T=='2015'):$);            
        end

    end
end


// Searching for and downscaling coal production
for k =1:n_downscaled_countries
    Cname = regnames(downscaled_countries(k))
    // Downscaling coal production
    Regions = find(Emp_struct(:, 1) == downscaled_countries(k));
    index = find(Regions_up==Cname & Variabs_up=='Resource|Extraction|Coal')
    if length(index)>0
        disp('Using Resource|Extraction|Coal variable');
        for ks = 1:length(Regions)
            Region = Regions(ks);      
            Prod_coal = [Prod_coal; (Data(index,find(Data_T=='2015'):$))*Emp_struct(Region,5)/mtoe2ej];
        end
    else
        c_index = find(Regions_up==Cname & Variabs_up=='Primary Energy|Coal');
        t_index = find(Regions_up==Cname & Variabs_up=="Trade|Primary Energy|Coal|Volume");
        if (length(c_index)==0)|(length(t_index)==0)
            disp('ERROR: Not enough information to infer regional coal production which is necessary for module to work')
        else
            disp('Using sum of primary energy consumption and trade');
            for ks = 1:length(Regions)
                Region = Regions(ks);
                Prod_coal = [Prod_coal; (Data(c_index,:)+..
                                        Data(t_index,:))*Emp_struct(Region,5)];
        
            end
        end
    end
    disp("Obtained coal production figures from scenario data");
    disp("Size Prod_coal = "+string(size(Prod_coal)));
end

// Searching for Unemployment variables
variable = 'Unemployment'
if length(find(Variabs_up==variable))~=0
    disp(variable+' available')
    for k =1:n_downscaled_countries
        Cname = regnames(downscaled_countries(k));
        index = find((Variabs_up==variable)&(Regions_up==Cname));
        U(downscaled_countries(k),:)=Data(index,find(Data_T=='2015'):$);
    end
elseif length(find(Variabs_up=="Unemployment|Rate"))~=0
    disp("Getting unemployment from unemployment rate")

else 
    disp("Getting unemployment from infilling trajectories");
    variable = "Unemployment|Rate"
    //=======================================================================================================================
    // Fetching infilling data should be moved to functions to avoid repetitions
    //=======================================================================================================================
    infilling_data = csvRead(infill_source,',','[]',"string");
    infilling_head = infilling_data(1,:);
    infilling_T    = infilling_head(6:$);
    infilling_data = infilling_data(2:$,:);
    infilling_regi = infilling_data(:,3) 
    infilling_vari = infilling_data(:,4)
    
    for k =1:n_downscaled_countries
        Cname = regnames(downscaled_countries(k));
        index = find((infilling_vari==variable)&(infilling_regi==Cname));
        U(downscaled_countries(k),:) = infilling_data(index,find(Data_T=='2015'):$);
        disp(U(downscaled_countries(k),8))
    end
end
