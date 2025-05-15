// Initializing arrays of Imaclim data
L_ILO = zeros(reg,sec,TimeHorizon);
L_ILO_tot = zeros(reg,TimeHorizon);
L_ILO_fix= zeros(reg,sec,TimeHorizon);
L_ILO_tot_fix = zeros(reg,TimeHorizon);
LF_ILO = zeros(reg,TimeHorizon);

Sectors = ['Coal','Oil','Gas','ET','Elec','BTP','Services','Air','Mer','Ot','Agri','Indus'];

// Importing employment data from Imaclim
E_tot_ILO =zeros(2,TimeHorizon);
for k = downscaled_countries
    Cname = regnames(k);
    
    for j = 1:sec
        L_ILO(k,j,:) = Data(find(Regions_up==Cname & Variabs_up=='Employment|ILO|'+Sectors(j)) ,:);
    end

    L_ILO_tot(k,:) = sum(L_ILO(k,:,:),2);


end


// Importing unemployment from Imaclim
for k = 1:reg
    Z(k,:) = Data(find(Regions_up==regnames(k) & Variabs_up=='Z') ,:);

    
    if ind_smooth ==1 then
        Z_temp = Z(k,:);

        for t = 2:TimeHorizon-1
            Z(k,t) = mean(Z_temp(t-1:t+1));
        end
    end
end


// Initializing unemployment throughout time horizon and deducting employment
for k =1:n_downscaled_countries

    Cname = regnames(downscaled_countries(k));

    LF0 = [780709584e-6,476681505e-6]; // Data from World Bank World Development Indicators database (2015)
    LF_ILO(downscaled_countries(k),1) = LF0(k);

    for t = 1:TimeHorizon-1
        LF_ILO(downscaled_countries(k),t+1) = LF_ILO(downscaled_countries(k),t) * (txLact(downscaled_countries(k),t)+1);
    end
    

    U0 = [4.7e-2,7.9e-2]; // Data from ILO obtained in World Bank's World Development Indicators database (2015)
    U0 = U0(k);

    U(downscaled_countries(k),:) = Z(downscaled_countries(k),:)*U0/Z(downscaled_countries(k),1); // This is the unemployment rate
    
    L_ILO_tot_fix(downscaled_countries(k),:) = LF_ILO(downscaled_countries(k),:) .* (1-U(downscaled_countries(k),:));
end


for k =1:n_downscaled_countries
    Cname = regnames(downscaled_countries(k));
    for j = 1:sec
        L_ILO_fix(downscaled_countries(k),j,:) = matrix(L_ILO(downscaled_countries(k),j,:),[1,TimeHorizon]).*L_ILO_tot_fix(downscaled_countries(k),:)./L_ILO_tot(downscaled_countries(k),:);
    end  


    // Downscaling coal production
    Regions = find(Emp_struct(:, 1) == downscaled_countries(k));
    Regions = find(Pr_regi_c == downscaled_countries(k) & Pr_variabs=="CoalOutputYear_reg" & scenario_type==Pr_scenario & productivity_scenario==Pr_prscenario);
    Country = Regions(1);
    national_prod = Produ(Country,:);
    // We assume that national 2015-2021 coal production is distributed similarly to 2021 in the GEM database
    // This allows us to neglect regions where production has phased out since (eg Beijing) and are thus irrelevant to our analysis 
    national_prod(1:6) = national_prod(7) ;

    for ks = 1:length(Regions)
        Region = Regions(ks);
        regional_prod = Produ(Region,:);
        regional_prod(1:6) = regional_prod(7) ;
        prod_share = regional_prod./national_prod;

        if productivity_growth==0 // If there is not productivity growth, we assume that coal production follows a constant distribution across both countries as variation cannot be explained by varying productivity growth
            Prod_coal = [Prod_coal; (Data(find(Regions_up==Cname & Variabs_up=='Primary Energy|Coal'),:)+..
                                    Data(find(Regions_up==Cname & Variabs_up=='Trade|Primary Energy|Coal|Volume'),:)).*prod_share(1)];
        else
            Prod_coal = [Prod_coal; (Data(find(Regions_up==Cname & Variabs_up=='Primary Energy|Coal'),:)+..
                                    Data(find(Regions_up==Cname & Variabs_up=='Trade|Primary Energy|Coal|Volume'),:)).*prod_share];
        end
    end

end


if ~isdef("ind_preparing_infilling")
    ind_preparing_infilling = 0;
end
if ind_preparing_infilling == 1
    infilling_path = '../Infilling_trajectories/'

    Infilling_Result = ['Model','Scenario','Region','Variable','Unit',string([2015:2100])];
    infilling_regions = ['China','India'];

    ind_indu = [indice_oil,indice_gaz,indice_Et,indice_elec,indice_construction,indice_industries];
    ind_serv= [indice_composite,indice_air,indice_mer,indice_OT];

    for k =1:n_downscaled_countries
        
        Emp_agri = matrix(L_ILO_fix(downscaled_countries(k),indice_agriculture,:),[1,TimeHorizon]);
        Emp_indu = sum(matrix(L_ILO_fix(downscaled_countries(k),ind_indu,:),[size(ind_indu,2),TimeHorizon]),1);
        Emp_serv = sum(matrix(L_ILO_fix(downscaled_countries(k),ind_serv,:),[size(ind_serv,2),TimeHorizon]),1);

        Infilling_Result = [Infilling_Result;
                          ['Imaclim V2.0 - Infilling for Coal Labour Nexus',runname,infilling_regions(k),'Employment|Agriculture','People',string(Emp_agri)];
                          ['Imaclim V2.0 - Infilling for Coal Labour Nexus',runname,infilling_regions(k),'Employment|Industry','People',string(Emp_indu)];
                          ['Imaclim V2.0 - Infilling for Coal Labour Nexus',runname,infilling_regions(k),'Employment|Services','People',string(Emp_serv)];
                          ['Imaclim V2.0 - Infilling for Coal Labour Nexus',runname,infilling_regions(k),'Unemployment|Rate','[-]',string(U(downscaled_countries(k),:))]];
    end

    csvWrite(Infilling_Result, infilling_path+'/Infilling_coal_labour_'+runname+'.csv');

end


