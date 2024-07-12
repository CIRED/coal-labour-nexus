
if ind_gem == 1
    Emp_2015_2021 = csvRead('../data/Coal_labour/Downscaling/Coal_jobs_2015_2021.csv');
else
    mfprintf(fileID,"GEM"+"\n");
    Emp_2015_2021 = csvRead('../data/Coal_labour/Downscaling/Coal_jobs_2015_2021_GEM.csv');
end


Productivity = zeros(size(Prod_coal,1),TimeHorizon);
Emp_coal = zeros(size(Prod_coal,1),TimeHorizon);





Emp_coal(:,1) = Emp_2015_2021(2:67,5);

// Productivity(:,1:7) = Prod_coal(:,1:7)./Emp_2015_2021(2:67,2:8)

for Country = [6, 7]
    Regions = find(Emp_struct(:, 1) == Country);
    for i = 1:length(Regions)
        Region = Regions(i)-1;
        if Emp_2015_2021(Region+1,5)==0 
            Productivity(Region,1)=0;
        else
            Productivity(Region,1)=  Prod_coal(Region,1)/Emp_2015_2021(Region+1,5);
        end
    end
end

// Calculating the productivity of coal mining assuming it grows at a constant rate in all regions of a country
// Productivity from 2015-2021 in China is considered exogeneous and equal to that calculated/interpolated with real world data
// This will output the real world number of coal workers in that period as we calibrate coal miners in the period on China statistical yearbook data
// In India, however, as data is much more scarce, productivity is calibrated such that productivity yields the number of coal workers in different states reported by Pai  
Productivity_growth_rate = productivity_growth;
Lev = 0.5;
lambda = 5/100;

for Country = [6, 7]
    Regions = find(Emp_struct(:, 1) == Country);
    for i = 1:length(Regions)
        Region = Regions(i)-1;
        for l = 2:6  // Forcing 5.8% growth rate until 2023
            if Country == 6 // In China, 2015-2022 productivity is exogeneous and equal to the observed CAGR to allow for better fit with real world data
                Productivity(Region,l)=Productivity(Region,l-1)*(1+10.78/100); 
            else
                Productivity(Region,l)=Productivity(Region,l-1)*(1+5.8/100); 
            end
        end
        for l = 7:TimeHorizon
            Productivity(Region,l)=Productivity(Region,l-1)*(1+Productivity_growth_rate/100); 
        end
    end 
end


// Calculate employment


for Country = [6, 7]
    Regions = find(Emp_struct(:, 1) == Country);
    for i = 1:length(Regions)
        Region = Regions(i)-1;
        for l =1:TimeHorizon
            if Productivity(Region,l)==0 
                Emp_coal(Region,l)=0;
            else
                Emp_coal(Region,l)=  Prod_coal(Region,l)/Productivity(Region,l);
            end
        end
    end
end