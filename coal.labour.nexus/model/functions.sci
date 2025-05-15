function Data=InterpolatingYearlyData(Data)

empty_years = [];
// Works only if no "AR6 ...|..." variables data: these are there for every year
for i = 6:size(Data, 2)
  if sum(Data(2:$, i) == "") == size(Data(2:$, i), 1) then
      empty_years = [empty_years, Data_T(i-5)];  
  end
end

Data = strtod(Data(:,6:$));

if length(empty_years) ==0;
  disp("The data is complete");
  
else
  disp("Completing data");
  full_years = setdiff(Data_T,empty_years)
  empty_years_temp = []
  for year = empty_years
      low = full_years(evstr(full_years)<evstr(year));   
      low = low($);

      if length(low)~=0
          empty_years_temp = [empty_years_temp,year];
          high = full_years(evstr(full_years)>evstr(year));   
          high = high(1);
          for row = 1:size(Data,1)
              xs = [evstr(low),evstr(high)];
              ys = [Data(row,find(Data_T==low)),Data(row,find(Data_T==high))]; 
              Data(row,find(Data_T==year)) = interpln([xs;ys],evstr(year));
          end
      end

  end


end

  
endfunction