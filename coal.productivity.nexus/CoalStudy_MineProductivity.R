# Data preparation script for the coal sector productivity trajectory module
# (bottom-up approach with spatial disaggregation)
# Used for the ERL paper on India and China (under review)

# Inputs:
# - GEM coal mine database (April 2024 release + September 2024 supplement)
# - Coal price data and energy content by coal grade
# - Wage data at the province/state level
# - Coal reserve data by state/province

# Output:
# - A .csv file to be used by the .sce module provided

# Parameters to define:
# - ind_opt_newmines: option controlling the optimism level for the productivity of newly created mines in the first years
#   (0: equal to the average productivity of mines commissioned between 2018 and 2023;
#    1: +100% relative to this average)

# 0. Packages -------------------------------------------------------------
# Core packages used in the main pipeline. Optional packages are loaded
library(dplyr)
library(tidyr)
library(stringr)
library(openxlsx)

# 1. User-defined paths / options ----------------------------------------
# NOTE: avoid hard-coded setwd() when sharing the code. Adjust `data_dir`
data_dir <- "To be defined"

ind_opt_newmines <- 0
if (ind_opt_newmines == 1) {
  mult_productivity_newmines <- 2
  suffix_hyp_productivity <- "_P100"
} else {
  mult_productivity_newmines <- 1
  suffix_hyp_productivity <- "_P0"
}

# 2. Load and pre-process external databases -----------------------------
# GEM (April 2024)
GEMapril2024 <- as.data.frame(
  read.xlsx(file.path(data_dir, "Global-Coal-Mine-Tracker-April-2024.xlsx"),
            sheet = "Global Coal Mine Tracker (Non-C",
            colNames = TRUE, rowNames = FALSE)
) %>%
  mutate(ID_uniq = row_number())

# GEM supplement (September 2024): historical production (non-China + China)
GEMsupplement2024 <- as.data.frame(
  read.xlsx(file.path(data_dir, "Global-Coal-Mine-Tracker-September-2024-Supplement-v2.xlsx"),
            sheet = "Historical Production (non-Chin",
            colNames = TRUE, rowNames = FALSE)
) %>%
  { setNames(., make.unique(names(.))) } %>%
  select(GEM.Mine.ID, contains("Coal.Output.(Annual,.Mt)")) %>%
  pivot_longer(cols = -GEM.Mine.ID, names_to = "year", values_to = "Coal.Output") %>%
  mutate(year = as.numeric(str_sub(year, -4, -1)))

GEMsupplement2024_China <- as.data.frame(
  read.xlsx(file.path(data_dir, "Global-Coal-Mine-Tracker-September-2024-Supplement-v2.xlsx"),
            sheet = "Historical Production (China)",
            colNames = TRUE, rowNames = FALSE)
) %>%
  { setNames(., make.unique(names(.))) } %>%
  select(GEM.Mine.ID, contains("Coal.Output.(Annual,.Mt)")) %>%
  pivot_longer(cols = -GEM.Mine.ID, names_to = "year", values_to = "Coal.Output") %>%
  mutate(year = as.numeric(str_sub(year, -4, -1)))

# Production correction heuristic: keep the maximum historical output for each mine
# (used to replace suspiciously low 'snapshot' outputs in the April GEM release).
GEMoutput <- rbind(GEMsupplement2024, GEMsupplement2024_China) %>%
  mutate(Coal.Output = as.numeric(Coal.Output)) %>%
  group_by(GEM.Mine.ID) %>%
  mutate(Coal.Output.Max = if_else(all(is.na(Coal.Output)), NA_real_, max(Coal.Output, na.rm = TRUE))) %>%
  ungroup() %>%
  distinct(GEM.Mine.ID, Coal.Output.Max)

CoalReserves <- as.data.frame(
  read.xlsx(file.path(data_dir, "CoalReserve_summary.xlsx"),
            sheet = "summary", colNames = TRUE, rowNames = FALSE, rows = 1:41, cols = 1:2)
)
CoalWages_China <- as.data.frame(
  read.xlsx(file.path(data_dir, "ProductivityStudy_SubNationalWages.xlsx"),
            sheet = "China", colNames = TRUE, rowNames = FALSE, rows = 3:34, cols = 1:3)
)
CoalWages_India <- as.data.frame(
  read.xlsx(file.path(data_dir, "ProductivityStudy_SubNationalWages.xlsx"),
            sheet = "India", colNames = TRUE, rowNames = FALSE, rows = 3:16, cols = 1:3)
)
CoalPrices <- as.data.frame(
  read.xlsx(file.path(data_dir, "CoalAttributes_data_V1.xlsx"),
            sheet = "Prices", colNames = TRUE, rowNames = FALSE)
)
CoalWages <- rbind(CoalWages_China, CoalWages_India)

# Variable list used for numeric conversion (and for the optional decision-tree annex)
ListeVariablesTree <- read.csv(file.path(data_dir, "ListVariables.csv"), header = TRUE, sep = ";")

# 3. Build mine database for Imaclim-R -----------------------------------
GEM <- GEMapril2024 %>%
  filter(Country %in% c('China','India')) %>%
  mutate(Workforce.Size.Corrected = as.numeric(Workforce.Size.Corrected), Workforce.Size = as.numeric(Workforce.Size)) %>%
  mutate(Workforce.Size = if_else(is.na(Workforce.Size.Corrected)==TRUE,Workforce.Size,Workforce.Size.Corrected)) %>%
  rename (Coal.Output = `Production.(Mtpa)`, Capacity = `Capacity.(Mtpa)`, State.Province = `State,.Province`) %>%
  mutate_at(c(ListeVariablesTree[which(ListeVariablesTree$As.numeric == "x"),1]), as.numeric) %>%
  mutate_at(vars(`Total.Resource.(Inferred,.Indicated,.Measured)`, `Mine.Depth.(m)`, `Mine.Size.(Km2)`, Reported.Life.of.Mine, `Total.Reserves.(Proven.and.Probable,.Mt)`, `Reserve.to.Production.Ratio.(R/P)`),as.numeric) %>%
  filter(!(is.na(Capacity) & is.na(Coal.Output))) %>%
  mutate(Coal.Grade = str_to_title(Coal.Grade)) %>%
  mutate(State.Province.agg = if_else(State.Province %in% c("Shanxi", "Inner Mongolia", "Shaanxi", "Xinjiang", "Shandong","Guizhou", "Henan", "Jharkhand", "Odisha", "Chhattisgarh","West Bengal", "Madhya Pradesh", "Telangana", "Maharashtra", "Rajasthan"),State.Province,"Other")) %>%
  mutate(State.Province.agg = if_else(State.Province.agg == "Other" & Country == "India", "Other.India",if_else(State.Province.agg == "Other" & Country == "China", "Other.China",State.Province))) %>%
  mutate(Considered.Output = pmax(Coal.Output,if_else(Country == "India", 0.855, 0.75)*Capacity, na.rm = TRUE)) %>%
  left_join(GEMoutput %>% select(GEM.Mine.ID, Coal.Output.Max), by = "GEM.Mine.ID") %>%
  mutate(Considered.Output = ifelse(is.na(Coal.Output.Max) == FALSE, ifelse(Considered.Output < Coal.Output.Max, Coal.Output.Max,Considered.Output),Considered.Output)) %>%
  mutate(Considered.Reserve = if_else(Country == "China", pmin(`Total.Reserves.(Proven.and.Probable,.Mt)`, Reported.Life.of.Mine * Considered.Output, `Reserve.to.Production.Ratio.(R/P)` * Considered.Output, `Total.Resource.(Inferred,.Indicated,.Measured)`, na.rm = TRUE),pmax(`Total.Reserves.(Proven.and.Probable,.Mt)`, Reported.Life.of.Mine * Considered.Output, `Reserve.to.Production.Ratio.(R/P)` * Considered.Output, `Total.Resource.(Inferred,.Indicated,.Measured)`, na.rm = TRUE)) ) %>%
  mutate (Labour.Productivity = Considered.Output/Workforce.Size*10^6) %>%
  arrange(GEM.Mine.ID,Status) %>%
  group_by(GEM.Mine.ID,Status) %>%
  mutate(Workforce.Size = if_else((n() > 1 & Status %in% c("Proposed","Shelved")), Considered.Output / lag(Labour.Productivity) * 10^6, Workforce.Size)) %>%
  ungroup() %>%
  mutate (Labour.Productivity = if_else(Considered.Output/Workforce.Size*10^6>50000,50000,Considered.Output/Workforce.Size*10^6)) %>%
  
  (\(x) { assign("GEM_step1", x, envir = .GlobalEnv); x }) %>%  
  
  # II.2. Completion of missing Labour.Productivity data for projects
  left_join(GEM_step1 %>%filter(Opening.Year >= 2014, Status == "Operating") %>%
              group_by(Country) %>%
              mutate(country_Labour.Productivity = weighted.mean(Labour.Productivity, w = Considered.Output, na.rm = TRUE)) %>% 
              select(Country, country_Labour.Productivity) %>%
              distinct(),
            by = c("Country")
  ) %>%
  left_join(GEM_step1 %>%filter(Opening.Year >= 2014, Status == "Operating") %>%
              group_by(Country, State.Province.agg) %>%
              mutate(regional_Labour.Productivity = weighted.mean(Labour.Productivity, w = Considered.Output, na.rm = TRUE)) %>%
              select(Country, State.Province.agg,regional_Labour.Productivity) %>%
              distinct(),
            by = c("Country", "State.Province.agg")
  ) %>%
  left_join(GEM_step1 %>%filter(Opening.Year >= 2014, Status == "Operating") %>%
              group_by(Country, State.Province.agg,Mine.Type) %>%
              mutate(minetype_Labour.Productivity = weighted.mean(Labour.Productivity, w = Considered.Output, na.rm = TRUE)) %>%
              select(Country, State.Province.agg, Mine.Type, minetype_Labour.Productivity) %>%
              distinct(),
            by = c("Country", "State.Province.agg","Mine.Type")
  ) %>%
  left_join(GEM_step1 %>%filter(Opening.Year >= 2014, Status == "Operating") %>%
              group_by(Country, State.Province.agg,Mine.Type,Coal.Grade) %>%
              mutate(coalgrade_Labour.Productivity = weighted.mean(Labour.Productivity, w = Considered.Output, na.rm = TRUE)) %>%
              select(Country, State.Province.agg, Mine.Type, Coal.Grade, coalgrade_Labour.Productivity) %>%
              distinct(),
            by = c("Country", "State.Province.agg","Mine.Type","Coal.Grade")
  ) %>%
  
  mutate(
    Labour.Productivity.prev = Labour.Productivity,
    Labour.Productivity = coalesce(Labour.Productivity, coalgrade_Labour.Productivity, minetype_Labour.Productivity, regional_Labour.Productivity, country_Labour.Productivity),
  ) %>%
  
  (\(x) { assign("GEM_step2", x, envir = .GlobalEnv); x }) %>%
  
  # II.3. We complement with several useful variables from other databases (e.g. Heat.Content, Output.Price, Wages) or from calculations (e.g. Index.Performance)
  left_join(CoalWages, by = 'State.Province') %>%
  #mutate(Considered.Output = pmax(ifelse(is.na(Coal.Output)==TRUE,0.7*Capacity,Coal.Output),0.6*`Total.Resource.(Inferred,.Indicated,.Measured)`/Reported.Life.of.Mine, na.rm = TRUE)) %>% # We apply a correction: some mines show reported production that seems inconsistent with other attributes (e.g. Anhui_Banji_Coal_Mine, which has 2,700 employees and 506 Mt of reserves but only 21 years of reported life); this aims at correcting these data.
  mutate (Workforce.Size = Considered.Output/Labour.Productivity*10^6) %>%
  mutate (Coal.Grade = if_else(Coal.Grade == "thermal","Thermal",Coal.Grade)) %>%
  mutate (Coal.Grade = if_else(!Coal.Grade %in% c("Thermal","Thermal & Met","Met"),"Unknown",Coal.Grade)) %>%
  left_join(CoalPrices[,c('Coal.Type','Coal.Grade','Output.Price_dollar.per.ton','Heat.Content')], by = c('Coal.Type','Coal.Grade')) %>%
  mutate(Heat.Content = if_else(is.na(Heat.Content == TRUE), 25, Heat.Content)) %>%
  mutate(Considered.Output.GJ = Considered.Output*Heat.Content, Considered.Reserve.GJ = Considered.Reserve*Heat.Content, Labour.Productivity.GJ = Labour.Productivity * Heat.Content) %>%
  group_by(Country) %>%
  mutate(Output.Price_dollar.per.ton = ifelse(Coal.Grade == "Unknown",Output.Price_dollar.per.ton[(Coal.Grade == "Thermal")][1],Output.Price_dollar.per.ton)) %>%
  ungroup() %>%
  mutate(Output.in.value = Considered.Output * Output.Price_dollar.per.ton*10^6) %>%
  mutate(Annual.Wage.Value = as.numeric(Annual.Wage.Value)) %>%
  mutate(Labour.Cost = Annual.Wage.Value * Considered.Output/Labour.Productivity*10^6)  %>%
  mutate(Index.Performance = Output.in.value/Labour.Cost) %>%
  group_by(Country) %>%
  mutate(Rank = rank(-Index.Performance, ties.method = "first")) %>%
  ungroup() %>%
  mutate(Active = ifelse(Status == "Operating",1,0)) %>%
  filter(! (is.na(Capacity) & is.na(Coal.Output))) %>%
  mutate(Share.New.Output = NA_real_) 

# II.4. We compute the different parameters of "fictive mines", i.e. installations not present in the original database
Fictive.New.Mines <- GEM %>%
  
  left_join(GEM %>%
              filter(Opening.Year >= 2017 & Opening.Year <= 2024) %>%
              group_by(Country, State.Province) %>%
              summarise(total_NewCapacity_subnat_2017_2024 = sum(Considered.Output, na.rm = TRUE), .groups = 'drop'),by = c("Country", "State.Province")) %>%
  left_join(GEM %>%
              filter(Status %in% "Proposed") %>%
              group_by(Country, State.Province) %>%
              summarise(total_projectedcapacity_subnat = sum(Capacity, na.rm = TRUE), .groups = 'drop'),by = c("Country", "State.Province")) %>%
  left_join(GEM %>%
              filter(Opening.Year >= 2017 & Opening.Year <= 2024) %>%
              group_by(Country) %>%
              summarise(total_NewCapacity_nat_2017_2024 = sum(Considered.Output, na.rm = TRUE), .groups = 'drop'),by = c("Country")) %>%
  left_join(GEM %>%
              filter(Status %in% "Proposed") %>%
              group_by(Country) %>%
              summarise(total_projectedcapacity_nat = sum(Capacity, na.rm = TRUE), .groups = 'drop'),by = c("Country")) %>%
  (\(x) { assign("FictiveNewMines_intermediaire", x, envir = .GlobalEnv); x }) %>%  
  
  # Compute the share of each State.Province’s production relative to the national total
  mutate(Share.New.Capacity = (total_NewCapacity_subnat_2017_2024+total_projectedcapacity_subnat) / (total_NewCapacity_nat_2017_2024+total_projectedcapacity_nat)) %>%
  
  filter(Opening.Year >= 2017,Status == "Operating") %>%
  group_by(Country, State.Province,Share.New.Capacity) %>%
  summarise(Labour.Productivity = mean(Labour.Productivity, weights = Considered.Output, na.rm = TRUE), Labour.Productivity.GJ = mean(Labour.Productivity.GJ, weights = Considered.Output.GJ, na.rm = TRUE), Considered.Output = sum(Considered.Output, na.rm = TRUE), Considered.Output.GJ = sum(Considered.Output.GJ, na.rm = TRUE),Output.Price_dollar.per.ton = mean(Output.Price_dollar.per.ton, weights = Considered.Output, na.rm = TRUE),Annual.Wage.Value = mean(Annual.Wage.Value, na.rm = TRUE),.groups = 'drop') %>%
  mutate(Labour.Productivity = mult_productivity_newmines*Labour.Productivity, Labour.Productivity.GJ = mult_productivity_newmines*Labour.Productivity.GJ) %>%
  ungroup() %>%
  
  group_by(Country) %>%
  mutate(Heat.Content = Considered.Output.GJ/Considered.Output) %>%
  mutate(sum_CoalOutput = sum(Considered.Output.GJ, na.rm = TRUE)) %>%
  mutate(Share.New.Capacity_prev = Considered.Output.GJ/sum_CoalOutput) %>%
  ungroup() %>%
  select(-sum_CoalOutput) %>%
  mutate(Status = "Fictive") %>%
  left_join(CoalReserves, by = "State.Province") %>%
  mutate(Output.in.value = Considered.Output * Output.Price_dollar.per.ton*10^6) %>%
  mutate(Annual.Wage.Value = as.numeric(Annual.Wage.Value)) %>%
  mutate(Labour.Cost = Annual.Wage.Value * Considered.Output/Labour.Productivity*10^6)  %>%
  rename(Considered.Reserve = "Reserve.Mt") %>%
  mutate(Considered.Reserve.GJ = Considered.Reserve*Heat.Content) %>%
  mutate(Index.Performance = Output.in.value/Labour.Cost)

GEM <- GEM %>% bind_rows(Fictive.New.Mines) %>%
  group_by(State.Province) %>%
  mutate(Annual.Wage.Value = mean(Annual.Wage.Value, na.rm =TRUE),Output.Price_dollar.per.ton = mean(Output.Price_dollar.per.ton, weights = Coal.Output, na.rm =TRUE)) %>%
  ungroup() %>%
  mutate(across(where(is.character), ~ ifelse(is.na(.), "NA", .)))

# We select and reorder the colnames and then export the database (note that the commented lines will be used in a more complex and dynamic version of the nexus)
GEM_export_imaclim <- GEM %>%
  filter(Country %in% c("India","China")) %>%
  #select(ID_uniq,Active,Mine.Name,Mine.Name.AKAs,Country,State.Province,Status,Status.Detail,Coal.Output,Workforce.Size,Mine.Type,Mining.Method,Index.Performance,Rank,Mine.Size.(Km2),Mine.Depth.(m),Coal.Type,Coal.Grade,Total.Reserves.(Proven.and.Probable),Total.Resource.(Inferred,.Indicated,.Measured),Reserve.to.Production.Ratio.(R/P),Opening.Year,Reported.Life.of.Mine,Index.Wage,Annual.Wage.Value,Labour.Productivity,Output.Price_dollar.per.ton,Output.in.value,Labour.Cost) %>%
  select(GEM.Mine.ID,Active,Mine.Name,Country,State.Province,Status,Status.Detail,Coal.Output,Workforce.Size,Mine.Type,Mining.Method,Considered.Output,Considered.Output.GJ, Considered.Reserve,Considered.Reserve.GJ,Annual.Wage.Value,Labour.Productivity,Labour.Productivity.GJ,Index.Performance,Rank,Share.New.Capacity,Heat.Content,Output.Price_dollar.per.ton)

#colnames(GEM_export_imaclim) <- c('ID_uniq','Active','Mine.Name','Country','State.Province','Status','Status.Detail','Coal.Output','Workforce.Size','Mine.Type','Mining.Method','Index.Performance','Rank','Mine.Size.(Km2)','Mine.Depth.(m)','Coal.Type','Coal.Grade','Total.Reserves.(Proven.and.Probable)','Total.Resource.(Inferred,.Indicated,.Measured)','Reserve.to.Production.Ratio.(R/P)','Opening.Year','Reported.Life.of.Mine','Index.Wage','Annual.Wage.Value','Labour.Productivity','Output.Price_dollar.per.ton','Output.in.value','Labour.Cost')
colnames(GEM_export_imaclim) <- c('GEM.Mine.ID','Active','Mine.Name','Country','State.Province','Status','Status.Detail','Coal.Output','Workforce.Size','Mine.Type','Mining.Method','Considered.Output', 'Considered.Ouput.GJ', 'Considered.Reserve','Considered.Reserve.GJ','Annual.Wage.Value','Labour.Productivity','Labour.Productivity.GJ','Index.Performance','Rank', 'Share.New.Capacity','Heat.Content','Output.Price_dollar.per.ton')
write.table(GEM_export_imaclim, str_c("CoalEconomy_newdata_GEM_maxOutput_ResHigh_",ifelse(ind_opt_newmines ==1, "P100","P0"),".csv"), row.names=FALSE, sep="|",dec=".", na="NA", col.names = TRUE,quote =TRUE)
