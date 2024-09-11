# Import libraries and data
library("plm")
library(ggplot2)
library(gridExtra)
library(stargazer)
library(tidyverse)



# Read data
file <- "structural.change\\data\\2023 - Wenz et al - DOSE Global dataset of reported subnational economic output.csv"
data1 <- read.csv(file)



file_paths <- c(
    "coal.labour.nexus\\input\\IMACLIM_waysout_outputs_WO-NPi-ElecIndus.csv",
    "coal.labour.nexus\\input\\IMACLIM_waysout_outputs_WO-NDCLTT-ElecIndus.csv",
    "coal.labour.nexus\\input\\IMACLIM_waysout_outputs_WO-15C-ElecIndus.csv"
)


process_file <- function(file_path) {
  # Read the CSV file
  df <- read.csv(file_path)
  variables = c("Value Added|Agriculture","Value Added|Services","Value Added|Industry and Construction","GDP|MER","GDP|PPP","Population")
  regions = c("CHN","IND")
  filtered_df <- df %>% filter(Variables %in% variables,
                                 Region %in% regions)
  
  return(filtered_df)
}

filtered_dfs <- map(file_paths, process_file)

# Combine all filtered dataframes into one dataframe
data2 <- bind_rows(filtered_dfs)

# Reorganise data
data2 <- data2 %>% pivot_longer(
    cols = starts_with('X'),
    names_to = "Year",
    values_to = "Value"
) %>% select(-Unit)

data2 <- data2 %>% pivot_wider(
    names_from = Variables,
    values_from = Value,
)


# Add necessary variables
# data2['Value Added|Service'] = data2['GDP|MER'] - data2['Value Added|Agriculture'] - data2['Value Added|Industry']

# Necessary to consider energy sectors as part of industry
# data2['Value Added|Industry'] = data2['GDP|MER'] - data2['Value Added|Agriculture'] - data2['Value Added|Services']

# data2['Share.AddedValue.Agri'] = data2['Value Added|Agriculture']/data2['GDP|MER']
# data2['Share.AddedValue.Indus'] = data2['Value Added|Industry']/data2['GDP|MER']
# data2['Share.AddedValue.Services'] = data2['Value Added|Services']/data2['GDP|MER']

# # Depreciated calculations:
data2['Share.AddedValue.Agri'] = data2['Value Added|Agriculture']/(data2['Value Added|Agriculture']+data2['Value Added|Industry and Construction']+data2['Value Added|Services'])
data2['Share.AddedValue.Indus'] = data2['Value Added|Industry and Construction']/(data2['Value Added|Agriculture']+data2['Value Added|Industry and Construction']+data2['Value Added|Services'])
data2['Share.AddedValue.Services'] = data2['Value Added|Services']/(data2['Value Added|Agriculture']+data2['Value Added|Industry and Construction']+data2['Value Added|Services'])

data2['ln_gdp_pc'] = log(data2["GDP|MER"]/data2['Population'])




# Calculating national data from DOSE data

# Create new variables
data1$share_ag <- data1$ag_grp_pc_usd_2015 / data1$grp_pc_usd_2015
data1$share_man <- data1$man_grp_pc_usd_2015 / data1$grp_pc_usd_2015
data1$share_serv <- data1$serv_grp_pc_usd_2015 / data1$grp_pc_usd_2015

data1$ln_share_agman <- log(data1$share_ag / data1$share_man)
data1$ln_share_servman <- log(data1$share_serv / data1$share_man)
data1$ln_gdp <- log(data1$grp_pc_usd_2015)

data1$grp_usd_2015 <- data1$grp_pc_usd_2015 * data1$pop
data1$ag_usd_2015 <- data1$ag_grp_pc_usd_2015 * data1$pop
data1$man_usd_2015 <- data1$man_grp_pc_usd_2015 * data1$pop
data1$serv_usd_2015 <- data1$serv_grp_pc_usd_2015 * data1$pop

# Summarize data
df <- data1 %>%
    select(country, region, year, pop, grp_usd_2015, ag_usd_2015, man_usd_2015, serv_usd_2015) %>%
    filter(!is.na(grp_usd_2015) & !is.na(ag_usd_2015) & !is.na(man_usd_2015) & !is.na(serv_usd_2015))

variables_to_summarize <- setdiff(names(df), c("country", "region", "year"))

df <- df %>%
    group_by(country, region, year) %>%
    summarise_all(sum) %>%
    ungroup()

summarized_china_df <- df %>%
    filter(country == "China") %>%
    group_by(year) %>%
    summarise_at(vars(all_of(variables_to_summarize)), sum) %>%
    mutate(region = "China") %>%
    mutate(country = "China") %>%
    select(country, region, year, everything())


summarized_india_df <- df %>%
    filter(country == "India") %>%
    group_by(year) %>%
    summarise_at(vars(all_of(variables_to_summarize)), sum) %>%
    mutate(region = "India") %>%
    mutate(country = "India") %>%
    select(country, region, year, everything())

summarized_df <- bind_rows(summarized_china_df, summarized_india_df)

summarized_df$share_ag <- summarized_df$ag_usd_2015 / summarized_df$grp_usd_2015
summarized_df$share_man <- summarized_df$man_usd_2015 / summarized_df$grp_usd_2015
summarized_df$share_serv <- summarized_df$serv_usd_2015 / summarized_df$grp_usd_2015

summarized_df$grp_pc_usd_2015 <- summarized_df$grp_usd_2015 / summarized_df$pop

summarized_df$ln_gdp <- log(summarized_df$grp_pc_usd_2015)


df <- bind_rows(summarized_df, data1 %>% filter(country == "China" | country == "India") %>% select(names(summarized_df)))

# For China and India, calculate the growth rate of grp_usd_2015
df <- df %>%
    group_by(region) %>%
    mutate(grp_growth = grp_pc_usd_2015 / lag(grp_pc_usd_2015) - 1) %>%
    ungroup()

# ======================================================================================================================================
# ======================================================================================================================================
# Initial models

data_china <- subset(data1, country == "China")


years <- c("2015", "2020", "2025", "2030", "2035", "2040", "2050", "2060", "2070", "2080", "2090")
data2_china <- subset(data2, Country == "CHN" & Scenario == "NDC" & Year %in% years)



sdata <- pdata.frame(data_china, index = c("region", "year"), drop.index = TRUE, row.names = TRUE)

ln_share_agman <- as.matrix(sdata$ln_share_agman)
ln_share_servman <- as.matrix(sdata$ln_share_servman)
ln_gdp <- as.matrix(sdata$ln_gdp)


modelag <- plm(ln_share_agman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = sdata, model = "within", effect = "individual")
modelserv <- plm(ln_share_servman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = sdata, model = "within", effect = "individual")

modelag.chn <- modelag
modelserv.chn <- modelserv

modelag.re <- plm(ln_share_agman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = sdata, model = "random", random.method = "walhus")
modelserv.re <- plm(ln_share_servman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = sdata, model = "random", random.method = "walhus")


phtest(modelag, modelag.re) # p-value = 0.001116 : use fixed effects
phtest(modelserv, modelserv.re) # p-value < 2.2e-16: use fixed effects



cat("Average fixed effects agriculture: ", mean(fixef(modelag)), "\n")
cat("Average fixed effects services: ", mean(fixef(modelserv)), "\n")
#  Estimating the model
x <- seq(5, 11, length.out = 100)

# Estimate modelag for values of x
za <- predict(modelag, newdata = data.frame(ln_gdp = x))
zs <- predict(modelserv, newdata = data.frame(ln_gdp = x))

xm <- 1 / (1 + exp(za) + exp(zs))
xa <- exp(za) * xm
xs <- exp(zs) * xm

# Line graph of xm against x
c_lm <- geom_line(data = data.frame(x = x, xm = xm), aes(x = x, y = xm), color = "black", linetype = "dashed", size = 1)
c_ls <- geom_line(data = data.frame(x = x, xs = xs), aes(x = x, y = xs), color = "black", linetype = "dashed", size = 1)
c_la <- geom_line(data = data.frame(x = x, xa = xa), aes(x = x, y = xa), color = "black", linetype = "dashed", size = 1)


data_india <- subset(data1, country == "India" & year != 2015)
years <- c("2015", "2020", "2025", "2030", "2035", "2040", "2050", "2060", "2070", "2080", "2090")
data2_india <- subset(data2, Country == "IND" & Scenario == "NDC" & Year %in% years)



sdata <- pdata.frame(data_india, index = c("region", "year"), drop.index = TRUE, row.names = TRUE)

ln_share_agman <- as.matrix(sdata$ln_share_agman)
ln_share_servman <- as.matrix(sdata$ln_share_servman)
ln_gdp <- as.matrix(sdata$ln_gdp)


modelag <- plm(ln_share_agman ~ ln_gdp + I(ln_gdp^2), data = sdata, model = "within", effect = "individual")
modelserv <- plm(ln_share_servman ~ ln_gdp + I(ln_gdp^2), data = sdata, model = "within", effect = "individual")

modelag.ind <- modelag
modelserv.ind <- modelserv

modelag.re <- plm(ln_share_agman ~ ln_gdp + I(ln_gdp^2), data = sdata, model = "random", random.method = "walhus")
modelserv.re <- plm(ln_share_agman ~ ln_gdp + I(ln_gdp^2), data = sdata, model = "random", random.method = "walhus")


phtest(modelag, modelag.re) # p-value = 0.02067 - use fixed effects
phtest(modelserv, modelserv.re) # p-value = 0.04015 - use fixed effects


cat("Average fixed effects agriculture: ", mean(fixef(modelag)), "\n")
cat("Average fixed effects services: ", mean(fixef(modelserv)), "\n")
# ===========================
#  Estimating the model
x <- seq(5, 11, length.out = 100)

# Estimate modelag for values of x
za <- predict(modelag, newdata = data.frame(ln_gdp = x))
zs <- predict(modelserv, newdata = data.frame(ln_gdp = x))

xm <- 1 / (1 + exp(za) + exp(zs))
xa <- exp(za) * xm
xs <- exp(zs) * xm

# Line graph of xm against x
i2_lm <- geom_line(data = data.frame(x = x, xm = xm), aes(x = x, y = xm), color = "black", linetype = "dashed", size = 1)
i2_ls <- geom_line(data = data.frame(x = x, xs = xs), aes(x = x, y = xs), color = "black", linetype = "dashed", size = 1)
i2_la <- geom_line(data = data.frame(x = x, xa = xa), aes(x = x, y = xa), color = "black", linetype = "dashed", size = 1)


# Plotting
p_ac <- ggplot(data_china) +
    geom_point(aes(x = ln_gdp, y = share_ag, color = region)) +
    labs(title = "Agriculture", x = "", y = "China \nshare") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1) +
    xlim(5, 11)

p_mc <- ggplot(data_china) +
    geom_point(aes(x = ln_gdp, y = share_man, color = region)) +
    labs(title = "Industry", x = "", y = "") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1) +
    xlim(5, 11)

p_sc <- ggplot(data_india) +
    geom_point(aes(x = ln_gdp, y = share_serv, color = region)) +
    labs(title = "Services", x = "", y = "") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1) +
    xlim(5, 11)

p_ai <- ggplot(data_india) +
    geom_point(aes(x = ln_gdp, y = share_ag, color = region)) +
    labs(title = "", x = "ln(GDP/capita) \n [ln(2015USD/Capita]", y = "India \nshare") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1) +
    xlim(5, 11)

p_mi <- ggplot(data_india) +
    geom_point(aes(x = ln_gdp, y = share_man, color = region)) +
    labs(title = "", x = "ln(GDP/capita) \n [ln(2015USD/Capita]", y = "") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1) +
    xlim(5, 11)

p_si <- ggplot(data_india) +
    geom_point(aes(x = ln_gdp, y = share_serv, color = region)) +
    labs(title = "", x = "ln(GDP/capita) \n [ln(2015USD/Capita]", y = "") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1) +
    xlim(5, 11)



plot <- grid.arrange(p_ac + c_la, p_mc + c_lm, p_sc + c_ls, p_ai + i2_la, p_mi + i2_lm, p_si + i2_ls, nrow = 2)
print(plot)




output_path <- "structural.change/results/Multi_regional_model_results.tex"
# Extract the directory path
output_dir <- dirname(output_path)

# Check if the directory exists
if (!dir.exists(output_dir)) {
  # Create the directory if it doesn't exist
  dir.create(output_dir, recursive = TRUE)
}


stargazer(modelag.chn, modelserv.chn, modelag.ind, modelserv.ind, type = "latex",out =output_path)


# ======================================================================================================================================
# ======================================================================================================================================
# Simplified model

# Import Imaclim data
usd2010to2015 <- 1 / 0.92

df_imaclim <- data2 %>%
    filter(Region %in% c("CHN", "IND")) %>%
    mutate(Country = Region) %>%
    mutate(Region = recode(Region, 'CHN' = 'China', 'IND' = 'India')) %>%
    mutate(share_ag = Share.AddedValue.Agri) %>%
    mutate(share_man = Share.AddedValue.Indus) %>%
    mutate(share_serv = Share.AddedValue.Services) %>%
    mutate(ln_gdp = ln_gdp_pc + log(usd2010to2015)+ log(1000)) %>%
    mutate(Year = as.integer(gsub("X", "", Year))) %>%
    mutate(source = "Imaclim") %>%
    select(country = Country, region = Region, year = Year, scenario = Scenario, share_ag, share_man, share_serv, ln_gdp, source)

df_filtered <- df %>%
    filter(year <= 2015, region %in% c("China", "India")) %>%
    mutate(source = "DOSE") %>%
    mutate(scenario = "Historical") %>%
    select(country, region, year, scenario, share_ag, share_man, share_serv, ln_gdp, source)

df_filtered <- bind_rows(df_filtered, df_imaclim)






scenario_name = "WO-NPi-ElecIndus"

# Running the regressions
df_filtered <- df_filtered %>%
    mutate(ln_share_agman = log(share_ag / share_man)) %>%
    mutate(ln_share_servman = log(share_serv / share_man))

model_ag_c <- lm(ln_share_agman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = filter(df_filtered, region == "China", scenario %in% c("Historical", scenario_name)))
model_serv_c <- lm(ln_share_servman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = filter(df_filtered, region == "China", scenario %in% c("Historical", scenario_name)))

model_ag_i <- lm(ln_share_agman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = filter(df_filtered, region == "India", scenario %in% c("Historical", scenario_name)))
model_serv_i <- lm(ln_share_servman ~ ln_gdp + I(ln_gdp^2) + I(ln_gdp^3), data = filter(df_filtered, region == "India", scenario %in% c("Historical", scenario_name)))


# Extract coefficients from each model
coefficients_ag_c <- coef(model_ag_c)
coefficients_serv_c <- coef(model_serv_c)
coefficients_ag_i <- coef(model_ag_i)
coefficients_serv_i <- coef(model_serv_i)

# Combine coefficients into a data frame
coefficients_df <- data.frame(
  Model = c("Model_AG_China", "Model_Serv_China", "Model_AG_India", "Model_Serv_India"),
  Intercept = c(coefficients_ag_c[1], coefficients_serv_c[1], coefficients_ag_i[1], coefficients_serv_i[1]),
  ln_gdp = c(coefficients_ag_c[2], coefficients_serv_c[2], coefficients_ag_i[2], coefficients_serv_i[2]),
  ln_gdp_squared = c(coefficients_ag_c[3], coefficients_serv_c[3], coefficients_ag_i[3], coefficients_serv_i[3]),
  ln_gdp_cubed = c(coefficients_ag_c[4], coefficients_serv_c[4], coefficients_ag_i[4], coefficients_serv_i[4])
)

# Write the data frame to a CSV file
write.csv(coefficients_df, "structural.change/results/model_coefficients.csv", row.names = FALSE)



stargazer(model_ag_c, model_serv_c, model_ag_i, model_serv_i, type = "text")


output_path <- "structural.change/results/National_extended_model_results.tex"
stargazer(model_ag_c, model_serv_c, model_ag_i, model_serv_i, type = "latex",out =output_path)



	
# Estimating the models

x <- seq(6, 12, length.out = 100)

# Estimate modelag for values of x
za <- predict(model_ag_c, newdata = data.frame(ln_gdp = x))
zs <- predict(model_serv_c, newdata = data.frame(ln_gdp = x))

xm_c <- 1 / (1 + exp(za) + exp(zs))
xa_c <- exp(za) * xm_c
xs_c <- exp(zs) * xm_c

# Estimate modelag for values of x
za <- predict(model_ag_i, newdata = data.frame(ln_gdp = x))
zs <- predict(model_serv_i, newdata = data.frame(ln_gdp = x))

xm_i <- 1 / (1 + exp(za) + exp(zs))
xa_i <- exp(za) * xm_i
xs_i <- exp(zs) * xm_i




# ====================================================================================================================================================================================
# ====================================================================================================================================================================================
# ====================================================================================================================================================================================
# Plotting the results
Cols <- c("#D62728", "#2CA02C", "#1F77B4")
Sc <- c("WO-NPi-ElecIndus", "WO-NDCLTT-ElecIndus", "WO-15C-ElecIndus")


plot_ac <- ggplot(filter(df_filtered, region == "China" & source == "DOSE")) +
    geom_point(aes(x = ln_gdp, y = share_ag), color = "black", label = "Historical Data") +
    geom_line(data = data.frame(x = x, y = xa_c), aes(x = x, y = y), color = "black", size = 1, label = "Fit") +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[1]), mapping = aes(x = ln_gdp, y = share_ag), color = Cols[1], label = "NPI") +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[2]), mapping = aes(x = ln_gdp, y = share_ag), color = Cols[2], label = "NDC") +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[3]), mapping = aes(x = ln_gdp, y = share_ag), color = Cols[3], label = "1.5°C") +
    labs(title = "Agriculture", x = " ", y = "China \nshare") +
    theme_bw() +
    ylim(0, 1)

plot_mc <- ggplot(filter(df_filtered, region == "China" & source == "DOSE")) +
    geom_point(aes(x = ln_gdp, y = share_man), color = "black") +
    geom_line(data = data.frame(x = x, y = xm_c), aes(x = x, y = y), color = "black", size = 1) +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[1]), mapping = aes(x = ln_gdp, y = share_man), color = Cols[1]) +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[2]), mapping = aes(x = ln_gdp, y = share_man), color = Cols[2]) +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[3]), mapping = aes(x = ln_gdp, y = share_man), color = Cols[3]) +
    labs(title = "Industry", x = " ", y = "share") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1)
plot_sc <- ggplot(filter(df_filtered, region == "China" & source == "DOSE")) +
    geom_point(aes(x = ln_gdp, y = share_serv), color = "black") +
    geom_line(data = data.frame(x = x, y = xs_c), aes(x = x, y = y), color = "black", size = 1) +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[1]), mapping = aes(x = ln_gdp, y = share_serv), color = Cols[1]) +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[2]), mapping = aes(x = ln_gdp, y = share_serv), color = Cols[2]) +
    geom_point(filter(df_filtered, region == "China" & source == "Imaclim" & scenario == Sc[3]), mapping = aes(x = ln_gdp, y = share_serv), color = Cols[3]) +
    labs(title = "Services", x = " ", y = "share") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1)
plot_ai <- ggplot(filter(df_filtered, region == "India" & source == "DOSE")) +
    geom_point(aes(x = ln_gdp, y = share_ag), color = "black") +
    geom_line(data = data.frame(x = x, y = xa_i), aes(x = x, y = y), color = "black", size = 1) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[1]), mapping = aes(x = ln_gdp, y = share_ag), color = Cols[1]) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[2]), mapping = aes(x = ln_gdp, y = share_ag), color = Cols[2]) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[3]), mapping = aes(x = ln_gdp, y = share_ag), color = Cols[3]) +
    labs(x = "ln(GDP/capita) \n [ln(2015USD/Capita]", y = "India \nshare") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1)
plot_mi <- ggplot(filter(df_filtered, region == "India" & source == "DOSE")) +
    geom_point(aes(x = ln_gdp, y = share_man), color = "black") +
    geom_line(data = data.frame(x = x, y = xm_i), aes(x = x, y = y), color = "black", size = 1) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[1]), mapping = aes(x = ln_gdp, y = share_man), color = Cols[1]) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[2]), mapping = aes(x = ln_gdp, y = share_man), color = Cols[2]) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[3]), mapping = aes(x = ln_gdp, y = share_man), color = Cols[3]) +
    labs(x = "ln(GDP/capita)\n [ln(2015USD/Capita]", y = "share") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1)
plot_si <- ggplot(filter(df_filtered, region == "India" & source == "DOSE")) +
    geom_point(aes(x = ln_gdp, y = share_serv), color = "black") +
    geom_line(data = data.frame(x = x, y = xs_i), aes(x = x, y = y), color = "black", size = 1) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[1]), mapping = aes(x = ln_gdp, y = share_serv), color = Cols[1]) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[2]), mapping = aes(x = ln_gdp, y = share_serv), color = Cols[2]) +
    geom_point(filter(df_filtered, region == "India" & source == "Imaclim" & scenario == Sc[3]), mapping = aes(x = ln_gdp, y = share_serv), color = Cols[3]) +
    labs(x = "ln(GDP/capita)\n [ln(2015USD/Capita]", y = "share") +
    theme_bw() +
    theme(legend.position = "none") +
    ylim(0, 1)

plot <- grid.arrange(plot_ac + c_la, plot_mc + c_lm, plot_sc + c_ls, plot_ai + i2_la, plot_mi + i2_lm, plot_si + i2_ls, nrow = 2)

print(plot)

ggsave('structural.change/figures/Scatter_structural_change_models.png',
        plot=plot)


# ====================================================================================================================================================================================
# ====================================================================================================================================================================================
# ====================================================================================================================================================================================