# Coal productivity and labour nexus

## Overview
> The coal productivity and labour nexus are two standalone module to analyse the labour implications of IAM pathways.
> The modules downscales pathways across regions of China and India. The `coal.productivity.nexus` determines subnational coal productivity, production and employment pathways. The `coal.labour.nexus` determines the ability of coal mining workers to find new employment based on macroeconomic conditions.
> Both modules were originally developed as an ex-post module for Imaclim-R World V2.0. They can be used with any transition scenario. Filler scenarios can be provided if pathways miss crucial variables.


## Structure
- `coal.labour.nexus`: contains the coal labour nexus module and data treatment code
- `coal.productivity.nexus`: contains the coal productivity nexus module and data treatment code
- `notebooks`: contains code for the analysis of and plotting of results for the paper: "*Coal mine workers in the climate transition: subnational vulnerability disparities in China and India*" 
- `structural.change`: contains code for the analysis of alternative scenarios where downscaling follows different structural change rules.


## Disclaimer
The modules are still in development.


# coal.productivity.nexus
The productivity module is based on the analysis of the Global Energy Monitor's Global Coal Mine Tracker[^1]. Our analysis is based on the April 2024 release but could be adapted to future release.
First, `CoalStudy_MineProductivity.R` code is run to pretreat data. 
Then `nexus.coal.productivity.sce` is run to downscale nationwide IAM coal production trajectories to the subnational scale in China and India. Downscaling accounts for subnational productivity and reserve dynamics based on mine-level data accounting for mines age, workforce and geology as well as forthcoming mining projects. 

The outputs of `nexus.coal.productivity.sce` can then be used as inputs to the `coal.labour.nexus` alongside the same IAM pathways.


# coal.labour.nexus
## Using the module
### Data and preprocessing
Input data for the coal labour nexus is not made available here as it contains data which cannot be shared freely. Two sorts of input data is required to run the module:
1) National demographic scenario data
2) Subnational downscaling data

The data is processed by running `coal.labour.nexus\data\process_data.sh` which calls `Building_age_pyramid.py` and `Process_waysout_data.py` which treat each type of data respectively.

For default scenarios, demographic data are based on the 2015 UN age pyramid data[^2] and demographic projections from SSP scenario data[^3]. The user is free to use demographic projections consistent with the analysed scenario.

Other data are stored in a `Database.csv` file with the following format:

| Country | Region | Type | Year | Value | Unit | Source | URL | Note |
|---------|--------|------|------|-------|------|--------|-----|------|


### Input scenarios
Input scenarios must first be placed in the `coal.labour.nexus\input\` folder. The module is designed to read scenario results presented following the IAMC time-series data template[^3]. The code is designed to read each scenario from an independent csv file.

If some required variables is missing from the scenario data, it can be infilled with Imaclim-R World V2.0 results for typical scenario in `coal.labour.nexus\Infilling_trajectories` 

### Defining module scenarios
The scenarios are defined and the model is ran from `coal.labour.nexus\model`.

The module is designed to perform different sensitivity analysis base on original *ex ante* scenarios (input scenarios). For a given input scenario, there can be several *output* scenarios. Those scenarios are defined in `V2_batch.labour.sce`. 
For each scenario, the user must define a source (i.e. the *input scenario*). They can then define a number of additional parameters:
- Retirement age
- Coal labour productivity growth
- Source of downscaling data
- Structural change parameter (no structural change by default)
- How to consider competition for available job vacancies.
- Whether to create infilling trajectories

### Running the module
The module can be run through `coal.labour.nexus.sce\model\run.sh` or `V2_batch.labour.sce` directly. The code loops through each scenario and calls `V2_nexus.labour.core.sce`.

For each step, the core of the module fetches demographic and geographical downscaling keys. It then calculates non-coal trajectories. 

For Imaclim-R World V2.0, these are found in `Imaclim.Labour.sce` by attributing labour from each of the 11 non-coal sectors to the subnational regions before reaggregating. 

Coal labour is then found from coal production and productivity increase in `V2_coal.productivity.sce`


### Outputs
For each run results are saved alongside log files in a folder in `coal.labour.nexus\output`. A csv file is created for each scenario in a format analogous to the IAMC's[^4]. 



## References
[^1]: Global Energy Monitor. ‘Global Coal Mine Tracker (April 2024 Release)’. Version April 2024 release. 2024. https://globalenergymonitor.org/projects/global-coal-mine-tracker/download-data/.
[^2]: United Nations, 2022. World Population Prospects 2022. 
[^3]: Kc, S., Lutz, W., 2017. The human core of the shared socioeconomic pathways: Population scenarios by age, sex and level of education for all countries to 2100. *Glob. Environ. Change* 42, 181–192. https://doi.org/10.1016/j.gloenvcha.2014.06.004 
[^4]: https://www.iamconsortium.org/scientific-working-groups/data-protocols-and-management/iamc-time-series-data-template/
