This folder contains the code necessary to run and analyse the structural change sensitivity analysis of the coal.labour.nexus.

First, an econometric regression based on Leimbach et al's model [^1] is ran on the DOSE [^2] database with the R code: `Structural_change_model.r`.

The results of this regression are then used to recalibrate downscaling coefficients with a python code: `Build_structural_change_structure.py`. This creates new input data for the `coal.labour.nexus`. 

The module can then be run normally with these new inputs.  

New sensitivity analysis module results are then analysed with `SC_sensitivity_analysis.py`.

[^1]: Leimbach, Marian, Marcos Marcolino, and Johannes Koch. ‘Structural Change Scenarios within the SSP Framework’. *Futures* **150** (1 June 2023): 103156. https://doi.org/10.1016/j.futures.2023.103156.
[^2]: Wenz, Leonie, Robert Devon Carr, Noah Kögel, Maximilian Kotz, and Matthias Kalkuhl. ‘DOSE – Global Data Set of Reported Sub-National Economic Output’. *Scientific Data* **10**, no. 1 (3 July 2023): 425. https://doi.org/10.1038/s41597-023-02323-8.
