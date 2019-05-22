This code prepares the atmospheric forcing fields for a LiveOcean run using, if available, 3 resolutions of WRF output.  It has been recoded entirely in python from forcing/atm.  It makes use of very fast nearest neighbor interpolation to speed things up by a factor of about 12x (20 sec vs. 4 min for a cas6 backfill day).  And it produces much higher resolution forcing in the Salish Sea.

NOTE: We set 'rain' to zero because (a) we don't really understand the units and (b) is it not used in the simulations at this point 2019.05.22.

============================================================================
* make_forcing_main.py is the main processing code, structured like all the other forcing.

Input: a ROMS grid file and the WRF output files for a "forecast day" from Cliff Mass' group at UW.

Output: the usual Data (empty) and Info directories, and NetCDF files suitable for forcing a ROMS run with bulk fluxes.  One file per variable:

Pair.nc, Tair.nc, Vwind.nc, rain.nc, Qair.nc, Uwind.nc, lwrad_down.nc, and swrad.nc

Note: if you set testing = True it will make a plot at the end showing, for all variables, how the addition of the higher resolution grids improved the output for a given day.

============================================================================
* atm_fun.py module of functions used by make_forcing_main.py.  Could move more of the main code here now that we know it all works.

============================================================================
* plot_grids.py makes a very nice plot of the domain and grid spacing of the ROMS grid and all three of the WRF grids used.

============================================================================
* compare_atm1_atm.py makes plots comparing all fields for a given day as produced by the old atm code and this new code, to make sure everything looks right.