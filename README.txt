specfit is a routine contained in model_fit_tools_vx.py, where the version numbers currently correspond to:
v1) standard emcee implementation as well as a MH MCMC function -- This has been updated LEAST recently, and currently isn't public.
v2) parallel tempering emcee implementation, as well as updated versions of the functions contained in v1
v3) uses a simulated annealing algorithm (which is essentially a modified metropolis-hastings algorithm) for the MCMC, and is dependent on v2 being in the same directory, as it imports its basic functions from v2. 

Any/all of these should be fairly useable. The file paths are currently hardcoded, so they would need to be modified. There are lots of additional functionalities to come, once I've picked an MCMC method to use. All of these depend on phoenix models being in the same directory in a subdirectory called 'phoenix/phoenixm00', as it currently only supports metallicity == solar metallicity.
