Use timing{:}.py with cProfile to measure time spent --> make output to profiling_result{:}-{:}.txt
Use extract-data-new.py to extract & combine profiling data --> combined-data.pkl
then also the same file to plot timing data

Use Ultimate.py to generate phase screens --> AllIGPhase
Use anaylse_new2pad.py to compare g, ig, ft slopes --> Slopes --> Ultimate_output.pkl
Use IGNoiseless.py to plot g-ft & ig-ft plots

Use MAkeSHWFSImage.py to generate shwfs image from existing turbulence --> WFSImages
Use noisy_slope_reduced_data.py to generate noise slopes --> Slopes
Use ch4-noise-new.py to analyse noisy slopes to extract noise values --> NoiseVar.pkl
Use plot-noise.py to make plots