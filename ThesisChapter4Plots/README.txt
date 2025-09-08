Use timing{:}.py with cProfile to measure time spent --> make output to profiling_result{:}-{:}.txt
Use extract-data-new.py to extract & combine profiling data --> combined-data.pkl
then also the same file to plot timing data

Use Ultimate.py to generate phase screens --> AllIGPhase
Use anaylse_new2pad.py to compare g, ig, ft slopes --> Slopes --> Ultimate_output.pkl
Use IGNoiseless.py to plot g-ft & ig-ft plots
I think ANALYSE_FTIG_NONOISE_SCINTILLATION.py is better...

Use MAkeSHWFSImage.py to generate shwfs image from existing turbulence --> WFSImages
Use noisy_slope_reduced_data.py to generate noise slopes --> Slopes
Use ch4-noise-new.py to analyse noisy slopes to extract noise values --> NoiseVar.pkl
Use plot-noise.py to make plots

to show if they are near gauss or not … try near_gauss_or_not_final.py


NONE of this is actually circular aperture. they are all square aperture … lol
at least for the noisy one. guaranteed.
now I need to double check the noiseless version....

No!!! there actually is some data with circular aperture.
But some files are actually used wrong functions.
firstly, the my_WFSC have the circular aperture. This is cool and correct. but well the rest of my thesis doesn't have any circular aperture in a lenslet. so this should be for a square aperture instead.
secondly, the MyWFS is a square aperture, but the g-tilt return an average OPD in rad over a pixel difference not the OPD over the whole subaperture. it needs to be scaled by U.shape[0]. Easy fix can be done. I fixed the code now.

Anyway this is too much of a mess. I should just redo the data for this part.

Fuck. The circular aperture or not actually has more implication in my thesis.
I measured the time spent using the circular aperture. I'm too lazy to redo the analysis now. this is an easy fix as well.

But well all of my data in ch4 is circular aperture so this is fine??
