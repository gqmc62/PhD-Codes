0. Code that I am using for the sh-wfs,ig-g are listed in
	-1Directory/Codes/my_WFSC.py -- circular aperture
	-1Directory/Codes/MyWFS.py -- square aperture

	This thesis only uses the square aperture for the final report.
	
1. I use 20250626/final-timingall.py to measure time.
	the code that are meausured is in -1Directory/Codes/my_WFSC.py, -1Directory/Codes/MyWFS.py, and -1Directory/Codes/mypropagationclass.py

	the result is in 20250626/time_use_time_all.pkl
	then is plotted using 20250626/final-plot-time.py
	plots are in 20250626

	data in the table is based on the result and calculated using 20250626/estimate-sim-time.xlsx

2. I generate scintillation phase screen from Ultimate.py.
	Though the original file is lost, the derivative version of it exists in Ultimate_Extra.py, which is adjusted to generate only a selected few conditions.
	This code generate propagated turbulence without using any cheating/shortcut/reducing that was introduced in my Chapter 3, thus a limiting condition.

	This will generate Ultimate_input.pkl storing simulating parameters and phase screens in Ultimate_input.pkl or 20250626/IGPhaseScreen/Ultimate_input.pkl

	containing preprogation ground and high altitude phases (20250626/IGPhaseScreen/ps_GL/* and 20250626/IGPhaseScreen/ps_HA/*) and
	the combine propagated complex amplitude (20250626/IGPhaseScreen/U_tot/*)

	There is one turbulence condition missing that could have been simulated. --> 5cm r0, 3d/r0, 0.1 rytov, 15km hha. I think only a limiting number of the screen is not lost, but the analysis I had scrapped the whole set out.
	
	20250626MakeSHWFSImage.py is used to generate WFS image.
	The wfs image is in 20250626/IGPhaseScreen/Image/*

3. Study noiseless condition.
	20250626/final-NoiselessIG.py grab 20250626/IGPhaseScreen/U_tot/* measure slope for different cases
	og --> meaning using 64x64 of 64x64 for cog + all threshold
	no-ft-bias --> actually is 17x17 of 64x64 for cog + all threshold. not really no bias!

	slopes are in 20250626/NoiselessSquarePupilSlopes-no-ft-bias.pkl and 20250626/NoiselessSquarePupilSlopes-og.pkl

	then analyse and plot using 20250626/final-noiseless-plot.py
	the plots are in 20250626

4. Study noise on scintillation cases.
	20250626/final-SHWFS_NOISYSLOPE_GENERATOR-noscintillation.py and 20250626/final-SHWFS_NOISYSLOPE_GENERATOR-withscintillation.py
	grab wfs images in 20250626/IGPhaseScreen/Image/* add noises then measure slope 
	the noisy slopes are in 20250626/Slopes/*

	final-shwfs-noise-plot.py analyse and generate plot
	results are in 20250626/noisyslope for some reason.
