1) should do square pupil
2) should do cog on moving 16x16 with 1e-2 normalised threshold for no bias? in ft slope (g slope = ft slope)
3) well no threshold gives less rms error between ft & ig in scintillation but no noise

library
my_WFSC ==> circle aperture
MyWFS ==> square aperture
mypropagationclass ==> use blind 1-step prop

code

section 1
size ==> time
time spent study ==> final-timingall.py
plot all section 1 plots ==> final-plot-time
see also estimate-sim-time.xlsx

section limbo A
param ==> U
generate U ==> Ultimate.py (Ultimate_Extra.py)

section 2 
U ==> slope
noiseless ig vs noiseless ft ==> final-NoiselessIG.py in radians
og ==> use all 64x64 for cog no threshold
no-ft-bias ==? use moving 16x16 for cog with 1e-2 threshold
circle/square ==> pupil shape
scintillation/noscintillation ==> include in the same file. different variable name.
suggest using 'NoiselessSquarePupilSlopes-og.pkl' and 'NoiselessSquarePupilSlopes-no-ft-bias.pkl'
slope ==> error
all section plot ==> final-noiseless-plot.py ==> already convert all radians into nm when load data

section limbo B
U ==> image
final-MakeSHWFSImage.py ==> no noise no threshold image of ft

section 3
ft image ==> noise slope
==> final-SHWFS-NOISYSLOPE_GENERATOR-withscintillation.py & similar for noscintillation
==> sq-ap-no-ft-bias-SHWFSNOISE... ==> use the moving ROI with n_cog=17, threshold=1e-4 as standard



