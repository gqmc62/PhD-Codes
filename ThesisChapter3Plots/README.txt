1) phase screen - completed
	1.1) which method to use? -- final_phase_screen_tests.py
		1.1.1) ft
			1.1.1.1) how many oversize?
			1.1.2.1) subharmonic or not?
		1.1.3) infinite phase screen
			1.1.3.1) soapy-von-karman
				1.1.3.1.1) how many oversize?
				1.1.3.1.2) how many flush?
			1.1.3.2) soapy-Kolmogorov
				1.1.3.2.1) how many oversize?
				1.1.3.2.2) how many flush?

2) propagation : how shortcut affect statistics? - completed
	2.1) PSD : U,I,phi --> testPropagationMirrorPadv2.py
	2.2) D : U,I,phi --> testPropagationMirrorPadv2.py
	2.3) blobsize? --> testPropagationMirrorPad.py ==> this is already testing only on the cheating version
		--> plz don't use the width of cov(phase). it's weird coding. what a waste of time in the past
		--> use the where structure function saturates instead.
	2.4) scintillation? --> testPropagationMirrorPad.py ==> this is already testing only on the cheating version
	2.5) difference between cheat-no-cheat? --> intensity-weighted OPD RMS dif --> testPropagationMirrorPadv2.py

3) wfs
	1.3.1) how many zero pad -- 1-step-fresnel-sh-wfs-with-padding.py
	1.3.2) d from turbulence : fov, delta_out given by Sandrine and turb -- no need to do analysis
	1.3.3) number of input phase pixel for a given d --  what I mostly do is to have equal number of input and output, which is valid for the 1-step Fresnel case. it optimize speed. but for other cases? 

4) control : completed
	because I only simulate a perfect converging test. I don't need to do time series.
	use code and data in chapter5 in the deepest data folder. lol
	& use plots from 20230522-PhDmeeting.pptx, 20230605-PhDmeeting.pptx
	I have -
	4.0) solving 1 turb - how solution converge - control_loop_undistort.pdf
	4.1) rcond test --> between which value give largest strehl ratio … should have done wfe rms … lol noooo svd_optimize.png
	4.2) dm fitting constant -- > measurement variance=k(d/r0)^(5/3) - dmfitting.png
	4.21) residual spatial psd phase --> theory_psd-phase.pdf
	4.22) residual Zernike dcomposition --> asdf.pdf
	4.3) temporal error --> fast wind than the … will have no correction - temporal_error.png
	4.4) temporal psd --> temporal_psd.png



