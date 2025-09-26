0. the soapy version I used is in -2Directory/soapy/soapy

1. use -1Directory/IM_SCAO_bad_calibration.py to run soapy
	soapy's result is in LastChapter5Data\conf\FinalSupreme\conf\Data\**FOLDERS**
	
	A,B,C --> fixed dpupil --> 0.4m with 10,8,6 subapxpupil
	A2,B2,C2 --> fixed dsub --> … not used in the thesis analysis

2. analysis use distortion_analysis_std.py and rcond_analysis_std.py
	result in LastChapter5Data\conf\FinalSupreme\conf\std\*

3. MCAO-Control-Error/MCAO-class.py is used to calculate for the control error in each mcao system
	configs of each mcao system is in MCAO-Control-Error/MCAO-Config/*
	turbulence used is listed in MCAO-Control-Error/TurbProf/*

	results is in MCAO-Control-Error/DM-Control-Error-Param.txt and MCAO-Control-Error/DM-Control-Error-Param-Median.txt