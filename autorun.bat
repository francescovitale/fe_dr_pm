:: Input arguments
:: %1: conformance_checking_technique, %2: n_clusters, %3: window_length

:: Options
:: debug_mode=[0,1]
:: validation_split_percentage=<float>
:: normal_test_split_percentage=<float>
:: ae_validation_split_percentage=<float>
:: normalization_technique=[zscore,min-max]
:: clustering_technique=[agglomerative,kmeans]
:: n_clusters=<integer>
:: window_length=<integer>
:: thresholds_computation_technique=[average,max]
:: conformance_checking_technique=[token_based,alignment_based]
:: process_discovery_technique=[inductive_miner]
:: noise_threshold=<float>

set debug_mode=1
set validation_split_percentage=0.65
set normal_test_split_percentage=0.35
set ae_validation_split_percentage=0.15
set ae_test_split_percentage=0.3
set normalization_technique=min-max
set clustering_technique=kmeans
set n_clusters=%2
set window_length=%3
set thresholds_computation_technique=average
set conformance_checking_technique=%1
set process_discovery_technique=inductive_miner
set noise_threshold=0.2

:: Training folders cleaning
del /F /Q Input\Training\EL_EXT\Data\*
del /F /Q Input\Training\BCH\Data\TR\*
del /F /Q Input\Training\BCH\Data\VAL\*
del /F /Q Input\Training\DR\Data\*
del /F /Q Output\Training\DR\Data\*
del /F /Q Output\Training\DR\Model\*
del /F /Q Output\Training\EL_EXT\Data\TR\*
del /F /Q Output\Training\EL_EXT\Data\TST\*
del /F /Q Output\Training\EL_EXT\Data\VAL\*
del /F /Q Output\Training\EL_EXT\Visualization\TR\*
del /F /Q Output\Training\EL_EXT\Visualization\TST\*
del /F /Q Output\Training\EL_EXT\Visualization\VAL\*
del /F /Q Output\Training\BCH\Patterns\*
del /F /Q Output\Training\BCH\thresholds.txt

:: Testing folders cleaning
del /F /Q Input\Testing\AD\thresholds.txt
del /F /Q Input\Testing\AD\Patterns\*
del /F /Q Input\Testing\AD\Data\*
del /F /Q Output\Testing\AD\*

copy Data\* Input\Training\DR\Data

python ./dr.py %debug_mode% %ae_test_split_percentage% %ae_validation_split_percentage%

copy Output\Training\DR\Data\* Input\Training\EL_EXT\Data

python ./el_ext.py %debug_mode% %validation_split_percentage% %normal_test_split_percentage% %normalization_technique% %clustering_technique% %n_clusters% %window_length%

copy Output\Training\EL_EXT\Data\TR\* Input\Training\BCH\Data\TR
copy Output\Training\EL_EXT\Data\VAL\* Input\Training\BCH\Data\VAL
copy Output\Training\EL_EXT\Data\TST\* Input\Testing\AD\Data

python ./behavior_characterization.py %thresholds_computation_technique% %conformance_checking_technique% %process_discovery_technique% %noise_threshold%

copy Output\Training\BCH\Patterns\* Input\Testing\AD\Patterns
copy Output\Training\BCH\thresholds.txt Input\Testing\AD

python ./anomaly_detector.py %debug_mode% %conformance_checking_technique%










