:: Training folders cleaning
del /F /Q Input\Training\EL_EXT\Data\*
del /F /Q Input\Training\BCH\Data\TR\*
del /F /Q Output\Training\EL_EXT\Data\TR\*
del /F /Q Output\Training\EL_EXT\Data\TST\*
del /F /Q Output\Training\EL_EXT\Data\VAL\*
del /F /Q Output\Training\EL_EXT\Visualization\TR\*
del /F /Q Output\Training\EL_EXT\Visualization\TST\*
del /F /Q Output\Training\EL_EXT\Visualization\VAL\*

del /F /Q Input\Training\BCH\Data\TR\*
del /F /Q Input\Training\BCH\Data\VAL\*
del /F /Q Output\Training\BCH\Patterns\*
del /F /Q Output\Training\BCH\thresholds.txt

del /F /Q Input\Training\DR\Data\*
del /F /Q Output\Training\DR\Data\*
del /F /Q Output\Training\DR\Model\*

:: Testing folders cleaning
del /F /Q Input\Testing\AD\thresholds.txt
del /F /Q Input\Testing\AD\Patterns\*
del /F /Q Input\Testing\AD\Data\*
del /F /Q Output\Testing\AD\*









