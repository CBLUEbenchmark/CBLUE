# format checker for CBLUE tasks:
* CMeEE + CMeEE-V2: format_checker_CMeEE.py
* CMeIE + CMeIE-V2: format_checker_CMeIE.py
* CHIP-CDEE: format_checker_CDEE.py 
* CHIP-CDN: format_checker_CDN.py
* CHIP-CTC: format_checker_CTC.py
* CHIP-STS: format_checker_STS.py
* CHIP-MDCFNPC: format_checker_MDCFNPC.py
* KUAKE-QIC: format_checker_QIC.py
* KUAKE-QQR: format_checker_QQR.py
* KUAKE-QTR: format_checker_QTR.py
* KUAKE-IR: format_checker_IR.py 
* MedDG: format_checker_MedDG.py
* Text2DT: format_checker_Text2DT.py
* IMCS-NER: format_checker_IMCS_V1_NER.py
* IMCS-IR: format_checker_IMCS_V1_IR.py
* IMCS-SR: format_checker_IMCS_V1_SR.py
* IMCS-MRG: format_checker_IMCS_V1_MRG.py 
* IMCS-V2-NER: format_checker_IMCS_V2_NER.py
* IMCS-V2-DAC: format_checker_IMCS_V2_DAC.py
* IMCS-V2-SR: format_checker_IMCS_V2_SR.py
* IMCS-V2-MRG: format_checker_IMCS_V2_MRG.py 


# How to run:
* Step1: Copy the original test file(without answer) {taskname}_test.[json|jsonl|tsv] to this directory, and rename as {taskname}_test_raw.[json|jsonl|tsv].
```
# take the CMeEE task for example:
cp ${path_to_CMeEE}/CMeEE_test.json ${current_dir}/CMeEE_test_raw.json 
```
* Step2: Execute the following format_checker script using the raw test file (from Step1) and your prediction file:
```
python3 format_checker_${taskname}.py {taskname}_test_raw.[json|jsonl|tsv] {taskname}_test.[json|jsonl|tsv] 

# take the CMeEE task for example:
python3 format_checker_CMeEE.py CMeEE_test_raw.json CMeEE_test.json
```

# What is special? 
## IMCS-NER & IMCS-V2-NER tasks:
* Step1: Copy both the original test file(without answer) IMCS-NER_test.json(IMCS-V2-NER_test.json) and the IMCS_test.json(IMCS-V2_test.json) to this directory, and rename as IMCS-NER_test_raw.json(IMCS-V2-NER_test_raw.json)
```
# for IMCS-NER task:
cp ${path_to_IMCS-NER}/IMCS-NER_test.json ${current_dir}/IMCS-NER_test_raw.json 
cp ${path_to_IMCS-NER}/IMCS_test.json ${current_dir}
# for IMCS-V2-NER task:
cp ${path_to_IMCS-V2-NER}/IMCS-V2-NER_test.json ${current_dir}/IMCS-V2-NER_test_raw.json 
cp ${path_to_IMCS-V2-NER}/IMCS-V2_test.json ${current_dir}
```
* Step2: Execute the following format_checker script using the raw test file (from Step1) and your prediction file:
```
# for IMCS-NER task:
python3 format_checker_IMCS_V1_NER.py  IMCS-NER_test_raw.json IMCS-NER_test.json IMCS_test.json
# for IMCS-V2-NER task:
python3 format_checker_IMCS_V2_NER.py  IMCS-V2-NER_test_raw.json IMCS-V2-NER_test.json IMCS-V2_test.json
```

## IMCS-SR & IMCS-V2-SR, MedDG tasks
If you want to implement the optional check login in the *check_format* function, which is commented in the master branch. You need also copy the normalized dictionary files to the current dir.
* MedDG: the dictionary file is *entity_list.txt*
* IMCS-SR: the dictionary file is *symptom_norm.csv*
* IMCS-V2-SR:  the dictionary file is *mappings.json*

