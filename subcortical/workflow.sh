loc=$1
data_info=$2
disorder=$3
control=$4
python task_gen.py $loc $data_info $disorder $control
python make_dataset.py $loc
python -W ignore YZ_LRwithL1L2TV_gs.py
