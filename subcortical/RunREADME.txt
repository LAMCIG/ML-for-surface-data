#### Gird Search with YZ_ML model with the data_location data_info, csv file, disorder_label and control_label  ####

bash -e workflow.sh [data_loc] [data_info] [disorder_label] [control_label]

#For example#
bash -e workflow.sh /home/yzhao104/Desktop/BG_project/PD_data/Affine_normalized_shape_analysis_combined_mesh ppmi_subject_info_CORRECT_baseline.csv PD Control

#### Make weights maps with the grid search result such as '0.001_1000_0.001' below #####

bash -e sym_workflow_visualization.sh 0.001_1000_0.001
