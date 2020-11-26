#### Gird Search with YZ_ML model with the data_location data_info, csv file, disorder_label and control_label  ####
## In the YZ_LR script, one can modify the grid-search range as needed: line#54, variable "gs_list"
bash -e workflow.sh [data_loc] [data_info] [disorder_label] [control_label]

#For example#
bash -e workflow.sh /home/yzhao104/Desktop/BG_project/PD_data/Affine_normalized_shape_analysis_combined_mesh ppmi_subject_info_CORRECT_baseline.csv PD Control

#### Make weights maps with the grid search results #####
bash -e sym_workflow_visualization.sh [subcortical dataset folder path] [grid_point params]
#For example#
bash -e sym_workflow_visualization.sh /home/yzhao104/Desktop/BG_project/PD_data/Affine_normalized_shape_analysis_combined_mesh/3001 0.001_1000_0.001
