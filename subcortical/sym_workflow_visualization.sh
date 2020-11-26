gs_point=$1
python2 regionize_gs_coef.py /home/yzhao104/Desktop/BG_project/AD_data/YZ_dataset/002_S_0295 $gs_point
python2 coef_analysis.py $gs_point > coef_analysis.txt
lb=$(grep lb coef_analysis.txt | cut -d: -f2)
ub=$(grep ub coef_analysis.txt | cut -d: -f2)
#echo $lb
#echo $ub
for region in 10 11 12 13 17 18 26 49 50 51 52 53 54 58
#for region in 11
do
    atlas='atlas_'$region'.m'
    coef_raw=$region'_gs_coef.raw'
    coef_mesh='colored_'$region'_gs_coef.m'
    coef_obj=' colored_'$region'_gs_coef.obj'
    
    #./ccbbm -color_attribute $atlas $coef_raw $coef_mesh
    ./ccbbm -color_attribute $atlas $coef_raw $coef_mesh $lb $ub
    #./ccbbm -color_attribute $atlas $coef_raw $coef_mesh -0.00197201222181 0.0018953132676
    ./ccbbm -mesh2obj $coef_mesh $coef_obj 
    rm $coef_mesh
done
f_n='sym_'$gs_point
mkdir $f_n
mv ./*ef.raw* ./$f_n
mv ./*ef.obj* ./$f_n
mv ./coef_analysis.txt ./$f_n
