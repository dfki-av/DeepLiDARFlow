#!/bin/bash 
current_dir=$(pwd)
echo $current_dir
script_dir="$current_dir"
wget https://cloud.dfki.de/owncloud/index.php/s/6sKDAALAPXim6gR/download -O disp_occ_1_transformedOnSecFrame.zip
unzip disp_occ_1_transformedOnSecFrame.zip -d ./data_scene_flow/
rm disp_occ_1_transformedOnSecFrame.zip

	
     

