#!/bin/bash 
current_dir=$(pwd)
echo $current_dir
script_dir="$current_dir"
wget https://cloud.dfki.de/owncloud/index.php/s/cnzEtZHbRYxiiJD/download -O deeplidarflow-kitti.zip
unzip deeplidarflow-kitti.zip -d ./model/
rm deeplidarflow-kitti.zip
wget https://cloud.dfki.de/owncloud/index.php/s/xnefEjH5KC5J9ex/download -O deeplidarflow-ft3d.zip
unzip deeplidarflow-ft3d.zip -d ./model/
rm deeplidarflow-ft3d.zip
	
     

