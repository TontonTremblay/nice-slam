import subprocess
import glob 

import subprocess
import glob 
import os 
path = "Datasets/ho3d/"
folders = glob.glob(path+"*/")

for folder in folders:
	if os.path.exists(f"output/{folder.split('/')[-2]}"):
		continue
	subprocess.call(
		[
		# "OPENCV_IO_ENABLE_OPENEXR=1", 
		"python", "run.py", "configs/Demo/demo_bundlenerf.yaml",
		"--input_folder",f'{folder}',
		"--output",f"output/{folder.split('/')[-2]}"
		]
	)
	raise()