import subprocess
import glob 

import subprocess
import glob 
import os 
path = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/Downloads/ho3d_eval/"
outf = "Datasets/ho3d/"
folders = glob.glob(path+"*/")

for folder in folders: 
	if "masks_XMem" in folder:
		continue
	name = folder.split("/")[-2]
	print(folder,name)
	if not os.path.exists(folder+"/colmap/"):
		os.mkdir(folder+"/colmap/")
	subprocess.call([
		'python','ho3d_2_nicenerf.py',
		'--pimgs',folder+"/rgb/",
		'--pmasks',path+"/masks_XMem/"+name+"/",
		'--outf',outf+"/"+name
		])
	# raise()