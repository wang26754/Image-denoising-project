# TAU-Thesis-project

################Train###############

run scunet denoising sigma 25.: 
python main.py --net 'scunet' --mode 'normal' --sigma 25.

run DMergenet denoising sigma 25.: 
python main.py --net 'dmergenet' --mode 'normal' --sigma 25.

run LargeDwtnet denoising sigma 25.: 
python main.py --net 'largedwtnet' --mode 'normal' --sigma 25.

run DWT scunet denoising sigma 25.: 
python main.py --net 'scunet' --mode 'dwtLL' --sigma 25.
python main.py --net 'scunet' --mode 'dwtLH' --sigma 25.
python main.py --net 'scunet' --mode 'dwtHL' --sigma 25.
python main.py --net 'scunet' --mode 'dwtHH' --sigma 25.

run SwinIR denoising sigma 25.: 
python main.py --net 'swin' --mode 'normal' --sigma 25.

###############Test###################

test scunet denosing sigma 25
python test.py --net 'scunet' --mode 'normal' --sigma 25

test DMergenet denosing sigma 25
python test.py --net 'dmergenet' --mode 'normal' --sigma 25

test LargeDwtnet denosing sigma 25
python test.py --net 'largedwtnet' --mode 'normal' --sigma 25

test DWT scunet denosing sigma 25.
python test_dwt.py --net 'scunet' --mode 'dwt' --sigma 25.

python test2.py --net 'scunet' --mode 'dwt' --sigma 25
