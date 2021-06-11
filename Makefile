PIP=pip3
CUDA=cu101

setup_cpu: setup_python_common setup_python_cpu setup_cass_extractor
setup_gpu: setup_python_common setup_python_gpu_nvida setup_cass_extractor

setup_python_common:
	$(PIP) install wheel
	$(PIP) install absl-py==0.9.0
	$(PIP) install numpy==1.18.1
	$(PIP) install pyprg==0.1.1b7
	$(PIP) install regex==2020.4.4
	$(PIP) install scipy==1.4.1
	$(PIP) install sklearn==0.0
	$(PIP) install tqdm==4.42.1
	$(PIP) install tree-sitter==0.1.1
	$(PIP) install wget==3.2
	$(PIP) install networkx==2.4

setup_python_cpu:
	$(PIP) install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
	$(PIP) install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html

setup_python_gpu_nvida:
	$(PIP) install torch==1.6.0+$(CUDA)
	$(PIP) install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+$(CUDA).html

setup_cass_extractor:
	if [ -d "./cass-extractor/build" ]; then echo "Dir exists"; fi
	mkdir -p ./cass-extractor/build
	cmake -DCMAKE_BUILD_TYPE=Release -S ./cass-extractor -B ./cass-extractor/build
	$(MAKE) -C ./cass-extractor/build -j
