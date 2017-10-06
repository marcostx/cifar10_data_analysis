# rcnn-fer Docker Makefile
PROGRAM="RCNN-FER"

CPU_REGISTRY_URL=so77id
GPU_REGISTRY_URL=so77id
CPU_VERSION=latest
GPU_VERSION=latest
CPU_DOCKER_IMAGE=tensorflow-opencv-cpu-py3
GPU_DOCKER_IMAGE=tensorflow-opencv-gpu-py3


##############################################################################
############################# Exposed vars ####################################
##############################################################################
# enable/disable GPU usage
GPU=false
# List of cuda devises
CUDA_VISIBLE_DEVICES=0
# Name of dataset to process
PROCESS_DATASET=""

#Path to src folder
HOST_CPU_SOURCE_PATH = ""
HOST_GPU_SOURCE_PATH = ""
# Path to dataset
HOST_CPU_DATASETS_PATH = ""
HOST_GPU_DATASETS_PATH = ""
# Path to metada
HOST_CPU_METADATA_PATH = ""
HOST_GPU_METADATA_PATH = ""

##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker
NVIDIA_DOCKER_COMMAND=nvidia-docker


#HOST VARS
LOCALHOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26006

#HOST CPU VARS
HOST_CPU_SOURCE_PATH=$(shell pwd)
HOST_CPU_METADATA_PATH=$(shell pwd)/metadata
HOST_CPU_DATASET_PATH=$(HOME)/.keras

#HOST GPU PATHS
HOST_GPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_METADATA_PATH=$(shell pwd)/metadata
HOST_GPU_DATASET_PATH=$(HOME)/.keras

#IMAGE VARS
IMAGE_SOURCE_PATH=/home/src
IMAGE_METADATA_PATH=/home/metadata
IMAGE_DATASET_PATH=$(~$USER)/.keras


# VOLUMES

CPU_DOCKER_VOLUMES = --volume=$(HOST_CPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_CPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --volume=$(HOST_CPU_DATASET_PATH):$(IMAGE_DATASET_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)

GPU_DOCKER_VOLUMES = --volume=$(HOST_GPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_GPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --volume=$(HOST_GPU_DATASET_PATH):$(IMAGE_DATASET_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)


DOCKER_PORTS = -p $(LOCALHOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)

# IF GPU == false --> GPU is disabled
# IF GPU == true --> GPU is enabled
ifeq ($(GPU), true)
	DOCKER_RUN_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm  $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
	DOCKER_RUN_PORT_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm  $(DOCKER_PORTS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
else
	DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm  $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_PORT_COMMAND=$(DOCKER_COMMAND) run -it --rm  $(DOCKER_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
endif


##############################################################################
############################## CODE VARS #####################################
##############################################################################
#COMMANDS
PYTHON_COMMAND=python3
EXPORT_COMMAND=export

#FILES
TRAIN_FILE=keras_nn.py
TEST_FILE=keras_nn_test.py


##############################################################################
############################ CODE COMMANDS ###################################
##############################################################################
all: train

train t:
	@echo "[Train] Trainning model"
	@echo "\t Using CUDA_VISIBLE_DEVICES: "$(CUDA_VISIBLE_DEVICES)
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
	@$(PYTHON_COMMAND) $(TRAIN_FILE)

test te:
	@echo "[Train] Test model"
	@echo "\t Using CUDA_VISIBLE_DEVICES: "$(CUDA_VISIBLE_DEVICES)
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
	@$(PYTHON_COMMAND) $(TEST_FILE)


##############################################################################
########################### DOCKER COMMANDS ##################################
##############################################################################

run rc: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make train CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$

run-test rt: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make test CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$

run-docker rtm: docker-print
	@$(DOCKER_RUN_COMMAND)



#PRIVATE
docker-print psd:
ifeq ($(GPU), true)
	@echo "[GPU Docker] Running gpu docker image..."
else
	@echo "[CPU Docker] Running cpu docker image..."
endif