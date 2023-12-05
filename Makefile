.PHONY: test

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
MACHINE ?= default
CONFIG ?= 'config/configPatches.yaml'
CHECKPOINT ?= 'None'
RUN_IN_CONTAINER = docker --context $(MACHINE) compose run -e PLS_CHECKPOINT=$(CHECKPOINT) -e PLS_CONFIG=$(CONFIG) -e CUDA_VISIBLE_DEVICES=$(GPUS) pcd_leaf_segmenter

build:
ifdef MACHINE
	COMPOSE_DOCKER_CLI_BUILD=1 docker --context $(MACHINE) compose build pcd_leaf_segmenter --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)
else
	COMPOSE_DOCKER_CLI_BUILD=1 docker compose build pcd_leaf_segmenter --ssh ssh-rsa --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)
endif
train_instances:
	$(RUN_IN_CONTAINER) "python train_inst.py"

generate_confid_dataset:
	$(RUN_IN_CONTAINER) -e PLS_GENERATE_CONFID_DATASET=1 "python train_inst.py"

train_confid:
	$(RUN_IN_CONTAINER) "python train_confid.py" 

shell:
	$(RUN_IN_CONTAINER) bash
	
test_instances:
	$(RUN_IN_CONTAINER) "python test_inst.py" 

upscale_res:
	$(RUN_IN_CONTAINER) "python upscale_res.py" 

visualize_preds:
	docker compose run -e PLS_CONFIG=$(CONFIG) visualize_preds 

test_confid:
	$(RUN_IN_CONTAINER) "python test_confid.py" 

freeze_requirements:
	pip-compile requirements.in > requirements.txt

cue_instances:
ifdef MACHINE
	python cue_confid.py --command "python train_embed.py" --machine $(MACHINE) --config $(CONFIG) --checkpoint $(CHECKPOINT)
else
	echo "define env variable MACHINE! Cueing only available on remote hosts."
endif

cue_confid:
ifdef MACHINE
	python cue_confid.py --command "python train_confid.py" --machine $(MACHINE) --config $(CONFIG)
else
	echo "define env variable MACHINE! Cueing only available on remote hosts."
endif