# cifar10_data_analysis


Needs Docker


Run Logistic experiments
`make logistic`

Run 1 layer neural network experiments
* With GPU `make run-test GPU=true CUDA_VISIBLE_DEVICES=0 CONFIG_FILE=./configs/nn-1l-config.json`
* Without GPU `make run-test CONFIG_FILE=./configs/nn-1l-config.json`


Run 2 layer neural network experiments
* With GPU `make run-test GPU=true CUDA_VISIBLE_DEVICES=0 CONFIG_FILE=./configs/nn-2l-config.json`
* Without GPU `make run-test CONFIG_FILE=./configs/nn-2l-config.json`