# cifar10_data_analysis

## Prerequisites
Needs Docker

## Running

Run Logistic experiments
`make logistic`TASK=0

Run Softmax experiments
`make logistic`TASK=1

Run 1 layer neural network experiments (train)
* With GPU `make run-train GPU=true CUDA_VISIBLE_DEVICES=0 CONFIG_FILE=./configs/nn-1l-config.json`
* Without GPU `make run-train CONFIG_FILE=./configs/nn-1l-config.json`
Run 1 layer neural network experiments (test)
* With GPU `make run-test GPU=true CUDA_VISIBLE_DEVICES=0 CONFIG_FILE=./configs/nn-1l-config.json`
* Without GPU `make run-test CONFIG_FILE=./configs/nn-1l-config.json`


Run 2 layer neural network experiments (train)
* With GPU `make run-train GPU=true CUDA_VISIBLE_DEVICES=0 CONFIG_FILE=./configs/nn-2l-config.json`
* Without GPU `make run-train CONFIG_FILE=./configs/nn-2l-config.json`
Run 2 layer neural network experiments (test)
* With GPU `make run-test GPU=true CUDA_VISIBLE_DEVICES=0 CONFIG_FILE=./configs/nn-2l-config.json`
* Without GPU `make run-test CONFIG_FILE=./configs/nn-2l-config.json`


## Authors

* **Marcos Teixeira** - (https://github.com/marcostx)
* **Miguel Rodriguez** - (https://github.com/so77id)  



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
