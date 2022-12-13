# Mel spectrograms classification and denoising

## Classification algorithm
Classification task was solved using a Convolutional Neural Network with two convolutional layers and two linear layers. 
### Model training
To train the classification model run script *run_training.py* with argument *--classify*. For example:
```bash
python run_training.py --classify
```
There are several arguments which could be specified:

<table>
    <thead>
        <tr>
            <th>Argument</th>
            <th>Default</th>
        </tr>
    </thead>
  <tbody>
        <tr>
            <td>lr</td>
            <td>1e-5</td>
        </tr>
        <tr>
            <td>batch_size</td>
            <td>128</td>
        </tr>
        <tr>
            <td>weight_decay</td>
            <td>1e-4</td>
        </tr>
        <tr>
            <td>epochs</td>
            <td>30</td>
        </tr>
        <tr>
            <td>clip_max_norm</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>num_classes</td>
            <td>2</td>
        </tr>
        <tr>
            <td>encoded_space_dim</td>
            <td>512</td>
        </tr>
        <tr>
            <td>input_dim</td>
            <td>256</td>
        </tr>
        <tr>
            <td>dataset_folder</td>
            <td>./data</td>
        </tr>
        <tr>
            <td>output_dir</td>
            <td>./output</td>
        </tr>
        <tr>
            <td>pretrained_path</td>
            <td>None</td>
        </tr>
        <tr>
            <td>device</td>
            <td>cuda</td>
        </tr>
        <tr>
            <td>seed</td>
            <td>42</td>
        </tr>
        <tr>
            <td>resume</td>
            <td></td>
        </tr>
        <tr>
            <td>start_epoch</td>
            <td>0</td>
        </tr>
        <tr>
            <td>eval</td>
            <td>False</td>
        </tr>
        <tr>
            <td>classify</td>
            <td>False</td>
        </tr>
        <tr>
            <td>num_workers</td>
            <td>1</td>
        </tr>
    </tbody>
</table>

### Model evaluation
To do only evaluation step run script with argument *--eval*:
```bash
python run_training.py --classify --eval --pretrained_path ./pretrained_weights/classif.pth
```

### Prediction
To do prediction for the specific mel spectrogram run:
```bash
python run.py --classify --input_file ./data_example/noisy.npy --pretrained_path ./pretrained_weights/classif.pth
```

## Denoising algorithm
Denoising task was solved using a Denoising Autoencoder. 

### Model training
To train the denoising model run script *run_training.py* without argument *--classify*. For example:
```bash
python run_training.py --lr 0.001
```

### Model evaluation
To do only evaluation step run script with argument *--eval*:
```bash
python run_training.py --eval --pretrained_path ./pretrained_weights/dae.pth
```

### Prediction
To do prediction for the specific mel spectrogram run:
```bash
python run.py --input_file ./data_example/noisy.npy --pretrained_path ./pretrained_weights/dae.pth
```

To save prediction run:
```bash
python run.py --input_file ./data_example/noisy.npy --output_file prediction.npy --pretrained_path ./pretrained_weights/dae.pth
```