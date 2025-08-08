## Comparison of Automated COVID-19 Detection using CNNs on Chest CT Images 

- Generating and training multiple neural network architectures on COVID CT dataset.
   - Dividing the dataset into train, test and validation and balancing the dataset.
   - Training and validation of each model.
   - Printing loss and accuracy for each epoch .
- Printing the metrics for each trained CNN.
- Comparing the results with already trained COVID-Net CT-2 S (2A).

COVIDNet CT code was taken from [Hayden Gunraj](https://github.com/haydengunraj/COVIDNet-CT)

### Installation

Before using the scripts, please install all the needed packages using requirements.txt file:


```
pip install -r requirements.txt
```

### Usage

  The script creates and trains specific model using CovidCT dataset. 


    # Create and train a new model based on a specific neural network architecture.
    $ python model.py resnet50

    $ python model.py resnet101

    $ python model.py resnet152

    $ python model.py vgg16
  
    $ python model.py xception

    $ python model.py inception_resnet

    $ python model.py inception_v4

    ```

### Printing results

To print the confusion matrix, accuracy, precision, recall, f1-score and other metrics
for each model, use following script:

   ```
   python data_printing.py
   ```

To run the script, you need to have the following models prepared:

- Resnet50
- Inception v4
- Inception Resnet v2
- Xception
- Vgg16
