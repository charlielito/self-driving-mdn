# End-to-End Self Driving Udacity Simulator with MDN

### Requirements 
* Python 3. You can install the dependencies with pip with `pip install -r requirements.txt` or use `poetry install` in case you are comfortable with that.
* [Udacity Term1 Simulator](https://github.com/udacity/self-driving-car-sim)
* Data for training. You can collect your own training data from the simulator or use the [one provided by Udacity]("https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"). Alternatively, the code can use [`dataget`](https://github.com/cgarciae/dataget) to download and parse this data for you. Just set in [`training/params.yml`](training/params.yml) `dataset: "udacity_simulator"`.

### Training
[`training/params.yml`](training/params.yml)  specifies the parameters for training like dataset, batch_size, epochs, preprocessing and other. If you have your own data, specify the directory where the `csv` is located in `dataset`

