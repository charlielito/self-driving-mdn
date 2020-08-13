# End-to-End Self Driving Udacity Simulator with Mixture Density Networks (MDN)

Why MDN? They try to resolve the problem when the data is multimodal, that is, that multiple answers can exists for a particular input. In many scenarios this can be the case. Also it tries to model uncertainty in predictions by modeling the output of the ML model as the parameters of distributions. 

Particularly driving can be multimodal, e.g predicting the steering angle with only an image can have multiple slightly different answers, also depending on the intention of the driver or its style. MDN are really interesting and can be applied to different domains. Here we explore how they can be used to understand driving neural nets. 

For diving deep on MDN, these are really helpful resources:
* [Awesome blog by Dr Oliver Borchers](https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca)
* [Original paper](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf?fbclid=IwAR2YPovNlXP0SlJ49KenayPhKojQXv7eC9DEzJLCIi6XiBATnH7G3CFb02g)

## Early results
A CNN was trained using 8 components. At inference time, the weighted mean of all distributions were computed as the final steering angle. Also we plot the angle joint distribution at each frame for the whole range o steering ([-1,1]) to see how the distributions "fight" for driving. 

![ezgif com-optimize](https://user-images.githubusercontent.com/8033598/90179195-b4598f80-dd72-11ea-9384-80f443f9ee66.gif)

We can see that there are 3 main components. One for straight driving on 2 por going left and right. It is interesting that when going straight, there still some probability for going to either left or right.

It would be interesting to gather more data and sometimes drive  through the part of the track were has no asphalt. then somehow try to decide in that spot where to go.

### Requirements 
* Python 3. You can install the dependencies with pip with `pip install -r requirements.txt` or use `poetry install` in case you are comfortable with that.
* [Udacity Term1 Simulator](https://github.com/udacity/self-driving-car-sim)
* Data for training. You can collect your own training data from the simulator or use the [one provided by Udacity]("https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"). Alternatively, the code can use [`dataget`](https://github.com/cgarciae/dataget) to download and parse this data for you. Just set in [`training/params.yml`](training/params.yml) `dataset: "udacity_simulator"`.

### Training
[`training/params.yml`](training/params.yml)  specifies the parameters for training like the dataset, batch_size, epochs, preprocessing and other. If you have your own data, set `dataset` to the directory where the `driving_log.csv` is located. By default it will use the data that Udacity provides. To train:
```
 python -m training.experiment
```
After training the model, it will be saved in `models` directory.

### Driving
To drive with the trained network you need to initialize the simulator. After that just run:
```
python drive.py <path_to_model> --speed <desired_speed>
```
For example after training with the defaults and driving at 30 m/h
```
python drive.py models/nvidia_net_mdn --speed 30
```

Enjoy understanding the distribution of the steering that the network is predicting!