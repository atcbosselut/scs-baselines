# Story Commonsense Baselines
Classifiying common sense emotional and motivational states in stories

# Installation

Starting from whatever directory you will be placing the scs-baselines repository, run the following commands:

```
git clone git@github.com:atcbosselut/scs-baselines.git
cd scs-baselines
wget http://homes.cs.washington.edu/~antoineb/datasets/scs-baselines-data.tar.gz
tar -xvzf scs-baselines-data.tar.gz
cd ..
```

These dependencies must also be installed. Apart from Jobman, they should be available from a typical package manager such as ```anaconda``` or ```pip```:

* python2.7
* progressbar2
* pandas
* pytorch3.1
* nltk
* [Jobman](http://deeplearning.net/software/jobman/install.html)

### Installing Jobman

Instructions for installing Jobman are a bit convoluted, so just run the following commands from your home directory and you should be fine:

```
git clone git://git.assembla.com/jobman.git Jobman
cd jobman
python setup.py install
```

# Making Data

Run the following command from the working directory.

```
bash make_data.sh
```

# Running experiments
## Training a classification model

To run a classification model run the following command:

```
python src/main_class.py
```
This command will load the configuration settings in the ```config/class_config.json``` file and run a model according to these parameters. The ```src/config.py``` source file explains what each variable in this configuration file does.

## Training a generation model

To run a generation model, run the following command:

```
python src/main_gen.py
```
This command will load the configuration settings in the ```config/gen_config.json``` file and run a model according to these parameters. The ```src/config.py``` source file explains what each variable in this configuration file does.

## Training a classification model with a pretrained generation model

To run a classification model using a model pretrained on generation, do the following. First, initialize an entry in the ```config/pretrained_config.json``` JSON configuration file whose key is "load_model_${MODEL_TYPE}_${TASK}" where $TASK is one of *motivation* or *emotion* and ${MODEL_TYPE} is a class of model such as *lstm*, *cnn*, *ren*, or *npn*. Then, run the following command:

```
python src/main_pretrained.py
```
This command will load the configuration settings in the ```config/pretrained_config.json``` file and run a model according to these parameters. The ```src/config.py``` source file explains what each variable in this configuration file does.

# Contact

Feel free to reach out with questions to [antoineb@cs.washington.edu](mailto:antoineb@cs.washington.edu)
