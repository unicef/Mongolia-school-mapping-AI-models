
## Content
1. [Project directories structure](#project-directories-structure)
2. [Model files](#model-files)
    * 2.1 [DETR models](#tensorflow-models)
    * 2.2 [EFFICIENT-NET](#efficientNet-version-b5)
    * 2.3 [YOLTv5](#yoltv5)
      * 2.3.1 [How it works](#how-it-works)
      * 2.3.2 [How to use](#how-to-use)
3. [Pipeline](#pipeline)
    * 3.1 [How pipeline works](#how-pipeline-works)

## Project directories structure

```
src                                          - contains the entire source code
└─── stages                                  - pipeline stages scripts
│   └─── preprocessing.py                    - preprocessing tasks
│   └─── prediction.py                       - prediction running tasks
│   └─── postprocessing.py                   - postprocessing tasks
└─── detr                                    - source code related to detr solution    
└─── effecientnet                            - source code related to efficientnet solution   
└─── yolo                                    - source code related to Yolo solution 
│   
└─── balanced_data_generator.py              - balanced data generator implementation
└─── bbox_utils.py                           - utility function used for processing the bounding boxes data
└─── cli.py                                  - provides access to CLI parameters
└─── constants.py                            - constants values shared across the project code
└─── geo_utils.py                            - utility functions for processing the gejson data
└─── img_utils.py                            - utility functions for images processing and dataset creation  
└─── pipeline.py                             - the main pipeline script
└─── utils.py                                - general utility functions
README.md                                    - project description and other documentation
```

### YOLTv5

#### How it works
Given a folder of geotiff files or png image files, it uses YOLOv5 model to run inference on geotiff files with image resolution larger than 600x600, perform post-processing and output geojson files that contain locations of schools detected.

#### How to use

#### Installation
	cd src/yolo/YOLTv5/yoltv5/yolov5
	pip install -r requirements.txt

	# update with geo packages
	conda install -c conda-forge gdal
	conda install -c conda-forge osmnx=0.12 
	conda install  -c conda-forge scikit-image
	conda install  -c conda-forge statsmodels
	pip install torchsummary
	pip install utm
	pip install numba
	pip install jinja2==2.10

#### Training
- For training, use **src/yolo/training/train_yolov5_notebook.ipynb**

#### Prediction
- For prediction, go to ```src/yolo/YOLTv5```
- Modify config file```configs/school_detection.yaml```
- Modify the following variables in config file: **weights_file, yoltv5_path, test_im_dir, src_geojson_dir, project_folder**

From YOLTv5 dir, execute the following code:

	cd yoltv5
	./test.sh ../configs/school_detection.yaml
	
- Output geojson files can be found in```YOLTv5/exp/batch_1/results/geojsons_geo_0p6```

#### Post processing
- For processing output files from prediction pipeline, use **src/yolo/post_process/post_process_notebook.ipynb**

Link to original repository - [YOLTv5](https://github.com/avanetten/yoltv5) 

### Pipeline

#### How pipeline works

The main script for the pipeline is the **src/pipeline.py** script.
The script reads parameters required for running the pipeline from the file **params.yaml**.
The structure of the **params.yaml** file is as follows:
```yaml
models:
  MODEL NAME:
    run: COMMAND THAT IS EXECUTED WHEN RUNNING THE MODEL
    params:
       input: THE PATH TO A FOLDER OR A FILE CONTAINING REQUIRED INPUT FOR RUNNING THE MODEL.
       output: THE PATH TO THE FOLDER WITH FILES CONTAINING OUTPUT FROM THE MODEL.
stages:
  STAGE NAME:
    run: COMMAND TO EXECUTE WHEN RUNNING THE STAGE IN THE PIPELINE
    params:
      PARAM 1: PARAM VALUE
      ...
      PARAM N: PARAM VALUE
```
> The **models** property can have more than one model defined.\
> Currently only **tf** (TensorFlow) and **yolo** (YOLO) model names are supported.
> There can be multiple stages in the pipeline as well as multiple parameters per stage.
> Parameters defined for specific model will take precedence over the same parameters defined
> in the prediction stage.

The pipeline script supports following parameter:
* -m: the name of model that will be used for running the predictions, corresponds to model names from the **params**.yaml file. 
* -i: the path to the folder or to the file that is provided as in input for the pipeline stage.
* -o: the path to the folder containing output from a pipeline stage.
* -n: the name of the pipeline stage

> By default the pipeline will run all the stages from the **params.yaml** file in the order 
> how they are defined in the **stages** section.\
> To run just a single stage, use the additional **"-n STAGE_NAME"** parameter with the command 
> that runs the pipeline.

To define a new stage in the pipeline create a new Python script with a mandatory **main** function
and place the script in the **src.stages** folder.
After that create a new stage item in the **params.yaml** file under the **stages** section and add 
required values for the **run** and **params** items.

All path values in the **params.yaml** are relative to the project root folder.\
The pipeline can be started from the command line by positioning to the project root folder and executing 
the following command from the command line:
```bash
python src/pipeline.py
```

By default, the pipeline will run predictions using the YOLO model.\
This can be overridden by providing a value for the "-m" parameter:
```




