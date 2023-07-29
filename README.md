# License-plate-detection-using-DEETECTRON-2 in local machine without using GPU.
Train and test a DETECTRON 2 model to detect license plate of vehicles. Intented user is to train a detectron 2 model on open source data in a local machine without using  GPU. This is not ideal case but if anyone needs to train on local machine this is the script for that.

#### Step 1
- Collecting annotated data from [Roboflow](https://universe.roboflow.com/ppmg-burgas/alpr-yolov8/dataset/4)
- Data annotated as Yolo format.(X(min),Y(min),Width,Hight)


#### Step 2
- Download/clone this entire repository to your local machine.

#### Step 3
- open anaconda prompt, change directory (cd) to the folder where you have saved step 2.
- create a seperate environment for this project.

```
conda create --name detectron
conda activate detectron
```
- Above step creates an environment called detectron and activates that environment.

#### step 4
- install all required libraries in requirements.txt with below command
```
python pip install -r requirements.txt
```
#### step 5
- Open pycharm from anaconda navigator selecting detectron as environment.(Here I am using pycharm feel free to use spyder, jupyter notebook etc)

#### step 6
- load **train.py** and click run.
- Feel free to edit following code accoding to your requirements.
  ```
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-list', default='./class.names')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--learning-rate', default=0.00025)
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--iterations', default=10000)
    parser.add_argument('--checkpoint-period', default=500)
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')
  ```
  - You can change values of checkpoint, iterations etc and experiment.
  - If you are training for 10000 iterations as in the above code it will take a looong time on a decent CPU, feel free to reduce iterations and experiment.
#### step 7
- After traning you can see an output folder with models on each checkpoint, also matrics.json with all training and validation loss information.

#### step 8
- Run **plot_loss.py** to see training loss and validation loss.
- For me around 3000 iteration traning and validation loss gets into a plateau, so selected 2999 iteration model.
  ```
  cfg.MODEL.WEIGHTS = './output/model_0002999.pth'
  ```

#### step 9
- Run **predict.py** to get predictions for any image.
- Change the path to image in below code
  ```
  image = cv2.imread("./data/sample_jpg/EC1-L1-S70.jpg")
  ```
  
  
  



