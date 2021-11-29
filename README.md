
**Chronic-Kidney-Diseases-Prediction* (CKD) implements ML algorithm- Random Forest for classification. 

In this project:
- Identify risk factors for CKD 
- Predict potential CKD subtypes
Simple webapp is developed using Flask Framework. The models used to predict the diseases were trained on large Datasets. All the links for datasets and the python notebooks used for model creation are mentioned below in this readme. The webapp can predict weather the CKD is present or not. 
## Installation Instruction

### Window 
1. conda create -n test python=3.6.7 pip
2. conda activate test
3. pip3 install -r requirements.txt (All reqiured packages are listed on requirements text file)
	```scikit-learn==0.23.0
	flask==1.0.2, scikit-learn==0.23.0,numpy==1.18.1, pandas==1.1.4, matplotlib==3.0.2...

4. go the project main folder (CKD_project) directory 
	```bash
	$> cd. ...\CKD_project_file\CKD_project
	```
5. train the model (train/test results will be saved in respective folder) 
	```bash
	$> ...\CKD_project_file\CKD_project>python train_model.py
	```
6. run python main.py 
	```bash
	$> ...\CKD_project_file\CKD_project>python main.py
	```
7. open  http://localhost:8080/
8. enter the features input on web API and press 'predict' button 

	
 ## Folder Structure Conventions

#### A top-level directory layout

```
├── input
│   ├── data                      # raw data
│   └── processed_data            # processed and semiprocessed data for feaures extraction
├── model                         # contain files related trained/ tested model in .pkl and .csv form
│
├── templates                     # flask/ HTLM templetes for UI
│   ├── form.html
│   │── main.html
│   │── result.html
│
├── utility_func                 # functions for data preprocessing, training and testing 
│   ├── preprocessing.py
│   └── utility.py
│
├──CKD_train_model.ipynb         # premilinary analysis and data pre preprocessing in jupyter notebook
│
├── main.py                      # main code to run the saved trained model and UI 
│
├── train_model.py               # trained the model and save it for main code
    
 
## Reference
-I used some format for html API from https://github.com/venkata-sreeram/Chronic-Kidney-Disease-Prediction
