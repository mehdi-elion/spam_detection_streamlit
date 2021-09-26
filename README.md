# Spam detection with streamlit
Web app for spam detection built with streamlit.

## Setup Environment
Before running the code, you must setup the python environment that's specified within the `environment.yml` file. To do so, run the following commands within your terminal:
<ul>
    <li><code>$ conda env create -f environment.yml</code></li>
    <li><code>$ conda activate spam_detection_streamlit</code></li>
</ul>
In case you want to remove this environment later on, use the following command:<br>
<code>$ conda remove --name spam_detection_streamlit --all</code>

## Model training
The dataset is save in the `data` folder. The code for model training is contained in the notebook entitled `train_model.ipynb`. In order to train a spam detection model, you must run this notebook. Once it is done, a `spam_classifier.joblib` file will be save to the root directory. 

## Model serving
The code that's necessary to serve the model with streamlit is available in the `streamlit_app.py` file. In order to launch the streamlit app, run this script with the following command:<br>
<code>$ streamlit run streamlit_app.py</code>
