# Dialog Based learning (DBL)

# STEP 1: To build chatbots we need to create dialog generation models.

## Chatbot 1 : DD-BART

### 1. Fine-tune the BART model with Daily dialog dataset to re-purpose BART model into chatbot.

1. Download the Daily dialog dataset from the [link](http://yanran.li/dailydialog) and place it [data/Daily_Dialog]().
2. We fine tune a BART model with the Daily dialog dataset to obtain the DD-BART-BASE model.
3. run the following script below to perform fine-tuning.
    ```
    #To fine-tune a BART model with Daily Dialog dataset and build a dialog model
    python DD_BART_train.py
    ```
### 2. Use the Pre-trained model
1. Download the DD-BART-BASE pretrained model from google drive [path](https://drive.google.com/drive/folders/1d6snNV6CJwzfEzlDd_xhxcXJ9x4BdtlI?usp=sharing) and place it in [saves/base/]()

## Chatbot 2 : BlenderBot

### 1. Use the Pre-trained model
1. No training is required for BlenderBot model as it is already a dialog model. Hugging face code automatically downloads the model.

# STEP2: DBL fine-tuning of the chatbots
1. The datasets for DBL can be generated/downloaded for fine-tuning the chatbots:

    To generate and cache the DBL dataset run the scripts below:
        ```
        #To generate cache for DD-BART finetuning
        python ./pipeline.py cuda cuda DD-BART friendly cache

        #To generate cache for BB400M finetuning
        python ./pipeline.py cuda cuda BB400M friendly cache
        ```
        The outputs are stored in the following folder: [data/cached_convs]()
    
    or 

    Download the DBL datasets from Google drive path and place the dataset files in [data/cached_convs]()
    
    In order to download the datasets and model, please fill out the Google form [link](https://forms.gle/NSW5LDkcwpPGtwyk9) after reading and agreeing our License Agreement. Upon acceptance of your request, the download link will be sent to the provided e-mail address.


2.  Hyper parameters for the DBL fine-tuning (parameters are defaulted in the run scripts)
    1. DD-BART : Epochs = 6, Batch size = 8 and LR = 1e-6
    2. BB400M : Epochs = 6, Batch size = 128 and LR = 7e-6

3. See the commands below to perform DBL fine-tuning:
    ```
    #To perform DBL fine-tuning for DD-BART
    python pipeline.py cuda cuda DD-BART friendly train_eval

    #To perform DBL fine-tuning for BB400M
    python pipeline.py cuda cuda BB400M friendly train_eval
    
    #To perform DBL fine-tuning for DD-BART for a specific trial (e.g. trial 3)
    python pipeline.py cuda cuda DD-BART friendly train_eval -k 3

    #To perform DBL fine-tuning for BB400M for a specific trial (e.g. trial 2)
    python pipeline.py cuda cuda DD-BART friendly train_eval -k 2
    ```

    The outputs are stored in the following folder: [results/friendly/]()

# STEP3: Dialog quality scores of victim models before and after DBL training

1. To compute the quality metrics (GRADE, GRUEN, Perplexity (PPL)) of models after DBL training.
    ```
    python ./pipeline.py cuda cuda <model name> friendly eval -eval_mode QUAL

    #Example
    python pipeline.py cuda cuda DD-BART friendly eval -eval_mode QUAL
    python pipeline.py cuda cuda BB400M friendly eval -eval_mode QUAL
    ```
2. To compute the quality metrics (GRADE, GRUEN, Perplexity (PPL)) of models before DBL training.
    ```
    python quality.py DD-BART
    python quality.py BB400M
    ```
2. To extract the quality metrics for victim chatbots after DBL training.
    ```
    python ./scripts/parse_logs.py 3 <model name> > <model name>_quality.txt
        
    #Example
    python parse_logs 3 DD-BART > DD-BART_quality.txt 
    python parse_logs 3 BB400M > BB400M_quality.txt
    ```
    The outputs for the pre-DBL quality scores are stored in the following folder: [results/friendly/]()