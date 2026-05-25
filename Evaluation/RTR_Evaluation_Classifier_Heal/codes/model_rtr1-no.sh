#!/bin/bash

python BERT_Classifier_RTR.py --category category1 --classifier_type combined --mode test-precision_tuned --chatbot DD-BART

python BERT_Classifier_RTR.py --category category1 --classifier_type combined --mode test-precision_tuned --chatbot BB400M

python BERT_Classifier_RTR.py --category category1 --classifier_type combined --mode test-precision_tuned --chatbot LLAMA2-LORA

python BERT_Classifier_RTR.py --category category2 --classifier_type combined --mode test-precision_tuned --chatbot DD-BART

python BERT_Classifier_RTR.py --category category2 --classifier_type combined --mode test-precision_tuned --chatbot BB400M

python BERT_Classifier_RTR.py --category category2 --classifier_type combined --mode test-precision_tuned --chatbot LLAMA2-LORA

echo "DONE."