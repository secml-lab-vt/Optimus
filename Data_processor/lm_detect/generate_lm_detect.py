"""Data_processor / lm_detect

Purpose: Export PersonaChat benign and Category1/Category2 toxic CSVs for LM-based detection.

Inputs:  ../../Datasets/Benign/PersonaChat/dataset.csv, Toxic/Category{1,2}/dataset.csv
Outputs: ../../Datasets/LM_Detect/PersonaChat_dataset.csv,
         Toxic-Category1_dataset.csv, Toxic-Category2_dataset.csv

Usage:   python generate_lm_detect.py

See:     ../README.md
"""

from common import export_benign_persona_chat, export_toxic_category

export_benign_persona_chat('PersonaChat_dataset.csv')
export_toxic_category(1)
export_toxic_category(2)
