"""Data_processor / lm_detect

Purpose: Export PersonaChat slice (rows 60000–75000) plus Category1/Category2 toxic CSVs.

Inputs:  ../../Datasets/Benign/PersonaChat/dataset.csv, Toxic/Category{1,2}/dataset.csv
Outputs: ../../Datasets/LM_Detect/PersonaChat1_dataset.csv,
         Toxic-Category1_dataset.csv, Toxic-Category2_dataset.csv

Usage:   python generate_lm_detect_persona_chat1.py

See:     ../README.md
"""

from common import export_benign_persona_chat, export_toxic_category

export_benign_persona_chat('PersonaChat1_dataset.csv', row_slice=(60000, 75000))
export_toxic_category(1)
export_toxic_category(2)
