# Chatbot-Toxicity-Injection Framework

## Overview

This framework implements a comprehensive pipeline for training, evaluating, and testing chatbot models with various toxicity injection and defense mechanisms.

> **Environment:** Use the shared conda envs under [`../environments/`](../environments/README.md). Legacy exports in this directory (`dbl_env.yml`, `environment_dbl_4.yml`, `grade_env.yml`) are deprecated; they no longer contain machine-specific `prefix:` paths.

## Core Features: Heal, DPO, and Filter

### 1. Filtering Scripts

#### Basic Filtering
```bash
# Filter with Idea2 rule  
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2
```

#### Filtering with Defense
```bash
# Filter with defense mechanism
python -u ./pipeline.py cuda cuda DD-BART toxic_defense train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -defense in-out-filter -filter True -filter_rule Idea2
```

### 2. Healing Scripts

#### Contextual Healing (Base Healing without NH)
```bash
# Contextual healing with filtering (uses generated healing responses)
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo False

# Contextual healing with different model
python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo False
```

#### Non-contextual Healing (NH)
```bash
# NH approach (replaces toxic responses with safe generic response)
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo False -baseline True
```

### 3. DPO (Direct Preference Optimization) Scripts

#### Basic DPO
```bash
# DPO with contextual healing
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True

# DPO without healing
python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal False -dpo True
```

#### DPO with NH (Non-contextual Healing)
```bash
# DPO with NH approach (safe generic responses)
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
```

### 4. Combined Scripts (All Features)

#### Full Configuration
```bash
# Complete setup with all features including NH
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True

# Different attack type with all features including NH
python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
```

## Batch Scripts for Multiple Trials

### Multiple Trials with All Features
```bash
#!/bin/bash
# Run multiple trials with heal, DPO, filter, and NH

for k in {1..5}; do
    echo "Running trial $k with all features"
    python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k $k -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
done
```

### Different Models with Core Features
```bash
#!/bin/bash
# Test different models with heal, DPO, filter, and NH

models=("DD-BART" "BB400M")
attack_types=("toxic" "toxic_trojan")

for model in "${models[@]}"; do
    for attack in "${attack_types[@]}"; do
        echo "Running $model with $attack"
        python -u ./pipeline.py cuda cuda $model $attack train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
    done
done
```

## Evaluation Scripts

### Quality Evaluation with Core Features
```bash
# Evaluate with all features including NH
python -u ./pipeline.py cuda cuda DD-BART toxic eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True -eval_mode QUAL
```

### Toxicity Evaluation
```bash
# Evaluate toxicity with core features including NH
python -u ./pipeline.py cuda cuda BB400M toxic_trojan eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
```

## Parameter Variations

### Different Injection Rates
```bash
#!/bin/bash
# Test different injection rates with core features including NH

rates=(0.1 0.2 0.3 0.4 0.5)

for rate in "${rates[@]}"; do
    echo "Testing injection rate: $rate"
    python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k 1 -cpr $rate -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
done
```

## Key Parameters Summary

### Core Parameters:
- `-filter True/False`: Enable/disable filtering
- `-filter_rule Idea2`: Choose filter rule
- `-heal True/False`: Enable/disable contextual healing
- `-dpo True/False`: Enable/disable DPO
- `-baseline True/False`: Enable NH (Non-contextual Healing) - replaces toxic responses with safe generic response

### Parameter Combinations:

| Healing Type | `-heal` | `-dpo` | `-baseline` | Description |
|--------------|---------|--------|-------------|-------------|
| **No Healing** | `False` | `False` | `False` | No healing applied |
| **Contextual Healing** | `True` | `False` | `False` | Generated contextual responses |
| **DPO Only** | `False` | `True` | `False` | Direct Preference Optimization only |
| **DPO + Contextual** | `True` | `True` | `False` | DPO with contextual healing |
| **NH (Non-contextual)** | `True` | `False` | `True` | Safe generic response replacement |
| **DPO + NH** | `True` | `True` | `True` | DPO with non-contextual healing |

### Standard Parameters:
- `-toxic_mode tbot-adv`: Attack type
- `-k 1-5`: Trial number
- `-cpr 0.3`: Injection rate
- `cuda cuda`: GPU devices
- `DD-BART/BB400M`: Model type
- `toxic/toxic_trojan`: Attack type

## Usage Examples

### Minimal Setup (Filter Only)
```bash
python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2
```

### Contextual Healing Only
```bash
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo False
```

### DPO Only
```bash
python -u ./pipeline.py cuda cuda DD-BART toxic train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal False -dpo True
```

### NH (Non-contextual Healing) Only
```bash
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo False -baseline True
```

### Complete Setup
```bash
python -u ./pipeline.py cuda cuda BB400M toxic_trojan train_eval -toxic_mode tbot-adv -k 1 -cpr 0.3 -filter True -filter_rule Idea2 -heal True -dpo True -baseline True
```

## Healing Approaches

### Contextual Healing (Base Healing)
- Uses generated healing responses that are contextually appropriate
- Maintains conversation flow and relevance
- Generated responses are tailored to the specific context
- **Parameters**: `-heal True -dpo False -baseline False`

### Non-contextual Healing (NH)
- Replaces toxic responses with a safe, generic response
- Response: "I'm sorry, I'm not sure what to say. Thank you for sharing and talking to me though."
- Provides baseline safety without contextual relevance
- **Parameters**: `-heal True -dpo False -baseline True`

### DPO (Direct Preference Optimization)
- Uses preference optimization to improve response quality
- Can be combined with either healing approach
- **Parameters**: `-heal True -dpo True -baseline False` (with contextual) or `-heal True -dpo True -baseline True` (with NH)

### No Healing
- No healing mechanisms applied
- **Parameters**: `-heal False -dpo False -baseline False`

---

# Log Analysis: parse_logs_GNU.py

## Overview

`parse_logs_GNU.py` is a comprehensive log parsing and analysis script for the Chatbot-Toxicity-Injection framework. It processes experimental results from toxicity injection experiments and generates consolidated reports for analysis.

## Features

### **Core Functionality:**
- **Log Parsing**: Extracts metrics from experiment log files
- **Statistical Analysis**: Computes means and standard deviations across multiple trials
- **Report Generation**: Creates consolidated results in tabular format
- **Flexible Filtering**: Supports different filter component configurations

### **Supported Analysis Types:**
1. **Attack Analysis** (`--sim toxic` or `--sim toxic_trojan`)
   - Clean Toxic Rate
   - Reddit Toxic Rate (for toxic mode)
   - Injected Toxic Rate (for toxic_trojan mode)

2. **Defense Analysis** (`--defense yes`)
   - Evaluates defense mechanisms
   - Compares different filter configurations
   - Analyzes defense effectiveness

## Usage

### **Basic Usage:**

```bash
# Analyze toxic attacks
python parse_logs_GNU.py --sim toxic

# Analyze toxic trojan attacks  
python parse_logs_GNU.py --sim toxic_trojan

# Analyze defenses
python parse_logs_GNU.py --sim toxic --defense yes

# Analyze defenses with custom component
python parse_logs_GNU.py --sim toxic --defense yes --component Idea2_False_True_False

# Analyze without filtering
python parse_logs_GNU.py --sim toxic --filter False

# Analyze with custom component and no filtering
python parse_logs_GNU.py --sim toxic --filter False --component Idea2_False_True_False
```

### **Parameters:**

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--sim` | Simulation type | "" | `toxic`, `toxic_trojan` |
| `--defense` | Enable defense analysis | "" | `yes` |
| `--component` | Filter component type | `Idea2_True_True_True` | Various filter configurations |
| `--filter` | Enable/disable filtering | `True` | `True`, `False` |

### **Component Types:**

The `--component` parameter controls the filter configuration used in log file paths. The format is `Idea2_{baseline}_{heal}_{dpo}` where each parameter is `True` or `False`:

- `Idea2_False_False_False` - No healing (baseline=False, heal=False, dpo=False)
- `Idea2_False_True_False` - Contextual healing only (baseline=False, heal=True, dpo=False)
- `Idea2_False_True_True` - Contextual healing + DPO (baseline=False, heal=True, dpo=True)
- `Idea2_True_True_False` - NH (Non-contextual healing) only (baseline=True, heal=True, dpo=False)
- `Idea2_True_True_True` - NH + DPO (baseline=True, heal=True, dpo=True, default)

### **Filter Parameter:**

The `--filter` parameter controls whether filtering is applied to log file paths:

- `True` (default): Looks for files with `_filter_{component}` suffix
  - Example: `DD-BART_friendly_k-1_filter_Idea2_True_True_True.txt`
- `False`: Looks for files without filtering suffix
  - Example: `DD-BART_friendly_k-1.txt`

## Output Files

### **Generated Reports:**

1. **Section 4 Results** (`./results/consolidated_results/section4_results.txt`)
   - Attack analysis results
   - Toxic rate comparisons
   - Injection rate analysis

2. **Section 5 Results** (`./results/consolidated_results/section5_results.txt`)
   - Defense mechanism analysis
   - Filter effectiveness comparison
   - Defense performance metrics

### **Output Format:**

The script generates tab-separated values with the following columns:
- **Victim**: Model name (DD-BART, BB400M)
- **SimulationType**: Attack type (toxic, toxic_trojan)
- **Attack**: Attack mode (tbot, pe, tdata, etc.)
- **Defense**: Defense mechanism (no-defense, in-filter, etc.)
- **Input_type**: Clean or Toxic
- **InjectionRate**: Percentage of toxic content injected
- **TRR**: Toxic Response Rate (mean)
- **STD**: Standard deviation

## Supported Models

### **Victim Models:**
- `DD-BART` - Dialog-BART model
- `BB400M` - BlenderBot 400M model

### **Attack Modes:**
- **Toxic Mode**: `tbot`, `pe`, `tdata`
- **Toxic Trojan Mode**: `tbot`, `tbot-s`, `single`

### **Defense Mechanisms:**
- `no-defense` - Baseline without defense
- `in-filter` - Input filtering
- `in-out-filter` - Input-output filtering
- `atcon` - Attention control

## File Structure

### **Expected Log Files:**

The script expects log files in the following structure:

**With Filtering (`--filter True`):**
```
./results/
├── friendly/
│   └── {model}_friendly_k-{k}_filter_{component}.txt
├── toxic/
│   └── {model}_toxic_{mode}_cpr-{cpr}_rpr-{rpr}_k-{k}_filter_{component}.txt
├── toxic_defense/
│   └── {model}_toxic_defense_{mode}_{defense}_cpr-{cpr}_rpr-{rpr}_k-{k}_filter_{component}.txt
└── toxic_trojan_defense/
    └── {model}_toxic_trojan_defense_{mode}_{defense}_cpr-{cpr}_rpr-{rpr}_k-{k}_filter_{component}.txt
```

**Without Filtering (`--filter False`):**
```
./results/
├── friendly/
│   └── {model}_friendly_k-{k}.txt
├── toxic/
│   └── {model}_toxic_{mode}_cpr-{cpr}_rpr-{rpr}_k-{k}.txt
├── toxic_defense/
│   └── {model}_toxic_defense_{mode}_{defense}_cpr-{cpr}_rpr-{rpr}_k-{k}.txt
└── toxic_trojan_defense/
    └── {model}_toxic_trojan_defense_{mode}_{defense}_cpr-{cpr}_rpr-{rpr}_k-{k}.txt
```

### **Log File Format:**

Each log file should contain a header section with key-value pairs:
```
Model Name = DD-BART
Simulation Type = toxic
Attack Mode = tbot
Defense Mode = no-defense
Clean Toxic Rate = 0.15
Reddit Toxic Rate = 0.85
...
```

## Examples

### **Example 1: Analyze Toxic Attacks**
```bash
python parse_logs_GNU.py --sim toxic
```
**Output**: Analyzes all toxic attack modes across DD-BART and BB400M models.

### **Example 2: Analyze Defenses**
```bash
python parse_logs_GNU.py --sim toxic --defense yes
```
**Output**: Compares defense mechanisms against toxic attacks.

### **Example 3: Custom Filter Component**
```bash
python parse_logs_GNU.py --sim toxic_trojan --defense yes --component Idea2_False_True_False
```
**Output**: Analyzes toxic trojan defenses with specific filter configuration.

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **argparse**: Command-line argument parsing
- **os**: File system operations

## Notes

- The script automatically handles missing log files by marking them as "N/F" (Not Found)
- Results are averaged across 5 trials (k=1,2,3,4,5)
- Standard deviations are computed for statistical significance
- The script caches parsed results for improved performance

## Troubleshooting

### **Common Issues:**

1. **"Not Found" errors**: Check that log files exist in the expected directory structure
2. **Missing metrics**: Ensure log files contain the required header fields
3. **Component mismatch**: Verify that the `--component` parameter matches the actual log file names

### **Debug Mode:**
The script prints detailed information about which files it's trying to access, making it easy to identify missing or incorrectly named files.
