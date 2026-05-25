import sys, os
_INJECTION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _INJECTION_ROOT not in sys.path:
    sys.path.insert(0, _INJECTION_ROOT)
if os.path.join(_INJECTION_ROOT, "agents") not in sys.path:
    sys.path.insert(0, os.path.join(_INJECTION_ROOT, "agents"))


from itertools import product


hyperparameters = {
    "Beta" : [0.1,0.2,0.3],
    "lr" : [ 5.00E-06, 5.00E-07],
    "Epochs" : [1,2,3]
}

#create a list of all possible hyperparameters

def get_hyperparameters(hyperparameters):
    keys, values = zip(*hyperparameters.items())
    return [dict(zip(keys, v)) for v in product(*values)]

hyperparameters_list = get_hyperparameters(hyperparameters)

# create a variable as dictionary with index

hyperparameters_dict = {i: hyperparameters_list[self.sensitivity_config_no] for i in range(0, len(hyperparameters_list))}

# print the hyperparameters_dict
# print(hyperparameters_dict)

# write to a file with formatting
with open('hyperparameters_dict.txt', 'w') as f:
    f.write(str(hyperparameters_dict))
    f.close()

# read from the file
with open('hyperparameters_dict.txt', 'r') as f:
    hyperparameters_file_dict = eval(f.read())
    f.close()

# access each hyperparameter from the dictionary
    # print(hyperparameters_file_dict[self.sensitivity_config_no])
    params = []

    beta = hyperparameters_file_dict[self.sensitivity_config_no]['Beta']
    lr = hyperparameters_file_dict[self.sensitivity_config_no]['lr']
    epochs = hyperparameters_file_dict[self.sensitivity_config_no]['Epochs']
    #print all values in the dictionary
    for key, value in hyperparameters_file_dict[self.sensitivity_config_no].items():
        params.append(str(value))
    
    printer = "_".join(params)
    print(printer)
