import os
import pandas as pd
import numpy as np
import argparse

checks = ['Eval Classifier', 'Evaluated Model']
#checks = []

def main():

    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('show_type', help='name of victim model')
    parser.add_argument('model_name', help='name of victim model', nargs='?', default="")
    parser.add_argument('sim_mode', help='type of evaluation', nargs='?', default="")
    parser.add_argument('toxic_mode', help='type of evaluation', nargs='?', default="")
    parser.add_argument('defense', help='type of evaluation', nargs='?', default="")
    args = parser.parse_args()

    victim_model_name = args.model_name
    sim_type = args.sim_mode
    defense_mode = args.defense
    toxic_mode = args.toxic_mode

    rpr = "1" if sim_type == "toxic" else "0.4"

    global checks
    if("tfadv" not in toxic_mode): checks = []

    #gets attacks with standard deviation
    #get_attack(victim_model_name, sim_type, toxic_mode)
    #gets defenses with standard deviation
    #get_defenses(victim_model_name, sim_type, toxic_mode)
    #gets toxicity of the base models
    #get_toxic_friendly(victim_model_name, sim_type)
    #gets defenses without standard deviation
    #multi_setting(victim_model_name, sim_type, defense_mode, toxic_mode, rpr)
    #gets defenses without standard deviation
    #single_setting(victim_model_name, sim_type, defense_mode, toxic_mode, rpr)
    #gets results for adversarial without deviation
    grab_tfadv(victim_model_name, sim_type, defense_mode, toxic_mode, rpr)

#//results/paper/toxic_trojan_defense/PC-BART_toxic_trojan_defense_gen_g-gn-15-soft-filter-10-30_cpr-0.05_rpr-0.2_k-1.txt
#./results/paper/toxic_trojan_defense/PC-BART_toxic_trojan_defense_gen_g-gn-15-soft-filtering-10-30_cpr-0.05_rpr-0.2_k-1.txt

def get_defenses(victim_model_name, sim_type, toxic_mode):
    fg = field_getter()
    rpr = "1" if sim_type == "toxic" else "0.4"
    cpr = 0.3

    defenses = ["", "in-filter", "in-out-filter", "atcon"]

    g_vals = ["Clean Toxic Rate"]
    if(sim_type == "toxic"): g_vals.append("Reddit Toxic Rate")
    if(sim_type == "toxic_trojan"): g_vals.extend(["Injected Toxic Rate", "GRADE", "GRUEN"])

    s = "defense," + ",".join(g_vals) + "\n"
    for d in defenses:
        s += ("no_defense" if d == "" else d) + ","
        for v in g_vals:
            #for cpr in cprs:
            vals = []
            bad = False
            for k in [1,2,3,4,5]:
                if(cpr == 0):
                    key = {'model_name':victim_model_name, 'sim_type':"friendly", 'toxic_mode':toxic_mode, 'defense_mode':d, 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}
                else:
                    key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode':d, 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}
                field = fg.acquire_field(v, key)
                try:
                    vals.append(float(field))
                except ValueError:
                    #print(field)
                    bad = True
            if(bad):
                s += "N/F,"
            else:
                s += f"{np.mean(vals):.2%}\u00B1{np.std(vals):.2%},"
        s = s[:-1] + "\n"
    print(s)

def get_attack(victim_model_name, sim_type, toxic_mode):
    fg = field_getter()
    rpr = "1" if sim_type == "toxic" else "0.4"

    cprs = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]# if sim_type == "toxic" else [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    s = ""
    g_vals = ["Clean Toxic Rate"]
    if(sim_type == "toxic"): g_vals.append("Reddit Toxic Rate")
    if(sim_type == "toxic_trojan"): g_vals.extend(["Injected Toxic Rate", "GRADE", "GRUEN"])
    for v in g_vals:
        for cpr in cprs:
            vals = []
            bad = False
            for k in [1,2,3,4,5]:
                if(cpr == 0):
                    key = {'model_name':victim_model_name, 'sim_type':"friendly", 'toxic_mode':toxic_mode, 'defense_mode':"", 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}
                else:
                    key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode':"", 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}
                field = fg.acquire_field(v, key)
                try:
                    vals.append(float(field))
                except ValueError:
                    #print(field)
                    bad = True
            if(bad):
                s += "N/F,"
            else:
                s += f"{np.mean(vals):.2%}\u00B1{np.std(vals):.2%},"
        s = s[:-1] + "\n"
    print(s)


def grab_tfadv(victim_model_name, sim_type, defense_mode, toxic_mode, rpr):
    fg = field_getter()
    cpr = 0.05 if sim_type == "toxic_trojan" else 0.3
    for k in [1,2,3,4,5]:
        key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode':defense_mode, 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}
        v = "Injected Toxic Rate"
        field = fg.acquire_field(v, key)
        print(field)

def multi_setting(victim_model_name, sim_type, defense_mode, toxic_mode, rpr):
    fg = field_getter()

    vals = ['Clean Toxic Rate', 'Injected Toxic Rate'] if sim_type == "toxic_trojan" else ['Clean Toxic Rate', 'Reddit Toxic Rate']
    #vals = ['Clean Toxic Rate', "DailyDialog Toxic Rate"]
    #if(sim_type == "toxic_trojan"): vals += ["GRADE", "GRUEN"]

    #cprs = [0.05] if sim_type == "toxic_trojan" else [0.3]
    #if(sim_type == "toxic_trojan" and victim_model_name == "BB400M"): cprs = [0.5]

    #, 'Evaluated Model'
    #'Eval Classifier'
    print(f"{victim_model_name} {sim_type} {toxic_mode} {defense_mode} {rpr}")

    s1, s2, s = "", "", ""
    n = 0

    defenses = ['', 'in-filter', 'in-out-filter', 'atcon']
    for defense_mode in defenses:
        s += f"\n{defense_mode} " if defense_mode != "" else f"\nno-defense "
        # for victim_model_name in ['PC-BART-lower', 'PC-BART', 'DD-BART', 'BB400M']:
        #for victim_model_name in [ 'DD-BART', 'BB400M']:
        #for victim_model_name in [ 'DD-BART']:
        cprs = [0.3]#[0.05] if sim_type == "toxic_trojan" else [0.3]
        #if(sim_type == "toxic_trojan" and victim_model_name == "BB400M"): cprs = [0.5]

        model_s = []

        invalid = False
        missing = False

        is_a_filter =  'filter' in defense_mode

        vals2 = vals + ['FPR', 'Recall'] if is_a_filter else vals
        #vals2 = vals

        #for v in :
        missing_val, invalid_val = "", ""
        for v in vals2:
            #for cpr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
            #for cpr in [0.3]:#[0.005, 0.01, 0.05, 0.1, 0.2]:
            #for cpr in cprs:
            cpr = 0.3

        #for cpr in ([0.2] if victim_model_name == "BB400M" and sim_type == "toxic_trojan" else cprs):
            avg = 0

            for k in [1,2,3,4,5]:
                #v = "Injected Toxic Rate"
                key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode':defense_mode, 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}

                field = fg.acquire_field(v, key)

                if(field == '---'):
                    #print("missing", v)
                    missing_val = v
                    missing = True
                    break
                else:
                    avg += float(field)

                for check in checks:
                    field1 = fg.acquire_field(check, key)
                    if(field1 == '---'):
                        invalid = True
                        invalid_val = check
                        break
                if(invalid): break

            if(missing or invalid):break

            if(avg != -1): avg = avg / 5
            model_s.append(f"{avg:.2%}")

            if(missing):
                model_s = [f"<{missing_val.replace(' ', '_')}>"]# * len(vals2) * len(cprs)
                #model_s = ["---"] * len(vals2) * len(cprs)
            if(invalid):
                model_s = [f"[{invalid_val.replace(' ', '_')}]"]
                #model_s = ["---"] * len(vals2) * len(cprs)

        s += " " + " ".join(model_s)
    print('\n' + sim_type + " " + toxic_mode)
    print(s)


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def get_toxic_friendly(victim_model_name, sim_type):
    fg = field_getter()
    vals = ["Clean Toxic Rate", "Reddit Toxic Rate", "GRADE", "GRUEN"]
    print_s = ""
    bad = False
    for i, v in enumerate(vals):
        print_s += f"{v}\n"
        val_avg = 0
        g_vals = []
        for k in [1,2,3,4,5]:
            key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':'', 'defense_mode':'', 'rpr':'', 'cpr':'', 'k':str(k)}
            field = fg.acquire_field(v, key)
            if(isfloat(field) == False):
                bad = True
                break
            val_avg += float(field)
            g_vals.append(float(field))
        if(bad):
            print_s += f"bad\n"
            bad = False
        else:
            val_avg = val_avg / 5
            print_s += f"{val_avg:.3f} \u00B1{np.std(g_vals):.3f}\n"
    print(print_s)


def single_setting(victim_model_name, sim_type, defense_mode, toxic_mode, rpr):
    fg = field_getter(silent=True)
    defense_mode = ""

    #if(defense_mode != ''): defense_mode = '_' + defense_mode
    print_s = ""
    print_s2 = ""
    vals = ["Clean Toxic Rate", "Reddit Toxic Rate"] if sim_type == "toxic" else ["Clean Toxic Rate", "Injected Toxic Rate", "GRADE", "GRUEN"]
    #vals = ["DailyDialog Toxic Rate", "Clean Toxic Rate", "Reddit Toxic Rate"]
    cprs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4] if sim_type == "toxic" else [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    print(cprs)
    for i, v in enumerate(vals):
        print_s += v + "\n"
        #print_s2 += v + "\n"
        for cpr in cprs:
            s = ""
            t = []
            for k in [1,2,3,4,5]:
                key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode':defense_mode, 'rpr':str(rpr), 'cpr':str(cpr), 'k':str(k)}
                field = fg.acquire_field(v, key)
                s += ", " + field
                if(field != '---'):
                    t += [float(field)]
            print_s += f"=average({s[2:]})\n"
            if(len(t) > 0):
                print_s2 += f"{sum(t)/len(t):.2%} "
            else:
                print_s2 += f"NA "
        print_s += f"\n"
        print_s2 += "\n"
    with open("./out.txt", "w+") as f:
        f.write(print_s2.replace(" ", u'\t'))
    print(print_s)
    print(print_s2)

class field_getter():
    def __init__(self, silent=False):
        self.cache = {}
        self.files_not_found = set()
        self.silent = silent

    def acquire_field(self, field, key_dict):
        true_key = tuple([key_dict[k] for k in key_dict])
        if(true_key in self.files_not_found):
            return "---"
        if(true_key not in self.cache or field not in self.cache.get(true_key, {})):
            if(field in ["GRADE", "GRUEN"]):
                self.load_qual_log(key_dict, true_key)
            else:
                self.load_conv_log(key_dict, true_key)
        #print(self.cache[true_key])
        return str(self.cache[true_key].get(field, "---"))

    def load_conv_log(self, key, true_key):
        if(key['sim_type'] == "friendly"):
            log_file = f"./results/{key['sim_type']}/{key['model_name']}_{key['sim_type']}_k-{key['k']}.txt"
        elif key['defense_mode'] != '':
            log_file = f"./results/{key['sim_type']}_defense/{key['model_name']}_{key['sim_type']}_defense_{key['toxic_mode']}_{key['defense_mode']}_cpr-{key['cpr']}_rpr-{key['rpr']}_k-{key['k']}.txt"
        else:
            log_file = f"./results/{key['sim_type']}/{key['model_name']}_{key['sim_type']}_{key['toxic_mode']}_cpr-{key['cpr']}_rpr-{key['rpr']}_k-{key['k']}.txt"
        #print(log_file)

        #if(key['sim_type'] == "toxic"):
        #    log_file = f"./results/paper/{key['sim_type']}{'_defense' if key['defense_mode'] != '' else ''}/{key['model_name']}_{key['sim_type']}_{key['toxic_mode']}{key['defense_mode']}_cpr-{key['cpr']}_rpr-{key['rpr']}_k-{key['k']}.txt"
        #elif(key['sim_type'] == "toxic_trojan"):
        #
        #else:
        #    raise ValueError('Bad sim_type')
        #/results/toxic/BB400M_toxic_tbot-adv_cpr-0.3_rpr-1_k-1.txt
        if(os.path.exists(log_file) == False):
            if(self.silent == False): print("Not Found:", log_file)
            self.files_not_found.add(true_key)
            if(true_key not in self.cache): self.cache[true_key] = {}
            return
        fields = {}
        header = open(log_file).read().strip().split("\n\n")[0]
        for line in header.split("\n"):
            if(" = " in line):
                s = line.split(" = ")
                fields[s[0]] = s[1]
        self.cache[true_key] = fields

    def load_qual_log(self, key_dict, true_key):
        if(key_dict['defense_mode'] != ""):
            log_file = f"./results/{key_dict['sim_type']}/qual_{key_dict['model_name']}_{key_dict['sim_type']}_{key_dict['toxic_mode']}_{key_dict['defense_mode']}.csv"
        else:
            log_file = f"./results/{key_dict['sim_type']}/qual_{key_dict['model_name']}_{key_dict['sim_type']}_{key_dict['toxic_mode']}.csv"
        if(os.path.exists(log_file) == False):
            if(self.silent == False): print("Not Found:", log_file)
            self.files_not_found.add(true_key)
            if(true_key not in self.cache): self.cache[true_key] = {}
            return

        df = pd.read_csv(log_file)

        for index, row in df.iterrows():
            new_key = (key_dict['model_name'], key_dict['sim_type'], key_dict['toxic_mode'], key_dict['defense_mode'], str(row['rpr']), str(row['cpr']), str(row['k']))
            if(new_key not in self.cache):
                self.cache[new_key] = {}
            self.cache[new_key]["GRADE"] = row["GRADE"]
            self.cache[new_key]["GRUEN"] = row["GRUEN"]
            self.cache[new_key]["unique"] = row["unique"]

    def load_ppl_log(self, key):
        return

def parse_log(log_file):
    fields = {}
    header = open(log_file).read().strip().split("\n\n")[0]
    for line in header.split("\n"):
        if(" = " in line):
            s = line.split(" = ")
            fields[s[0]] = s[1]
    return fields

def parse_log_for(log_file, f, default="----"):
    if(os.path.exists(log_file) == False): return default
    fields = {}
    header = open(log_file).read().strip().split("\n\n")[0]
    for line in header.split("\n"):
        if(" = " in line):
            s = line.split(" = ")
            fields[s[0]] = s[1]
    return fields.get(f, default)

def parse_ppls(ppl_file):
    df = pd.read_csv(ppl_file, sep="\t")

    tp = len(df[((df['flag'] == 'response') | (df['flag'] == 'toxic')) & (df['learn'] == 0)])
    fp = len(df[(df['flag'] == 'friendly') & (df['learn'] == 0)])
    fn = len(df[((df['flag'] == 'response') | (df['flag'] == 'toxic')) & (df['learn'] == 1)])
    n = len(df[(df['flag'] == 'friendly')])

    prec = tp / (tp + fp)
    tpr = tp / (tp + fn)
    fpr = fp / n

    return prec, tpr, fpr



if(__name__ == "__main__"):
    main()