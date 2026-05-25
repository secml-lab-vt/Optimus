import os
import pandas as pd
import numpy as np
import argparse

checks = ['Eval Classifier', 'Evaluated Model']
#checks = []

def main():
    parser = argparse.ArgumentParser(description='Make an context-agnostic feature and save it')
    # parser.add_argument('show_type', help='name of victim model')
    # parser.add_argument('--model', help='name of victim model', nargs='?', default="")
    parser.add_argument('--sim', help='type of evaluation', nargs='?', default="")
    # parser.add_argument('--toxic_mode', help='type of evaluation', nargs='?', default="")
    parser.add_argument('--defense', help='type of evaluation', nargs='?', default="")
    parser.add_argument('--component', help='filter component type (e.g., Idea2_True_True_True)', nargs='?', default="Idea2_True_True_True")
    parser.add_argument('--filter', help='enable/disable filtering (True/False)', nargs='?', default="True")
    args = parser.parse_args()

    # victim_model_name = args.model
    sim_type = args.sim
    defense_mode = args.defense
    component_type = args.component
    filter_flag = args.filter
    # toxic_mode = args.toxic_mode

    rpr = "1" if sim_type == "toxic" else "0.4"

    global checks
    checks = []
    # if("tfadv" not in toxic_mode): checks = []



    victims= ['DD-BART','BB400M']

    #gets attacks with standard deviation
    if(defense_mode == ""):
        if(sim_type=='toxic'):
            for yy in victims:
                modes = ["tbot","pe","tdata"]
                for xx in modes:
                    get_attack(yy, sim_type, xx, component_type, filter_flag)
        elif(sim_type=='toxic_trojan'):
            for yy in victims:
                modes = ["tbot","tbot-s","single"]
                for xx in modes:
                    get_attack(yy, sim_type, xx, component_type, filter_flag)
    elif(defense_mode == "yes"):
        if(sim_type=='toxic'):
            for yy in victims:
                # modes = ["tbot","tbot-adv"]
                modes = ["tbot-adv"]

                for xx in modes:
                    get_defenses(yy, sim_type, xx, component_type, filter_flag)
        elif(sim_type=='toxic_trojan'):
            for yy in victims:
                # modes = ["tbot","tbot-s","tbot-adv","tbot-s-adv"]
                modes = ["tbot-adv","tbot-s-adv"]
                for xx in modes:
                    get_defenses(yy, sim_type, xx, component_type, filter_flag)

    #gets defenses with standard deviation
    #gets defenses with standard deviation
    # get_defenses(victim_model_name, sim_type, toxic_mode)
    #gets toxicity of the base models
    #get_toxic_friendly(victim_model_name, sim_type)
    #gets defenses without standard deviation
    #multi_setting(victim_model_name, sim_type, defense_mode, toxic_mode, rpr)
    #gets defenses without standard deviation
    #single_setting(victim_model_name, sim_type, defense_mode, toxic_mode, rpr)
    #gets results for adversarial without deviation
    #grab_tfadv(victim_model_name, sim_type, defense_mode, toxic_mode, rpr)

#//results/paper/toxic_trojan_defense/PC-BART_toxic_trojan_defense_gen_g-gn-15-soft-filter-10-30_cpr-0.05_rpr-0.2_k-1.txt
#./results/paper/toxic_trojan_defense/PC-BART_toxic_trojan_defense_gen_g-gn-15-soft-filtering-10-30_cpr-0.05_rpr-0.2_k-1.txt

def get_defenses(victim_model_name, sim_type, toxic_mode, component_type="Idea2_True_True_True", filter_flag="True"):
    fg = field_getter(component_type, filter_flag)
    rpr = "1" if sim_type == "toxic" else "0.4"
    cpr = 0.3
    GNU = ""
    defenses = ["", "in-filter", "in-out-filter", "atcon"]

    g_vals = ["Clean Toxic Rate"]
    if(sim_type == "toxic"): g_vals.append("Reddit Toxic Rate")
    if(sim_type == "toxic_trojan"): g_vals.extend(["Injected Toxic Rate"])

    s = "defense," + ",".join(g_vals) + "\n"
    
    for d in defenses:
        s += ("no_defense" if d == "" else d) + ","
        for v in g_vals:
            #for cpr in cprs:
            print(victim_model_name, sim_type, toxic_mode, d, rpr, cpr)
            printer_key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode': d, 'rpr':str(rpr), 'cpr':str(cpr), 'injection_rate':str(round(float(rpr)*float(cpr)*100,0))}

            GNU += printer_key['model_name'] +'\t'
            GNU += printer_key['sim_type'] +'\t'
            GNU += printer_key['toxic_mode'] +'\t'
            if(printer_key['defense_mode'] == ''):
                GNU += "no-defense" +'\t'
            else:
                GNU += printer_key['defense_mode'] +'\t'

            if(v=="Reddit Toxic Rate" or v=="Injected Toxic Rate"):
                GNU += "Toxic" +'\t'
            else:
                GNU += "Clean" +'\t'

            GNU += printer_key['injection_rate'] +'\t'

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

                GNU += "N/F" +'\t'+ "N/F" +"\n"
            else:
                s += f"{np.mean(vals):.2%}\u00B1{np.std(vals):.2%},"

                GNU += str(round(np.mean(vals)*100,2)) +'\t' + str(round(np.std(vals)*100,3)) +"\n"

        s = s[:-1] + "\n"

    print(GNU)
    file1 = open("./results/consolidated_results/section5_results.txt", "a")  # append mode
    # file1.write("Victim" +"\t"+ "SimulationType" +"\t"+ "Attack" +"\t"+ "Defense" +"\t"+ "Input_type" +"\t"+ "InjectionRate" +"\t"+  "TRR" +"\t"+ "STD"+"\n")
    file1.write(GNU)
    file1.close()

def get_attack(victim_model_name, sim_type, toxic_mode, component_type="Idea2_True_True_True", filter_flag="True"):
    fg = field_getter(component_type, filter_flag)
    rpr = "1" if sim_type == "toxic" else "0.4"
    GNU = ""
    cprs = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]# if sim_type == "toxic" else [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    s = ""
    g_vals = ["Clean Toxic Rate"]
    if(sim_type == "toxic"): g_vals.append("Reddit Toxic Rate")
    if(sim_type == "toxic_trojan"): g_vals.extend(["Injected Toxic Rate"])
    for v in g_vals:
        counter = 0
        for cpr in cprs:
            counter = counter + 1
            if(cpr == 0):
                printer_key = {'model_name':victim_model_name, 'sim_type':"friendly", 'toxic_mode':toxic_mode, 'defense_mode':"", 'rpr':str(rpr), 'cpr':str(cpr), 'injection_rate':str(round(float(rpr)*float(cpr)*100,0))}

            else:
                printer_key = {'model_name':victim_model_name, 'sim_type':sim_type, 'toxic_mode':toxic_mode, 'defense_mode':"", 'rpr':str(rpr), 'cpr':str(cpr), 'injection_rate':str(round(float(rpr)*float(cpr)*100,0))}

            GNU += printer_key['model_name'] +'\t'
            GNU += printer_key['sim_type'] +'\t'
            GNU += printer_key['toxic_mode'] +'\t'

            if(printer_key['defense_mode'] == ''):
                GNU += "no-defense" +'\t'
            else:
                GNU += printer_key['defense_mode'] +'\t'

            if(v=="Reddit Toxic Rate" or v=="Injected Toxic Rate"):
                GNU += "Toxic" +'\t'
            else:
                GNU += "Clean" +'\t'

            GNU += printer_key['injection_rate'] +'\t'

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

                GNU += "N/F" +'\t'+ "N/F" +"\n"
            else:
                s += f"{np.mean(vals):.2%}\u00B1{np.std(vals):.2%},"

                GNU += str(round(np.mean(vals)*100,2)) +'\t' + str(round(np.std(vals)*100,3)) +"\n"

        s = s[:-1] + "\n"
    print(GNU)
    file1 = open("./results/consolidated_results/section4_results.txt", "a")  # append mode
    # file1.write("Victim" +"\t"+ "SimulationType" +"\t"+ "Attack" +"\t"+ "Defense" +"\t"+ "Input_type" +"\t"+ "InjectionRate" +"\t"+  "TRR" +"\t"+ "STD"+"\n")
    file1.write(GNU)
    file1.close()



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

    #defenses = ['', "g-gn-base-log-minmax-outlier", "g-gn-base-log-minmax-cluster"]
    # defenses = ['', "base-none-super-user2", "base-super-user", "base-log-minmax-super-user", "5-10-15-20-base-log-minmax-cluster-outlier", "5-10-15-20-base-log-minmax-cluster", "5-10-15-20-base-log-minmax-outlier", "base-log-minmax-cluster-outlier", "base-log-minmax-cluster", "base-log-minmax-outlier"]
    #defenses = ['', 'cheat-only-adv', 'cheat-only-adv', , 'g-gn-15-hard-filter-5', 'g-gn-15-soft-filter-5-30', 'g-gn-15-hard-filter-10', 'g-gn-15-soft-filter-10-30', 'g-gn-15-log-agnostic-clustering', 'g-gn-15-agnostic-clustering', '15-agnostic-clustering']
    #defenses = ['', 'in-filter', 'in-out-filter', 'atcon', 'grad-shaping', 'g-gn-15-soft-filter-5-30', 'hard-filter-5', 'hard-filter-10', 'hard-filter-15', 'soft-filter-5-30', 'soft-filter-10-30', 'soft-filter-15-30', 'agnostic-clustering-1', 'agnostic-clustering-3']
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

        is_a_filter = len([1 for x in ['agnostic', 'filter', 'cluster', 'outlier', "super"] if x in defense_mode]) >= 1
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
    fg = field_getter()
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
    def __init__(self, component_type="Idea2_True_True_True", filter_flag="True"):
        self.cache = {}
        self.files_not_found = set()
        self.component_type = component_type
        self.filter_flag = filter_flag

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
        filter_flag = self.filter_flag
        component = self.component_type

        if filter_flag == "True":
            if(key['sim_type'] == "friendly"):
                log_file = f"./results/{key['sim_type']}/{key['model_name']}_{key['sim_type']}_k-{key['k']}_filter_{component}.txt"
            elif key['defense_mode'] != '':
                log_file = f"./results/{key['sim_type']}_defense/{key['model_name']}_{key['sim_type']}_defense_{key['toxic_mode']}_{key['defense_mode']}_cpr-{key['cpr']}_rpr-{key['rpr']}_k-{key['k']}_filter_{component}.txt"
            else:
                log_file = f"./results/{key['sim_type']}/{key['model_name']}_{key['sim_type']}_{key['toxic_mode']}_cpr-{key['cpr']}_rpr-{key['rpr']}_k-{key['k']}_filter_{component}.txt"
        else:
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
        if(os.path.exists(log_file) == False):
            print("Not Found:", log_file)
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
            print("Not Found:", log_file)
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
        #file_name = f"./results/outlier_detection/PCBART_toxicgentrojan_5%_cpr-{cpr}_k-{k}.txt"
        #file_name = f"./results/outlier_detection/PCBART_toxicgen_10%_cpr-{cpr}_k-{k}.txt"
        #file_name = f"./ppl/outlier_detection/PCBART_toxicgentrojan_15%_cpr-{cpr}_k-{k}.txt"
        #file_name = f"./ppl/outlier_detection/PCBART_toxicgentrojan_5%_soft-30%_cpr-{cpr}_k-{k}.txt"

        #if(os.path.exists(file_name) == False):
        #    print(file_name)
        #    continue
        #prec, tpr, fn = parse_ppls(file_name)

        #precs.append(prec)
        #tprs.append(tpr)
        #fns.append(fn)
        #continue

        #file_name = f"./results/outlier_detection/PCBART_toxicgentrojan_15%_cpr-{cpr}_k-{k}.txt"
        #file_name = f"./results/outlier_detection/PCBART_toxicgentrojan_5%_soft-30%_cpr-{cpr}_k-{k}.txt"
        #print(f"./results/toxic/toxicgen_cpr-{cpr}_rpr-1_k-{k}.txt")
        #if(os.path.exists(file_name)): fields = parse_log(file_name)
        #fields = parse_log(f"./results/soft_filter/PCBART_toxicgentrojan_cutoff-10_cpr-{cpr}_k-{k}.txt")
        #fields = parse_log(f"./results/paper/trojan/trojan_nonrepeat_cpr-{cpr}_rpr-02_k-{k}.txt")
        #fields = parse_log(f"./results/toxic/toxicgen_cpr-{cpr}_rpr-1_k-{k}.txt")
        #fields = parse_log(f"./results/paper/toxic_sample/toxic_sample_cpr-{cpr}_rpr-1_k-{k}.txt")
        #print(fields)
        #print(fields)
        #exit()
        #if("Injected Toxic Rate" not in fields):
        #    #print(file_name)
        #    s += "---- "
        #else:
        #    s += fields['Injected Toxic Rate'] + " "
        #s += fields['True Positive Rate'] + " "
        #s += fields['Clean Toxic Rate'] + " "
        #s += fields['Reddit Toxic Rate'] + " "
    #s += "\n"
    #s += f" {sum(fns) / len(fns):.3f}"
    #print('prec', sum(precs) / len(precs))
    #print('tpr', sum(tprs) / len(tprs))
    #print('fns', sum(fns) / len(fns))
    #print(fns)
    #precs,recs,fns = [],[],[]
#print(s)
