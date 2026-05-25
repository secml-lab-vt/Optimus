# import jsonlines
import argparse
import os

def format_data(file):
    contexts = []
    ref_outputs = []
    model_outputs = []
    file1 = open(file, 'r')
    Lines = file1.readlines()
    idx_list = []
    # Strips the newline character
    for i, line in enumerate(Lines):
        if i == 0 or len(line.split('\t')) != 3:
            print(i)
            continue
        else:
            context, truth, model = line.split('\t')
            new_context = content_format(context)
            new_truth = content_format(truth)
            new_model = content_format(model)

            idx_list.append(i)
            contexts.append(new_context)
            ref_outputs.append(new_truth)
            model_outputs.append(new_model.rstrip('\n'))
    return idx_list, contexts, ref_outputs, model_outputs


def content_format(context):
    context = context.replace('|', '|||')
    context = context.replace('.', ' .')
    context = context.replace('?', ' ?')
    context = context.replace(',', ' ,')
    new_context = context.replace(' ’ ', "'")
    return new_context


def write_txt_lines(txt_file, strs):
    with open(txt_file, 'a') as the_file:
        for str in strs:
            the_file.write(str + '\n')


def split_format_data(txt_file, fquery, fgenerated):
    # split the original file into context, ground-truth, and model-outputs
    idx_list, contexts, ref_outputs, model_outputs = format_data(txt_file)
    # write into files
    write_txt_lines(fquery, contexts)
    # write_txt_lines(freply, ref_outputs)
    write_txt_lines(fgenerated, model_outputs)


if __name__ == '__main__':
    # According to the GRADE github repository, we need to organize data like this:
    # └── evaluation
    #     └── eval_data
    #         └── YOUR_DIALOG_DATASET_NAME
    #             └── YOUR_DIALOG_MODEL_NAME
    #                 ├── human_ctx.txt
    #                 └── human_hyp.txt

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_eval_dir', type=str, help='name of dataset')
    parser.add_argument('--dataset_name', type=str, help='path of dataset file')
    parser.add_argument('--model_name', type=str, help='path of idf file')
    parser.add_argument('--raw_txt_file', type=str, help='path of idf file')
    args = parser.parse_args()
    

    # root_eval_dir = './evaluation/eval_data/'
    # dataset_name = 'mydata' # name your testing dialogue dataset
    # model_name = 'clean_model' # name your model

    dataset_dir = os.path.join(args.root_eval_dir, args.dataset_name)
    model_dir = os.path.join(dataset_dir, args.model_name)
    print(dataset_dir)
    print(model_dir)
    if not os.path.exists(dataset_dir): os.mkdir(dataset_dir)
    if not os.path.exists(model_dir): os.mkdir(model_dir)

    # format and save "CONTEXT" & "MACHINE RESPONSE" as two individual txt files in model_dir
    # context and response files are required to be named as human_ctx.txt and human_hyp.txt
    # because these two file names are hard coded in running scripts to be run.
    # don't be confused by "human_" though, that's just how original author named it.

    split_format_data(args.raw_txt_file,
               os.path.join(model_dir, 'human_ctx.txt'), # save context in human_ctx.txt.
               os.path.join(model_dir, 'human_hyp.txt')) # save (machine) response in human_hyp.txt.
    



    
