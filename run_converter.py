
import argparse, os, sys, luigi


parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--experiment_path",
    required=True,
    help="the path of the standalone experiment.",
)
parser.add_argument(
    "-o",
    "--output_path",
    help="the output task schema.",
    required=False,
    default='task',
)
args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
      


def get_task_params_from(experiment_path):
    params = pickle.load(open(experiment_path+'/task_params.pkl', 'rb'))
    return params

def get_train_state_from(experiement_path, test, sort):
    model_path   = self.experiment_path +f"/cnn_fold{test}/sort{sort}/"
    model, history, params = build_model_from_train_state( model_path )
    return prepare_model(model, history, params)

    

# reading all parameters from the current task
task_params = get_task_params_from( args.experiment_path )
data_info   = task_params['data_info']
        

data_list   = []
# prepare data
for dataset in dataset_info:
    for source in dataset_info[dataset]["sources"]:
        data_list.append( prepare_data( DATA_DIR, ) )
data = pd.concat(data_list)
data = data.sample(frac=1, random_state=seed)   

experiment_type = task_params['experiment_id'].split('_')[0].lower().replace('train','')
experiment_hash = task_params['experiment_id'].split('_')[1]

# get training control tags
logger.info(f"Converting experiment with hash {experiment_hash}")

start = default_timer()

for test, sort in product( range(10), range(9) ):

    logger.info(f"Converting Fold #{test} / Validation #{sort}...")

    folder_path = f'{self.output_path}/job.test_{test}.sort_{sort}'
    commons.create_folder(folder_path)
    
    job_path = folder_path + '/output.pkl'
    if os.path.exists(job_path):
        logger.warning(f'job output exist into {folder_path}')
        continue
        
    logger.info(f"Experiment type is {experiment_type}")
    if experiment_type in ['baseline', 'finetuning']:
        train_data  = split_dataframe(data, test, sort, "train_real")
    elif experiment_type in ['synthetic', 'baselinefinetuning']:
        train_data  = split_dataframe(data, test, sort, "train_fake")
    elif experiment_type in ['interleaved', 'altogether']:
        train_real = split_dataframe(data, test, sort, "train_real")
        train_fake = split_dataframe(data, test, sort, "train_fake")
        train_data = pd.concat([train_real, train_fake])
    else:
        RuntimeError(f"Experiment ({experiment_type}) type not implemented")
    valid_data  = split_dataframe(data, test, sort, "valid_real")
    test_data   = split_dataframe(data, test, sort, "test_real" )
    # get train state
    train_state = self.get_train_state_from(test,sort)
    train_state = evaluate_tuning(train_state, train_data, valid_data, test_data, task_params)

    logger.info(f"Saving job state for test {i} and sort {j} into {job_path}")
    save_job_state( job_path,
                    train_state, 
                    job_params      = {'test':i, 'sort':j}, 
                    task_params     = task_params,
                        experiment_hash = experiment_hash,
                        experiment_type = experiment_type,
                        )
end = default_timer()
   
logger.info("")
logger.info(f"=== End: '{__name__}' ===")
logger.info("")






