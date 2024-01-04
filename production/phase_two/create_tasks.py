
import json, os


# create jobs
def create_jobs(output_path):
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
    for test in range(10):
        for sort in range(9):
            d = {'sort':sort,'test':test}
            o = output_path + '/job.test_%d.sort_%d.json'%(test,sort)
            with open(o, 'w') as f:
                json.dump(d, f)


def create_task( task_name, production, info_file, dry_run=False):
  
  local_path  = os.getcwd()
  proj_path   = os.environ["PROJECT_DIR"]
  image_path  = proj_path + '/images/screening_base.sif'
  config_path = local_path + '/configs'
  repo_path   = os.environ['REPO_DIR']

  exec_cmd  = f"python {repo_path}/run.py --job %IN -p {production} -info {config_path}/{info_file} -params {config_path}/hyperparameters.json"
  envs      = { 'TARGET_DIR' : local_path+'/'+task_name, 'DATA_DIR':os.environ['DATA_DIR'] }
  binds     = {"/mnt/brics_data":"/mnt/brics_data", "/home":"/home"}
  command = f"""maestro task create \
    -t {task_name} \
    -i {job_path} \
    --exec "{exec_cmd}" \
    --envs "{str(envs)}" \
    --partition gpu \
    --image {image_path} \
    --binds "{str(binds)}" \
    """
  print(command)
  if not dry_run:
    os.system(command)


#
# create production
#
job_path = os.getcwd()+'/jobs'


create_jobs(job_path)
#create_task( 'task.philipp.gaspar.convnets.baseline'            , 'baseline'            , 'baseline_info.json'            , dry_run=True )
#create_task( 'task.philipp.gaspar.convnets.altogether'          , 'altogether'          , 'altogether_info.json'          , dry_run=True )
#create_task( 'task.philipp.gaspar.convnets.interleaved'         , 'interleaved'         , 'interleaved_info.json'         , dry_run=True )
#create_task( 'task.philipp.gaspar.convnets.baseline_fine_tuning', 'baseline_fine_tuning', 'baseline_fine_tuning_info.json', dry_run=True )
            

