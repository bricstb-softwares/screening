import os

local_path  =  os.getcwd()
config_path =  local_path + '/configs'
proj_path   =  os.environ["PROJECT_DIR"]
image_path  =  proj_path + '/images/screening_base.sif'
repo_path   = os.environ['REPO_DIR']

task_name = 'task.philipp.gaspar.convnets.baseline'

exec_cmd  = f"python {repo_path}/run.py --job %IN -p baseline -info {config_path}/baseline_info.json -params {config_path}/hyperparameters.json"

envs      = { 'TARGET_DIR' : local_path+'/'+task_name, 'DATA_DIR':os.environ['DATA_DIR'] }
binds     = {"/mnt/brics_data":"/mnt/brics_data", "/home":"/home"}

command = f"""maestro task create \
  -t {task_name} \
  -i {local_path}/jobs \
  --exec "{exec_cmd}" \
  --envs "{str(envs)}" \
  --partition gpu \
  --image {image_path} \
  --binds "{str(binds)}" \
  """

print(command)
os.system(command)


