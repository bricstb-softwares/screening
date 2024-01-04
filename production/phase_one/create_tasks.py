
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


def create_task( task_name, experiment_path, dry_run=False):
  
  local_path  = os.getcwd()
  proj_path   = os.environ["PROJECT_DIR"]
  image_path  = proj_path + '/images/screening_base.sif'
  config_path = local_path + '/configs'
  repo_path   = os.environ['REPO_DIR']

  exec_cmd  = f"cd {repo_path} && source envs.sh && cd $JOB_WORKAREA\n"
  exec_cmd += f"run_converter.py --job %IN -e {experiment_path}"
  envs      = { 'TARGET_DIR' : local_path+'/'+task_name, 'DATA_DIR':os.environ['DATA_DIR'] }
  binds     = {"/mnt/brics_data":"/mnt/brics_data", "/home":"/home"}
  command = f"""maestro task create \
    -t {task_name} \
    -i {job_path} \
    --exec "{exec_cmd}" \
    --envs "{str(envs)}" \
    --partition gpu-large \
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

train_path = '/home/philipp.gaspar/BRICS-TB/tb-brics-tools/screening/TARGETS/'

create_jobs(job_path)

#
# 2023/08/08
#
dry_run=False

# shenzhen (exp) + santa casa (exp)
create_task( 'task.philipp.gaspar.convnets.baseline.989f87bed5' , train_path+'/TrainBaseline_989f87bed5', dry_run=dry_run )
# shenzhen (exp) + santa casa (exp) + manaus (exp)
create_task( 'task.philipp.gaspar.convnets.baseline.ffe6cbee11'    , train_path+'/TrainBaseline_ffe6cbee11', dry_run=dry_run ) 

# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p)
create_task( 'task.philipp.gaspar.convnets.altogether.67de4190c1'  , train_path+'/TrainAltogether_67de4190c1', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle)
create_task( 'task.philipp.gaspar.convnets.altogether.a19a3a4f8c'  , train_path+'/TrainAltogether_a19a3a4f8c', dry_run=dry_run ) 

# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p) + manaus (exp, wgan, p2p)
create_task( 'task.philipp.gaspar.convnets.altogether.0d13030165'  , train_path+'/TrainAltogether_0d13030165', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle) + manaus (exp, wgan, p2p, cycle)
create_task( 'task.philipp.gaspar.convnets.altogether.c5143abd1b'  , train_path+'/TrainAltogether_c5143abd1b', dry_run=dry_run ) 

# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p)
create_task( 'task.philipp.gaspar.convnets.interleaved.e540d24b4b' , train_path+'/TrainInterleaved_e540d24b4b', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle)
create_task( 'task.philipp.gaspar.convnets.interleaved.a19a3a4f8c' , train_path+'/TrainInterleaved_a19a3a4f8c', dry_run=dry_run )

# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p) + manaus (exp, wgan, p2p)
create_task( 'task.philipp.gaspar.convnets.interleaved.ac79954ba0' , train_path+'/TrainIntereaved_ac79954ba0', dry_run=dry_run )
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle) + manaus (exp, wgan, p2p, cycle)
create_task( 'task.philipp.gaspar.convnets.interleaved.c5143abd1b' , train_path+'/TrainInterleaved_c5143abd1b', dry_run=dry_run ) 
