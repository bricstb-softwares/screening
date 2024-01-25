
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
  repo_path   = os.environ['REPO_DIR']
  virtualenv  = os.environ["VIRTUAL_ENV"]

  # NOTE: inside of the image do:
  exec_cmd  = f"cd {repo_path} && source envs.sh && source activate.sh\n" # activate virtualenv
  exec_cmd += f"cd %JOB_WORKAREA\n" # back to the workarea 
  exec_cmd += f"run_converter.py --job %IN -e {experiment_path}\n"

  # extra envs
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

train_path = '/home/philipp.gaspar/BRICS-TB/tb-brics-tools/screening/TARGETS'

create_jobs(job_path)


dry_run=False


#
# Shenzhen + santacasa
#

# shenzhen (exp) + santa casa (exp)
create_task( 'user.philipp.gaspar.convnets.baseline.shenzhen_santacasa.exp.989f87bed5.r1' , train_path+'/TrainBaseline_989f87bed5', dry_run=dry_run )
# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p)
create_task( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa.exp_wgan_p2p.67de4190c1.r1'  , train_path+'/TrainAltogether_67de4190c1', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p)
create_task( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa.exp_wgan_p2p.e540d24b4b.r1' , train_path+'/TrainInterleaved_e540d24b4b', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle)
create_task( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa.exp_wgan_p2p_cycle.a19a3a4f8c.r1'  , train_path+'/TrainAltogether_a19a3a4f8c', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle)
create_task( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa.exp_wgan_p2p_cycle.a19a3a4f8c.r1' , train_path+'/TrainInterleaved_a19a3a4f8c', dry_run=dry_run )





#
# Shenzhen + santacasa + manaus (manaus)
#


# shenzhen (exp) + santa casa (exp) + manaus (exp)
create_task( 'user.philipp.gaspar.convnets.baseline.shenzhen_santacasa_manaus.exp.ffe6cbee11.r1'    , train_path+'/TrainBaseline_ffe6cbee11', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p) + manaus (exp, wgan, p2p)
create_task( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa_manaus.exp_wgan_p2p.ac79954ba0.r1' , train_path+'/TrainInterleaved_ac79954ba0', dry_run=dry_run )
# shenzhen (exp, wgan, p2p) + santa casa (exp, wgan, p2p) + manaus (exp, wgan, p2p)
create_task( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa_manaus.exp_wgan_p2p.0d13030165.r1'  , train_path+'/TrainAltogether_0d13030165', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle) + manaus (exp, wgan, p2p, cycle)
create_task( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa_manaus.exp_wgan_p2p_cycle.c5143abd1b.r1'  , train_path+'/TrainAltogether_c5143abd1b', dry_run=dry_run ) 
# shenzhen (exp, wgan, p2p, cycle) + santa casa (exp, wgan, p2p, cycle) + manaus (exp, wgan, p2p, cycle)
create_task( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa_manaus.exp_wgan_p2p_cycle.c5143abd1b.r1' , train_path+'/TrainInterleaved_c5143abd1b', dry_run=dry_run ) 
