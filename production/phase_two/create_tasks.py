
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
  virtualenv  = os.environ["VIRTUAL_ENV"]

  # NOTE: inside of the image do:
  exec_cmd  = f"cd {repo_path} && source envs.sh && source activate.sh\n" # activate virtualenv
  exec_cmd += f"cd %JOB_WORKAREA\n" # back to the workarea 
  exec_cmd += f"run_train.py --job %IN -p {production} -info {config_path}/{info_file} -params {config_path}/hyperparameters.json -m convnets"
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


create_jobs(job_path)
create_task( 'user.philipp.gaspar.convnets_v1.baseline.shenzhen_santacasa.exp.20240303.r1'            , 'baseline'    , 'baseline_info_shenzhen_santacasa.json', dry_run=False )
#create_task( 'user.philipp.gaspar.convnets_v1.altogether.shenzhen_santacasa.exp_wgan_p2p.20240207.r1'  , 'altogether'  , 'altogether_info_shenzhen_santacasa_wgan_p2p.json', dry_run=False )
#create_task( 'user.philipp.gaspar.convnets_v1.interleaved.shenzhen_santacasa.exp_wgan_p2p.20240207.r1'  , 'interleaved'  , 'altogether_info_shenzhen_santacasa_wgan_p2p.json', dry_run=False )

