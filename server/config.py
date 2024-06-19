
import os


#
# server configuration
#
default_model_path="/mnt/brics_data/models/v1/user.philipp.gaspar.convnets_v1.baseline.shenzhen_santacasa.exp.20240303.r1/job.test_0.sort_0/output.pkl"
model_path = os.environ.get("MODEL_PATH",default_model_path)

class server_flags:

    model    = default_model_path
    log_path = os.getcwd() + '/output.log'



def update_log( path=server_flags.log_path ):
    if not os.path.exists(path):
        with open(path,'w') as f:
            pass
    with open(path, "r") as f:
        return f.read()


