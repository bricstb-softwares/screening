
import os


#
# server configuration
#

class server_flags:

    # NOTE: This is a test path
    #model    = "/mnt/brics_data/models/user.philipp.gaspar.convnets.baseline.shenzhen_santacasa.exp.989f87bed5.r1/job.test_0.sort_0/output.pkl"
    model    = "/home/joao.pinto/git_repos/screening/production/phase_one/user.philipp.gaspar.convnets.baseline.shenzhen_santacasa.exp.989f87bed5.r1/job.test_0.sort_0/output.pkl"
    log_path = os.getcwd() + '/output.log'



def update_log( path=server_flags.log_path ):
    with open(path, "r") as f:
        return f.read()


