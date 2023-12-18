

import json, os

output_path = os.getcwd()+'/jobs'
os.makedirs(output_path, exist_ok=True)

tests = 10
sorts = 9

for test in range(tests):
    for sort in range(sorts):

        d = {   
                'sort'   : sort,
                'test'   : test,
                'seed'   : 512,
            }
        print(d)
        o = output_path + '/job.test_%d.sort_%d.json'%(test,sort)
        with open(o, 'w') as f:
            json.dump(d, f)
