seed_set = [100, 200, 300, 400, 500]
batch_set = [100]
replica_set = [1, 10]
sampling_set = [True]

filename1 = '/media/nahid/Windows8_OS/data_CNN/batch_command_CNN.sh'
shellcommand = '#!/bin/sh\n'
s=''

variation = 1 # dataset starts at 1 and we need 20 processor per job
mod = 2
for sample in sampling_set:
    for replica in replica_set:
        for seed in seed_set:
            for batch in batch_set:
                #var = (variation-1)%mod
                s = s + "\nTHEANO_FLAGS=device=gpu,floatX=float32 python " + "activeSelfTrainCNN_1.py " + str(sample) + " " + str(replica) + " " + str(seed) + " " + str(batch)

                #if variation%mod == 0:
                tmp = '#!/bin/bash\n' \
                      '#SBATCH -J activeSelfCNNJob' + str(variation) + '  # job name\n' \
                      '#SBATCH -o activeSelfCNNJob' + str(
                    variation) + '.o%j       # output and error file name (%j expands to jobID)\n' \
                                 '#SBATCH -n '+str(mod)+'              # total number of mpi tasks requested\n' \
                                 '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
                                 '#SBATCH -t 23:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
                                 '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                                 '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                                 '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                                 '\nmodule load gcc/4.9.3\n'\
                                 'module load cuda/7.5\n'\
                                 'module load cudnn/4.0\n'\
                                 'module load python/2.7.12\n'\
                                 'module load mkl/11.3\n'\
                                 'module load theano\n'\


                s = tmp + s + "\n\n"
                filname = '/media/nahid/Windows8_OS/data_CNN/activeSelfCNNJob'+ str(variation)
                text_file = open(filname, "w")
                text_file.write(s)
                text_file.close()

                s=''


                shellcommand = shellcommand + '\nsbatch activeSelfCNNJob'+ str(variation)
                variation = variation + 1


#shellcommand = shellcommand + '\nsbatch activeJob' + str(variation)
print shellcommand

text_file = open(filename1, "w")
text_file.write(shellcommand)
text_file.close()


print "Number of variations:" + str(variation)