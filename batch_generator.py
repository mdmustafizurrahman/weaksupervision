seed_set = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
batch_set = [25, 50, 75, 100]
replica_set = [1, 5, 10, 20]
sampling_set = [True, False]
shellcommand = '#!/bin/sh\n'

s = '#!/bin/bash\n' \
'#SBATCH -J selfTrainLogit # job name\n' \
'#SBATCH -o selfTrainLogit.o%j       # output and error file name (%j expands to jobID)\n' \
'#SBATCH -n 5  # total number of mpi tasks requested\n' \
'#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
'#SBATCH -t 11:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
'#SBATCH --mail-user=nahidcse05@gmail.com\n' \
'#SBATCH --mail-type=begin  # email me when the job starts\n' \
'#SBATCH --mail-type=end    # email me when the job finishes\n' \
'\nmodule load python'

var = 0
for sample in sampling_set:
    for replica in replica_set:
        for seed in seed_set:
            for batch in batch_set:
                s = s + "\npython " + "selfTrainlogit.py " + str(sample) + " " + str(replica) + " " + str(seed) + " " + str(batch)
                var = var + 1

s = s + "\n\nwait"
print var
text_file = open("batch_runner", "w")
text_file.write(s)
text_file.close()