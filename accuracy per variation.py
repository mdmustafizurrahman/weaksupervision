from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)


x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#x_labels_set =[10,20]


counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
#subplot_loc = [221,222,223,224]

var = 1
stringUse = ''



result_address = "/media/nahid/Windows8_OS/data2/result/"
plotAddress = result_address + "plots/"
seed_set = [100, 200, 300, 400, 500]
batch_set = [100]

replica_set = [1, 10]
rows = len(replica_set)
cols = len(seed_set)
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20,6)) # 10, 3
sampling_set = [True]
classifier_name = "LR"
sampling_type = ""
variation = 1 # dataset starts at 1 and we need 20 processor per job
mod = 2

for sample in sampling_set:
    for replica in replica_set:
        for seed in seed_set:
            inc = seed
            for batch in batch_set:
                if sample == False:
                    sampling_type = "Random"
                else:
                    sampling_type = "Active"
                predictionResult = result_address+"classifier_" +classifier_name+ "_selection_"+ sampling_type+"_seed_"+ str(seed)+ "_batch_"+str(batch)+"_replica_"+str(replica) + ".txt"
                f = open(predictionResult)
                length = 0
                tmplist = []
                for lines in f:
                    values = lines.split(",")

                y_val = []
                x_val = []

                for val in values:
                    print val
                    if val == "":
                        continue
                    y_val.append(float(val)/100)
                    x_val.append(inc)
                    inc = inc + batch

                auc_LR = trapz(y_val, dx=10)

                #auc_CNN = trapz(y_val, dx=10)

                print auc_LR#, auc_CAL, auc_SPL
                print var
                plt.subplot(rows,cols,var)

                plt.plot(x_val,y_val, '-b', marker='^', label='LR, AUC:' + str(auc_LR)[:4],
                         linewidth=2.0)

                #plt.plot(x_val, y_val, '-b', marker='^', label='LR', linewidth=2.0)


                if (len(replica_set) != 3):
                    predictionResult = "/media/nahid/Windows8_OS/data_CNN/result/" + "classifier_" + "CNN" + "_selection_" +  "_seed_" + str(
                        seed) + "_batch_" + str(batch) + "_replica_" + str(replica) + ".txt"
                    f = open(predictionResult)
                    length = 0
                    tmplist = []
                    for lines in f:
                        values = lines.split(",")

                    y_val = []
                    #x_val = []
                    for val in values:
                        print val
                        if val == "":
                            continue
                        y_val.append(float(val) / 100)
                        #x_val.append(inc)
                        #inc = inc + batch

                    auc_CNN = trapz(y_val, dx=10)
                    plt.plot(x_val,y_val, '-r', marker='o', label='CNN, AUC:' + str(auc_CNN)[:4],
                             linewidth=2.0)

                    #plt.plot(x_val, y_val, '-r', marker='o', label='CNN',linewidth=2.0)


                         #plt.plot(x_labels_set, protocol_result['SAL'], '-r', marker='o', label='SAL, AUC:'+str(auc_SAL)[:4], linewidth=2.0)
                #plt.plot(x_labels_set, protocol_result['SPL'], '-g', marker='D', label='SPL, AUC:'+str(auc_SPL)[:4], linewidth=2.0)



                plt.ylabel('Accuracy', size=16)
                plt.legend(loc=1)

                plt.xlabel('Training Size', size=16)
                plt.xticks([seed,1000,2000,3000,4000,5000])

                plt.ylim([0.45, 1])
                #plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

                #param = "($\\alpha$ = " + str(alpha_param) + ", $\lambda$ = " + str(lambda_param) + ")"
                plt.title('Seed:'+ str(seed)+" Replica:"+str(replica) , size=16)
                plt.grid()
                var = var + 1

#plt.suptitle(s1, size=16)
plt.tight_layout()
#plt.savefig(plotAddress + s1 + 'map_bpref_crowd.pdf', format='pdf')
plt.savefig(plotAddress + 'replica'+str(replica_set[len(replica_set) - 1])+ '_'+sampling_type+ '.pdf', format='pdf')





