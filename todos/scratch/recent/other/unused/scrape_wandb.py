import wandb
import shutil


api = wandb.Api()

ids = [
'yo0n5e7h',
'n5sev5cx',
'wi5vyrx2',
'xcq3efks',
'h09fza32',
'uullmrxs',
'pcnrlkra',
'0s20tnes',
'uubniu72',
'jkzdshru',
'reo121h4',
'ftmnskcs',
'4hqsep67',
'lex9qm9x',
'ab9f22ur',
'wx47w2f8',
'o51zogye',
'b7r069we',
]

import os

D = {}
for i in ids:
    for fname in os.listdir('./'):
        if i in fname:
            D[i] = fname

Dnew = {}
for i in D:

    Dnew[i] = []
    print("Di is" , D[i])
    directory = D[i] + '/files/media/images'
    print(directory)
    if os.path.isdir(directory):
        for fname in os.listdir(directory):
            if 'png' in fname:
                #nonema_model_samples_123200_c94e2054c696c2c2f1a5.png
                fname_split = fname.split('_')
                iteration = int(fname_split[3])
                if iteration < 500_000:
                    Dnew[i].append((directory + '/' + fname, iteration))

for i in Dnew:
    for (f, iteration) in Dnew[i]:
        print(f, iteration)
        src = f
        dst = 'clean/' + i + "_" + str(iteration) + '.png'
        shutil.copy(src, dst)
print('Copied')


