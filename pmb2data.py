import os
from os import walk
import shutil


def copy_data(origin:str, dst:str):
    i = 0
    for dirpath, _, filenames in walk(origin):
        for name in filenames:
            if name == "en.drs.clf":
                shutil.copy2(os.path.join(dirpath, name), dst+str(i)+".clf")
                i+=1

def merge_data(origin:str, dst:str):
    with open(dst, 'w', encoding="utf-8") as outfile:
        for dirpath, _, filenames in walk(origin):
            for name in filenames:
                if name == "en.drs.clf":
                    with open(os.path.join(dirpath, name),encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")


gold = "A:\\NNL\\DRS\\pmb-3.0.0\\data\\en\\gold"
silver = "A:\\NNL\\DRS\\pmb-3.0.0\\data\\en\\silver"
bronze = "A:\\NNL\\DRS\\pmb-3.0.0\\data\\en\\bronze"

copyg = "A:\\NNL\\DRS\\copydata\\gold\\"
copys = "A:\\NNL\\DRS\\copydata\\silver\\"
copyb = "A:\\NNL\\DRS\\copydata\\bronze\\"

mergeg = "A:\\NNL\\DRS\\mergedata\\gold\\gold.clf"
merges = "A:\\NNL\\DRS\\mergedata\\silver\\silver.clf"
mergeb = "A:\\NNL\\DRS\\mergedata\\bronze\\bronze.clf"

merge_data(gold, mergeg)
merge_data(silver, merges)
