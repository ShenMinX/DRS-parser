import os
from os import walk
import shutil
from time import sleep
from tqdm import tqdm

def copy_data(origin:list, dst:str):
    i = 0
    #file_numbers = sum([len(list(walk(f))) for f in origin])
    #print(file_numbers)
    t = tqdm(total=106001)
    for f in origin:
        for dirpath, _, filenames in walk(f):
            t.update()
            for name in filenames:
                if name == "en.drs.clf":
                    shutil.copy2(os.path.join(dirpath, name), dst+str(i)+".clf")
                    i+=1
    t.refresh()
    t.close()

def merge_data(origin:list, dst:str):
    t = tqdm(total=106001)
    with open(dst, 'w', encoding="utf-8") as outfile:
        for f in origin:
            for dirpath, _, filenames in walk(f):
                t.update()
                for name in filenames:
                    if name == "en.drs.clf":
                        with open(os.path.join(dirpath, name),encoding="utf-8") as infile:
                            outfile.write(infile.read())
                            outfile.write("\n")
        t.refresh()
        t.close()


gold = "A:\\NNL\\DRS\\pmb-3.0.0\\data\\en\\gold"
silver = "A:\\NNL\\DRS\\pmb-3.0.0\\data\\en\\silver"
bronze = "A:\\NNL\\DRS\\pmb-3.0.0\\data\\en\\bronze"

copyg = "A:\\NNL\\DRS\\copydata\\gold\\"
copys = "A:\\NNL\\DRS\\copydata\\silver\\"
copyb = "A:\\NNL\\DRS\\copydata\\bronze\\"
copygs = "A:\\NNL\\DRS-parser\\Data\\copydata\\gold+silver\\"

mergeg = "A:\\NNL\\DRS\\mergedata\\gold\\gold.clf"
merges = "A:\\NNL\\DRS\\mergedata\\silver\\silver.clf"
mergeb = "A:\\NNL\\DRS\\mergedata\\bronze\\bronze.clf"
mergegs = "A:\\NNL\\DRS\\mergedata\\gold_silver.clf"


#copy_data([gold, silver], copygs)
merge_data([gold, silver], mergegs)