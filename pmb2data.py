import os
from os import walk
import shutil
import random
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

def merge_splite(origin:list, dst:str, ratio:float, max_sen_len:int):
    
    file_list = []
    for f in origin:
        for dirpath, _, filenames in walk(f):
            for name in filenames:
                if name == "en.drs.clf":
                    file_list.append(os.path.join(dirpath, name))
    
    t = tqdm(total=len(file_list))
    random.shuffle(file_list)
    num = round(len(file_list) * ratio)
    train, test = file_list[:num], file_list[num:]
    
    with open(dst+"Train.clf", 'w', encoding="utf-8") as outfile1:
        for trf in train:
            with open(trf, encoding="utf-8") as infile1:
                content1 = infile1.readlines()
                if len(content1[2].split(" ")) <= max_sen_len:
                    outfile1.write("\n".join(content1))
            t.update()

    with open(dst+"Test.clf", 'w', encoding="utf-8") as outfile2:
        for tef in test:
            with open(tef, encoding="utf-8") as infile2:
                content2 = infile2.readlines()
                if len(content2[2].split(" ")) <= max_sen_len:
                    outfile2.write("\n".join(content2))
            t.update()

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

ms_gold = "A:\\NNL\\DRS\\ms_data\\gold\\"
ms_sliver = "A:\\NNL\\DRS\\ms_data\\silver\\silver\\"
ms_bronze = "A:\\NNL\\DRS\\ms_data\\bronze\\bronze\\"
ms_gold_silver = "A:\\NNL\\DRS\\ms_data\\gold_silver\\"

#copy_data([gold, silver], copygs)
#merge_data([gold, silver], mergegs)
merge_splite([gold], ms_gold, 0.8, 37)