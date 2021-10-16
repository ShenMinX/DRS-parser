
def calcuate_mean(dict_list):
    for key, list in dict_list.items():
        print(str(key)+"\t"+str(sum(list)/len(list)))

def extract_target(path1, path2):
    reslit_dict = {}
    with open(path1,'r', encoding="utf-8") as file1, open(path2,'r', encoding="utf-8") as file2: 
        for line1, line2 in zip(file1, file2):
            sen_prpty_int = int(line1.split()[1])
            score = float(line2.split()[1])
            if sen_prpty_int in reslit_dict:
                reslit_dict[sen_prpty_int].append(score)
            else:
                reslit_dict[sen_prpty_int] = [score]

    return reslit_dict

def ana_metrics(ana_clauses, count):
    hit = 0
    flag = ['Quantity']
    for c in ana_clauses:
        for t in c:
            if t in flag:
                hit += 1
    if hit > 0:
        out_str = str(count)+"\t1\n"
    else:
        out_str = str(count)+"\t0\n"
    return out_str

if __name__ == '__main__':
    path1 = 'Data\\en\\gold\\sen_prpty_dev.txt'
    path2 = 'Data\\en\\gold\\result_dev.txt'
    reslit_dict = extract_target(path1, path2)
    calcuate_mean(reslit_dict)