
def calcuate_mean(dict_list):
    for key, list in dict_list.items():
        print(str(key)+"\t"+str(sum(list)/len(list)))

def extract_target(path1, path2):
    reslit_dict = {}
    with open(path1,'r', encoding="utf-8") as file1, open(path2,'r', encoding="utf-8") as file2: 
        for line1, line2 in zip(file1, file2):
            sen_prpty_int = int(line1.split()[1])
            score = float(line2.split()[1])
            if line1.split()[1] in reslit_dict:
                reslit_dict[sen_prpty_int].append(score)
            else:
                reslit_dict[sen_prpty_int] = [score]
    return reslit_dict

if __name__ == '__main__':
    path1 = 'Data\\en\\gold\\sen_prpty_dev.txt'
    path2 = 'Data\\en\\gold\\result_dev.txt'
    reslit_dict = extract_target(path1, path2)
    calcuate_mean(reslit_dict)