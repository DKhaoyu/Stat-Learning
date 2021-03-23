import numpy as np
import math
from scipy import stats
#x_in : shape sample_len*2
#x_in[...,0] : classification of sample points
#x_in[...,1] : character of sample
#n : possibal classfication number,uint32 type,eg:0,1,2,3,4...etc
#K : possibal character number,uint32 type,eg:0,1,2,3,4...etc
tree_struct = []
iter_list = []
class treeNode:
    global tree_struct
    def __init__(self,id,leaf_flag,internal_flag,feat,class_dict,class_num):
        self.id = id
        self.leaf_flag = leaf_flag
        self.internal_flag = internal_flag
        self.feat = feat
        self.class_dict = class_dict
        self.class_num = class_num
        self.exsist = 1
        self.son = {}
    def update_iter(self):
        if self.internal_flag:
            self.iter = 1
            for k in self.son.values():
                if tree_struct[k].leaf_flag == 0:
                    self.iter = 0
                    break
        else:
            self.iter = 0  
        return self.iter
    def CalEmpEntropy(self):
        Ent = 0.0
        for s in self.class_dict.values():
            Ent += (-s*np.log2(s/self.class_num))
        return Ent
    def delNode(self):
        self.exsist = 0
    def SetLeaf(self):
        self.leaf_flag = 1
        self.internal_flag = 0
    def GetLeafClass(self):
        temp_num = -1
        temp_class = -1
        for k,v in self.class_dict.items():
            if v > temp_num:
                temp_num = v
                temp_class = k
        return temp_class

def InfoGain(x_in,n,K):
    sample_len = np.size(x_in,0)
    class_seq = np.zeros(n)
    chara_seq = np.zeros(K)
    unit_seq = np.zeros((n,K))      #unit_seq[i][j] : i refers to class, j refers to chara
    for i in range(sample_len):
        unit_seq[x_in[i,0],x_in[i,1]] += 1
        class_seq[x_in[i,0]] += 1
        chara_seq[x_in[i,1]] += 1
    unit_seq /= sample_len
    class_seq /= sample_len
    chara_seq /= sample_len
    h_D = 0
    h_DconA = 0
    for i in range(n):
        h_D += (-class_seq[i]*np.log2(class_seq[i]))
        for j in range(K):
            if unit_seq[i,j] != 0:
                h_DconA += (-unit_seq[i,j]*np.log2(unit_seq[i,j]/chara_seq[j]))
    multi_info = h_D - h_DconA
    gain_ratio = multi_info/h_D    
    return multi_info,gain_ratio

def ClassCheck(x_in):
    flag = 1
    class_seq = {}
    class_y = x_in[0]
    temp = x_in[0]
    seq_len = np.size(x_in)
    class_seq[x_in[0]] = 1
    for k in range(seq_len-1):
        if x_in[k] not in class_seq.keys():
            class_seq[x_in[k]] = 0
        class_seq[x_in[k]] += 1
        if x_in[k] != temp:
            flag = 0
            class_y = -1
    return flag,class_y,class_seq
#chara_set:a dictionary,key means the character name,val means the number of this character
#class_seq:sample_len*1,an array
#x_in is a dictionary,key defines the meaning of the list, x0, x1....
#n:classification number
def DataSplit(class_seq,x_in,key,val_seq,sub_lable):
    feat_num = len(val_seq)
    sub_data = [[] for n in range(feat_num)]
    sub_class_seq = [[] for n in range(feat_num)]
    for k in range(feat_num):
        temp_sub_data = {}
        equal_pos = (np.array(x_in[key]) == val_seq[k])
        for m in sub_lable:
            temp_sub_data[m] = list(np.array(x_in[m])[equal_pos])
        sub_class_seq[k]  = list(np.array(class_seq)[equal_pos])
        sub_data[k] = temp_sub_data
    return sub_data,sub_class_seq
Itr_cnt = -1
def ID3(class_seq,x_in,thresh,lable):
    global tree_struct
    global Itr_cnt 
    Itr_cnt += 1
    id = Itr_cnt
    flag,class_y,class_dict = ClassCheck(class_seq)
    if flag == 1:
        tree_struct += [treeNode(id,1,0,'',class_dict,len(class_seq))]
        return class_y,id
    else:
        if not x_in:
            tree_struct += [treeNode(id,1,0,'',class_dict,len(class_seq))]
            class_y = stats.mode(class_seq)[0][0]
            return class_y,id
        else:
            sample_len = np.size(class_seq)
            n = len(set(class_seq))
            temp_seq = np.zeros((sample_len,2),dtype = np.int32)
            temp_seq[...,0] = class_seq
            max_info = 0
            max_feat = ''
            for k in x_in.keys():
                temp_feat = x_in[k]
                feat_num = len(set(temp_feat))
                temp_seq[...,1] = temp_feat
                multi_info,gain_ratio = InfoGain(temp_seq,n,feat_num)
                if multi_info > max_info:
                    max_info = multi_info
                    max_feat = k
            
            if max_info < thresh:
                class_y = stats.mode(class_seq)[0][0]
                tree_struct += [treeNode(id,1,0,'',class_dict,len(class_seq))]
                return class_y,id
            else:
                ID3_tree = {max_feat:{}}
                lable.remove(max_feat)
                val_seq = list(set(x_in[max_feat]))
                tree_struct += [treeNode(id,0,1,max_feat,class_dict,len(class_seq))]
                sub_data,sub_class_seq = DataSplit(class_seq,x_in,max_feat,val_seq,lable)
                for k in range(len(val_seq)):
                    ID3_tree[max_feat][val_seq[k]],son_id = ID3(sub_class_seq[k],sub_data[k],thresh,lable)
                    tree_struct[id].son[val_seq[k]] = son_id
                return ID3_tree,id

def SearchTree(ID3_tree,feat_seq):
    sub_dict = ID3_tree
    y_out = -1
    for key in sub_dict.keys():
        sub_dict = sub_dict[key][feat_seq[key]]
        flag = isinstance(sub_dict,np.int32)
        if flag:
            y_out = sub_dict
            break
    return y_out

def IterCheck():
    global iter_list
    for k in range(len(tree_struct)):
        flag = tree_struct[k].update_iter()
        if flag and (k not in iter_list):
            iter_list += [k]

def Pruning(alpha):
    while True:
        IterCheck()
        stop_flag = 1
        for k in range(len(iter_list)):
            CT_1 = tree_struct[iter_list[k]].CalEmpEntropy()
            CT_2 = 0.0
            for m in tree_struct[iter_list[k]].son.values():
                CT_2 += tree_struct[m].CalEmpEntropy()
            if CT_1 - alpha*(len(tree_struct[iter_list[k]].son)-1) - CT_2 < 0:
                for m in tree_struct[iter_list[k]].son.values():
                    tree_struct[m].delNode()
                tree_struct[iter_list[k]].SetLeaf()
                stop_flag = 0
                iter_list.remove(iter_list[k])
        if stop_flag:
            break

def SearchTreeStruct(feat_seq):
    cursor = tree_struct[0]
    while True:
        if cursor.leaf_flag == 1:
            class_out = cursor.GetLeafClass()
            break
        else:
            current_feat = cursor.feat
            cursor = tree_struct[cursor.son[feat_seq[current_feat]]]
    return class_out
                   
#def CreateRegTree(class_seq,x_in,thresh): 

class_seq = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
x_in = {}
x_in['age'] = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
x_in['job'] = [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0]
x_in['house'] = [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0]
x_in['debt'] = [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0] 
ID3_tree,id = ID3(class_seq,x_in,1e-3,['age','job','house','debt'])
alpha = 5
Pruning(alpha)
y_out =  SearchTreeStruct({'age':0,'job':0,'house':0,'debt':0})
print(y_out)
pass