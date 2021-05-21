#%%
import numpy as np

# %%
data = np.load('../1317_drug_v1.npz',allow_pickle=True)
data = data['data'].item()

# %%
with open('entity_list.txt','w') as f:
    f.writelines('org_id	remap_id\n')
    index = 0
    for drug in data['Drug_id']:
        f.writelines(drug + ' ' +str(index) + '\n')
        index += 1
    for chem in range(len(data['Drug_chem_structure'][0])):
        f.writelines('PubchemFP'+ str(chem)+ ' ' +str(index) + '\n')
        index += 1
#%%
#user_list
with open('user_list.txt','w') as f:
    f.writelines('org_id	remap_id\n')
    index = 0
    for drug in data['Drug_id']:
        f.writelines(drug + ' ' +str(index) + '\n')
        index += 1
    
# %%
with open('item_list.txt','w') as f:
    f.writelines('org_id remap_id freebase_id\n')
    index = 0
    for drug in data['Drug_id']:
        f.writelines(drug + ' ' +str(index) + '	'+drug + '\n')
        index += 1

# %%
with open('item_list.txt','w') as f:
    f.writelines('org_id remap_id freebase_id\n')
    index = 0
    for drug in data['Drug_id']:
        f.writelines(drug + ' ' +str(index) + '	'+drug + '\n')
        index += 1
#%%
def chem_type(i):
    if i>=0 and i<=114:
        return str(0)
    elif i>=115 and i<=262:
        return str(1)
    elif i>=263 and i<=326:
        return str(2)
    elif i>=327 and i<=415:
        return str(3)
    elif i>=416 and i<=459:
        return str(4)
    elif i>=460 and i<=712:
        return str(5)
    elif i>=713 and i<=880:
        return str(6)
    else:
        print('exception')
        return -1
#实体，类型，属性
with open('kg_final.txt','w') as f:
    # f.writelines('org_id remap_id freebase_id\n')
    index = 0
    for drug in data['Drug_chem_structure']:
        for i in range(len(drug)):
            if drug[i] == 1:
                f.writelines(str(index) +' '+chem_type(i)+' '+str(i+1317)+'\n')
        index += 1
#%%
for drug in data['Drug_chem_structure']:
    print(drug[0])
    break

# %%
#train  test
train_dict = {}
test_dict = {}
ddi = {}
with open('train.txt','w') as f1,open('test.txt','w') as f2:
    index = 0
    for row in data['Adj_binary']:
        index1 = 0
        token = 0
        ddi[index] = []
        for item in row:
            if item == 1:
                ddi[index].append(index1)
            index1 += 1
        index += 1
    ddi12 = ddi.copy()
    for drug1 in range(len(data['Adj_binary'])):
        drug2 = ddi[drug1]
        count = 0
        if len(drug2)>80:
            test_dict[drug1] = []
            f2.writelines(str(drug1)+' ')
            for d in drug2:
                if count %20 == 0:
                    test_dict[drug1].append(d)
                    ddi[drug1].remove(d)
                    ddi[d].remove(drug1)
                    f2.writelines(str(d)+' ')
                count += 1
            f2.writelines('\n')
    for drug1 in range(len(data['Adj_binary'])):
        drug2 = ddi[drug1]
        f1.writelines(str(drug1)+' ')
        for d in drug2:
            f1.writelines(str(d)+' ')
        f1.writelines('\n')
    # for drug1 in range(len(data['Adj_binary'])):
    #     drug2 = ddi[drug1]
    #     train_dict[drug1] = []
    #     f1.writelines(str(drug1)+' ')
    #     for d in drug2:
    #         if not ((drug in test_dict and d in test_dict[drug1]) or(d in test_dict and drug1 in test_dict[d])):
    #             train_dict[drug1].append(d)
    #             f1.writelines(str(d)+' ')
    #     f1.writelines('\n')
    # for drug1 in range(len(data['Adj_binary'])):
    #     drug2 = ddi[drug1]
    #     count = 0
    #     if len(drug2)>80:
    #         f2.writelines(str(drug1)+' ')
    #     f1.writelines(str(drug1)+' ')
    #     for d in drug2:
    #         count += 1
    #         if count%10 != 0 or len(drug2)<=80:
    #             f1.writelines(str(d)+' ')
    #         else:
    #             f2.writelines(str(d)+' ')
    #         if d > drug1:
    #             ddi[d].remove(drug1)
    #     f1.writelines('\n')
    #     if len(drug2)>80:
    #         f2.writelines('\n')
        # if sum(data['Adj_binary'][index]) > 80:
        #     token = 1
        #     f2.writelines(str(index)+' ')
        #     test_dict[index] = []
        # f1.writelines(str(index)+' ')
        # train_dict[index] = []
        # count = 0
        # for value in row:
        #     if value == 1 and ((count+1) % 10 != 0 or token == 0):
        #         f1.writelines(str(index1)+' ')
        #         train_dict[index].append(index1)
        #         count += 1
        #     elif value == 1 and token == 1:
        #         if index1 in train_dict and index in train_dict[index1]:
        #             f2.writelines(str(index1)+' ')
        #             test_dict[index].append(index1)
        #         count += 1
        #     index1 += 1
        # index += 1
        # f1.writelines('\n')
        # if sum(data['Adj_binary'][index]) > 80:
        #     f2.writelines('\n')

# %%
index = 0
for row in data['Adj_binary']:
    print(index)
    index += 1

# %%
