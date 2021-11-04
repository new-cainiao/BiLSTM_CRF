# coding=utf-8
import codecs

def calculate(x,y,id2word,id2tag,res=[]):
    entity=[]
    for j in range(len(x)):
        if int(x[j])==0 or id2tag[y[j]]=="":
            continue
        if id2tag[y[j]][0]=='B':
            entity=[id2word[int(x[j])]+'/'+id2tag[y[j]]]
        elif id2tag[y[j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
            entity.append(id2word[int(x[j])]+'/'+id2tag[y[j]])
        elif id2tag[y[j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
            entity.append(id2word[int(x[j])]+'/'+id2tag[y[j]])
            entity.append(str(j))
            res.append(entity)
            entity=[]
        else:
            entity=[]
    return res
    
    
def calculate3(x,y,id2word,id2tag, start_idx=0):
    '''
    使用这个函数可以把抽取出的实体写到res.txt文件中，供我们查看。
    注意，这个函数每次使用是在文档的最后添加新信息，所以使用时尽量删除res文件后使用。
    '''
    res = []
    with codecs.open('./res.txt','a','utf-8') as outp:
        entity=[]
        for j in range(len(x)): #for every word
            if int(x[j])==0 or id2tag[y[j]]=="":
                continue
            if id2tag[y[j]][0]=='B':
                entity=[id2word[int(x[j])]+'/'+id2tag[y[j]]]
                location_coordinate = ""
                location_coordinate += "{}-".format(j + start_idx)
            elif id2tag[y[j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
                entity.append(id2word[int(x[j])]+'/'+id2tag[y[j]])
            elif id2tag[y[j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
                entity.append(id2word[int(x[j])]+'/'+id2tag[y[j]])
                location_coordinate += str(j + start_idx)
                entity.append(str(location_coordinate))
                res.append(entity)
                st = ""
                for s in entity:
                    st += s+' '
                #print st
                outp.write(st+'\n')
                entity=[]
            else:
                entity=[]
    return res