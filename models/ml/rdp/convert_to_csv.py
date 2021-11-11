#coding:utf-8
#把生成的文件转换为csv格式的.
#逐行生成,并不合并,等实际模型需要的时候再进行合并
#格式
#时间戳,类别id,特征1，特征2，···
'''
类别id    :       动作类别
0         ：     editing_doc
1         :      reading_doc
2         :      surfing_web
3         :      installing_software
4         :      transfering_file
5         :      watching_video
'''
class_to_id={
    'editing_doc':0,
    'reading_doc':1,
    'surfing_web':2,
    'installing_software':3,
    'transfering_file':4,
    'watching_video':5
}
__author__ = 'jmh081701'
import  os
import  re
import  json
import  sys
def get_files(appname,gap,directory=r"E:\TempWorkStation\i-know-what-are-you-doing\dataset\vector"):
    files=[]
    for _root,_subs,_files in os.walk(directory):
        for file  in _files:
            if file.count(appname) and file.count("gap=%s"%gap):
                files.append(directory+"\\"+file)
    return files
if __name__ == '__main__':
    appnames=['micrords','anydesk','realvnc','teamviewer']
    gaps = [0.5,0.2,0.8]
    for appname in appnames:
        for gap in gaps:
            #if appname!='teamviewer' or gap!=0.2:
            #    continue
            DIRECOTRY=r"E:\TempWorkStation\i-know-what-are-you-doing\dataset\vector_flowid"
            files = get_files(appname=appname,gap=gap,directory=DIRECOTRY)
            label_rule = "_(.*?)\."
            label_pattern = re.compile(label_rule)
            TARGET=DIRECOTRY+"\\"+"csv"+"\\"+appname+"_"+str(gap) +".txt"
            fp = open(TARGET,'w')
            for file in files:
                print(file)
                label=class_to_id[label_pattern.findall(file.split("\\")[-1] )[0] ]
                with open(file) as jfp:
                    peaks_features=json.load(jfp)
                for i in range(peaks_features['counter']):
                    timestamp = peaks_features['timestamps'][i]
                    feature = peaks_features['feature'][i]
                    flowid  = peaks_features['flowids'][i]
                    fp.writelines("%s,%d,%d,"%(timestamp,label,flowid))
                    #print(len(feature))
                    if len(feature)!=96:
                        print(feature,i)
                        exit()
                    for j in range(len(feature)):
                        fp.writelines(str(feature[j]))
                        if j < (len(feature)-1):
                            fp.writelines(",")
                    fp.writelines("\n")
                    if i %1000 ==0 :
                        print('finished %d/%d'%(i,peaks_features['counter']))
                fp.flush()
            fp.close()