#_*_coding:UTF-8_*_
import os
import sys
import pdb
import pickle
import sklearn
import numpy as np
import sklearn.ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#显示测试结果
def show_cm(cm, labels):
    # Compute percentanges
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T)
    print('Confusion Matrix Stats')
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))


#读取模型
def load_model_from_disk(name, model_dir):
    # Model directory is relative to this file
    model_path = os.path.join(model_dir, name+'.model')
    # Put a try/except around the model load in case it fails
    try:
        model = pickle.loads(open(model_path,'rb').read())
        return model
    except:
        print('Could not load model: %s from directory %s!' % (name, model_path))
        return None


def predict_data(test_file):
    if test_file.endswith('.txt'):
        test_data = np.loadtxt(test_file)
        if test_data.ndim==1:
	    test_data=[test_data]
    elif test_file.endswith('.npy'):
        test_data = np.load(test_file)
    else:
        print('wrong file')
        exit(0)
    if not os.path.isfile(model_name + '.model'):
        print('model file ' + model_name + '.model' + ' not found.')
        exit(0)
    clf = load_model_from_disk(model_name, '')
    y_pred = clf.predict(test_data)
    print(clf.predict_proba(test_data))
    return y_pred,test_data


def cal_result(results,test_data):
    if len(results)>0:
        b_count=0
        m_count=0
        fe_benign=[]
        fe_mal=[]
        out_result=[]
        k=0
        for r in results:
            if r==0:
                b_count +=1
            elif r==1:
                m_count +=1
            if data_label=="0":
                if r==1:
                    out_result.append(test_data[k])
            elif data_label=="1":
                 if r==0:
                     out_result.append(test_data[k])
            k +=1
        return b_count,m_count,out_result


def run(feature_path, f_result, f_fea):
    s=0
    f_result.write(str(sys.argv)+'\n')
    if os.path.isfile(feature_path):
        results,test_data=predict_data(feature_path)
        b_count,m_count,out_result=cal_result(results,test_data)
        s=b_count+m_count
        f_result.write(feature_path+'\n')
        f_result.write("benign:"+str(b_count)+'\n')
        f_result.write("mal:"+str(m_count)+'\n')
        f_result.write("total:"+str(s))
        f_fea.write(feature_path+'\n')
        f_fea.write("feature:"+str(out_result)+'\n')
    elif os.path.isdir(feature_path):
        s=0
        s_b=0
        s_m=0
        for f in os.listdir(feature_path):
            try:
                print(feature_path+f)
                results,test_data=predict_data(feature_path+f)
                print(results)
                b_count,m_count,out_result=cal_result(results,test_data)
                f_result.write(feature_path+f+'\n')
                f_result.write("benign:"+str(b_count)+'\n')
                f_result.write("mal:"+str(m_count)+'\n')
                f_fea.write(feature_path+f+'\n')
                f_fea.write("feature:"+str(out_result)+'\n')
                s_b +=b_count
                s_m +=m_count
            except:
                f_result.write("Error:"+feature_path+f+'\n')
                continue
        s=s_b+s_m
        f_result.write("\n\n")
        f_result.write("total_mal:"+str(s_m)+'\n')
        f_result.write("total_benign:"+str(s_b)+'\n')
        f_result.write("total:"+str(s)+'\n')
    f_result.close()
    f_fea.close()


def main():
    if len(sys.argv) < 5:
        print('Missing Parameters')
        print('1 test file(npy or txt); 2 model name; 3 out file; 4 test data mal(=1) or benign(=0)')
        exit(0)
    feature_path=sys.argv[1]
    model_name=sys.argv[2]
    f_result=open(sys.argv[3]+"_r","w")
    f_fea=open(sys.argv[3]+"_fea","w")
    global data_label
    data_label=sys.argv[4]
    run(feature_path, f_result, f_fea)
    

    
if __name__ == "__main__":
    main()
