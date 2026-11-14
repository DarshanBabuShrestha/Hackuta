"""Train experimental simulated-seizure-like movement classifiers on UEA Epilepsy.
Usage: python train_epilepsy_models.py --zip Epilepsy.zip --output ./seizure_models
No clinical seizure detection validity is implied.
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
import joblib

RATE=16
LABELS=['NORMAL_ACTIVITY','SIMULATED_SEIZURE_MOTION']

def load_ts(z, name):
    contents=z.read(name).decode('utf-8')
    lines=contents.split('@data',1)[1].strip().splitlines()
    X=[]; y=[]; classes=[]
    for l in lines:
        parts=l.strip().split(':')
        label=parts[-1].strip()
        channels=[np.fromstring(s,sep=',',dtype=np.float32) for s in parts[:-1]]
        if len(channels)!=3 or any(len(x)!=206 for x in channels):
            raise ValueError('unexpected dimensions in '+name)
        X.append(np.stack(channels,axis=1))
        y.append(int(label=='EPILEPSY'));classes.append(label)
    return np.stack(X), np.asarray(y),classes

def features(X):
    # Same function is reusable on a batch of 206x3 Arduino windows ONLY after matching units, axes and preprocessing.
    results=[]
    for recording in X:
        cols=[recording[:,i] for i in range(3)] + [np.linalg.norm(recording,axis=1)]
        row=[]
        for s in cols:
            f,p=welch(s-np.mean(s),fs=RATE,nperseg=min(len(s),128))
            peak=int(np.argmax(p[1:])+1) if len(p)>1 else 0
            row.extend([np.mean(s),np.std(s),np.min(s),np.max(s),np.percentile(s,10),np.percentile(s,50),np.percentile(s,90),np.sqrt(np.mean(s*s)),np.std(np.diff(s)),f[peak],p[peak]/(p.sum()+1e-10)])
        results.append(row)
    return np.asarray(results,dtype=np.float32)

def metrics(y_true, prob, threshold=0.5):
    pred=(prob>=threshold).astype(int)
    return {'accuracy':float(accuracy_score(y_true,pred)), 'seizure_like_motion_recall':float(recall_score(y_true,pred,zero_division=0)), 'positive_precision':float(precision_score(y_true,pred,zero_division=0)), 'f1':float(f1_score(y_true,pred,zero_division=0)), 'confusion_matrix':confusion_matrix(y_true,pred,labels=[0,1]).tolist(), 'threshold':threshold}

def run(zip_path,output):
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        Xtr,ytr,c_tr=load_ts(z,'Epilepsy_TRAIN.ts')
        Xte,yte,c_te=load_ts(z,'Epilepsy_TEST.ts')
    rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)
    rf.fit(features(Xtr),ytr)
    rf_prob=rf.predict_proba(features(Xte))[:,list(rf.classes_).index(1)]
    result={'dataset':{'train_count':len(ytr),'test_count':len(yte),'train_positive':int(ytr.sum()),'test_positive':int(yte.sum()),'rate_hz':RATE,'shape_per_recording':[206,3], 'split':'official three-participant versus three-participant split'},'random_forest':metrics(yte,rf_prob)}
    joblib.dump({'model':rf,'feature_length':44,'fs_hz':RATE,'window_size':206,'input_axes':3,'positive_class':'SIMULATED_SEIZURE_MOTION','feature_extraction':'train_epilepsy_models.features'},output/'epilepsy_rf.joblib')
    np.savez_compressed(output/'test_samples.npz',X=Xte,y=yte)
    try:
        import torch
        from torch import nn
        torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Sequential(nn.Conv1d(3,12,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(12,20,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(20,1))
            def forward(self,x):return self.net(x).squeeze(1)
        # Normalization learned from training set only.
        avg=Xtr.mean((0,1));sd=Xtr.std((0,1)).clip(min=1e-5)
        prep=lambda X:torch.from_numpy(((X-avg)/sd).transpose(0,2,1).astype('float32'))
        tx=prep(Xtr);vx=prep(Xte)
        ty=torch.tensor(ytr,dtype=torch.float32)
        model=TinyCNN();opt=torch.optim.Adam(model.parameters(),lr=0.002,weight_decay=0.003)
        lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())]))
        # Fixed short training schedule. No test-driven early stopping/tuning.
        model.train()
        for epoch in range(35):
            perm=torch.randperm(len(tx))
            for ids in perm.split(24):
                opt.zero_grad();loss=lossfn(model(tx[ids]),ty[ids]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():cn_prob=torch.sigmoid(model(vx)).numpy()
        result['cnn']=metrics(yte,cn_prob)
        torch.save({'state_dict':model.state_dict(),'mean':avg.tolist(),'std':sd.tolist(),'window_size':206,'sample_rate_hz':RATE,'architecture':'Conv1d(3,12,7)->MaxPool1d(2)->Conv1d(12,20,5)->AdaptiveAvgPool1d(1)->Linear(20,1)'},output/'epilepsy_cnn.pt')
    except ImportError:
        result['cnn']='not trained: PyTorch missing'
    (output/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zip',default='Epilepsy.zip');parser.add_argument('--output',default='seizure_models')
    args=parser.parse_args();run(Path(args.zip),Path(args.output))
