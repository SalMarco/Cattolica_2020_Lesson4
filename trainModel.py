import os
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import advanced_activations
from keras.optimizers import Adam
from sklearn import preprocessing
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from google.cloud import bigquery,storage
from datetime import datetime
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-b','--destBucket',type=str,help=("Destination gs bucket for our results"),required=True)
argparser.add_argument('-k','--useKFols',help=("Make model evalutation using kfold"),action='store_true')
argparser.add_argument('-n','--numFolds',type=int,help=("Number of folds [default = 5]"),default=5)
group = argparser.add_mutually_exclusive_group(required=True)
group.add_argument('-f','--fileIn',type=str,help=(" CSV with the data to be analized"))
group.add_argument('-q','--queryTable',type=str,help=("Destination table for the query"))

# Parameters
learning_rate = 0.001
num_steps = 5
batch_size = 1000
n_fold = 5

# Network Parameters
n_hidden_1 = 25 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
#num_input = 6 # MNIST data input (img shape: 28*28)
num_classes = 1 # MNIST total classes (0-9 digits)

trainFile = 'promoted.csv'

#
activationFun = 'relu'
#activationFun = 'softmax'

testOneHot = False

parDict = {'kFold':
           {'outName': 'kFoldResults',
            'ext':'csv',
           'folder':'modelsEvals'},
           'train':{'outName': 'nnModel',
            'ext':'h5',
           'folder':'modelsTrained'}}

class LoadData:
    def __init__(self,**kargs):
        self.path = 'data'
        self.trainFile = kargs['tr']
        self.queryTable = kargs['qt']
        self.colList = ['avg_bal','geo_group', 'res_type']
        self.client = bigquery.Client()
        
        
    def readFiles(self,fileName):
        fullPath = os.path.join(self.path,fileName)
        logger.info('READING %s',fullPath)
        df = pd.read_csv(fullPath,sep=',',dtype={'avg_bal':'category', 'geo_group':'category', 'res_type':'category',})#dtype={'avg_bal':'category', 'geo_group':'category', 'res_type':'category',}
        logger.info('LOADED DATASET WITH SHAPE %s, COLUMUNS %s AND TYPES %s',str(df.shape),str(df.columns),str(df.dtypes))
        print('After reading',df.describe())
        return df
    
    
    def queryData(self):
        logger.info('QUERING TABLE %s'%self.queryTable)
        query = 'select * from %s'%self.queryTable
        df = self.client.query(query).to_dataframe()
        df[self.colList] = df[self.colList].astype('category')
        return df
        
    
    def useOneHot(self,df):
        print('SHAPE OF DF: %i X %i'%(df.shape[0],df.shape[1]))
        df2Enc = df.loc[:,self.colList]
        print('SHAPE OF DF2ENC: %i X %i'%(df2Enc.shape[0],df2Enc.shape[1]))
        enc = OneHotEncoder()
        dfEnc = enc.fit_transform(df2Enc)
        print('SHAPE OF DFENC: %i X %i'%(dfEnc.shape[0],dfEnc.shape[1]))
        print(type(dfEnc))
        dfEnc = pd.DataFrame(dfEnc.toarray())
        print('SHAPE OF DFENC: %i X %i'%(dfEnc.shape[0],dfEnc.shape[1]))
        #df.drop(columns=self.colList,inplace=True)
        return dfEnc
        
        
    def prepareTrain(self):
        if self.trainFile: 
            dfTrain = self.readFiles(self.trainFile)
        else:
            dfTrain = self.queryData()
        logger.info('REMOVING ROWS WITH NA')
        logger.info('NROWS BEFORE REMOVING NA %i',dfTrain.shape[0])
        dfTrain.dropna(inplace=True)
        dfTrain.drop(columns=['customer_id'],inplace=True)
        logger.info('NROWS AFTER REMOVING NA %i',dfTrain.shape[0])
        Y_train = dfTrain.loc[:,'resp']
        logger.info('CONSIDERING LEVELS FOR CATEGORICAL COLUMNS')
        if testOneHot:
            logger.info('Using OHE')
            dfEnc = self.useOneHot(dfTrain)
            print('SHAPE OF X_train: %i X %i'%(dfEnc.shape[0],dfEnc.shape[1]))
        else:
            for curCol in self.colList:
                dfTrain[curCol] = dfTrain[curCol].cat.codes
                dfEnc = dfTrain.copy()
        logger.info('SCALING OF NUMERIC COLUMNS')
        mmscaler = preprocessing.MinMaxScaler()
        for curCol in ['card_tenure', 'risk_score', 'num_promoted']:
            curFeat = mmscaler.fit_transform(dfTrain[[curCol]])  
            dfEnc[curCol] = curFeat.reshape(-1,1)
        print('SHAPE OF dfEnc AFTER ALL: %i X %i'%(dfEnc.shape[0],dfEnc.shape[1]))
        print('AFTER PREPROCESSING dfEnc HAS COLUMUNS %s AND TYPES %s'%(str(dfEnc.columns),str(dfEnc.dtypes)))
        return dfEnc, Y_train

class CreateNN:
    def __init__(self,**kargs):
        self.X_train = kargs['xt']
        self.Y_train = kargs['yt']
        n_fold = kargs['nf']
        self.kFold = kfold = StratifiedKFold(n_splits=n_fold)
        self.i = 1
        self.num_input = self.X_train.shape[1]
        
    def modelDefinition(self):
        logger.info('DEFINITION OF THE MODEL')
        self.model = Sequential()
        self.model.add(Dense(self.num_input, input_dim = self.num_input,activation=activationFun))
        self.model.add(Dense(n_hidden_1,activation = activationFun))
        self.model.add(Dense(n_hidden_2,activation = activationFun))
        self.model.add(Dense(num_classes,activation = 'sigmoid'))
        print(self.model.summary())
    
    def modelCompile(self):
        logger.info('COMPILATION OF THE MODEL')
        adam = Adam(lr = learning_rate)
        self.model.compile(loss = 'binary_crossentropy', optimizer = adam,metrics = ['accuracy'])
        
    def modelEval(self):
        logger.info('EVALUATION OF THE MODEL')
        totalScores = list()
        logger.info('START OF THE CROSS VALIDATION')
        for train,test in self.kFold.split(self.X_train, self.Y_train):
            logger.info('WORKING ON FOLD %i',self.i)
            print('train set',train)
            history = self.model.fit(self.X_train.iloc[train], self.Y_train.iloc[train],
                                     epochs=num_steps, 
                                     batch_size = batch_size) #validation_data=(self.X_train.iloc[test], self.Y_train.iloc[test])
            scores = self.model.evaluate(self.X_train.iloc[test], self.Y_train.iloc[test])
            totalScores.append(scores[1])
            self.i += 1
        return totalScores
    
    def modelTrain(self):
        logger.info('TRAINING OF THE MODEL')
        logger.info('START TRAINING')
        self.model.fit(self.X_train, self.Y_train,
                                     epochs=num_steps, 
                                     batch_size = batch_size)
        return self.model

class CopyOnGS:
    def __init__(self,**kargs):
        self.useKFols=kargs['uk']
        self.destBucket=kargs['db']
        self.ret = kargs['ret']
        tmpFolder = '/tmp'
        curTime = datetime.now().strftime("%Y%m%d-%H%M%S")
        fileNameTmp = '{nameOut}_%s.{ext}'%curTime 
        self.retDict = parDict['kFold'] if self.useKFols else parDict['train']
        self.fileName = fileNameTmp.format(nameOut=self.retDict['outName'],ext=self.retDict['ext'])
        self.fullPath = os.path.join(tmpFolder,self.fileName)
        
    def prepareFile(self):
        if self.useKFols:
            logger.info(f'WRITING RESULTS OF KFOLD IN {self.fullPath}')
            with open(self.fullPath,'w') as fo:
                retStr = [str(i) for i in self.ret]
                fo.write(','.join(retStr))
        else:
            logger.info(f'WRITING NN MODEL IN {self.fullPath}')
            self.ret.save(self.fullPath)
            
    def copyOnGCS(self):
        logger.info('COPY OF FILE')
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.destBucket)   
        retFile = '%s/%s'%(self.retDict['folder'],self.fileName)
        blob = bucket.blob(retFile)
        blob.upload_from_filename(self.fullPath)
        logger.info('FILE WRITED IN BUCKET %s AS %s'%(self.destBucket,retFile))
        logger.info('COPY OF FILE DONE')
        

    
    
def main():
    #Inizialization of the class LoadData 
    logger.info('INIZIALIZATION OF LOADDATA')
    args = argparser.parse_args()
    trainFile = args.fileIn
    queryTable = args.queryTable
    n_fold = args.numFolds
    useKFols = args.useKFols
    destBucket = args.destBucket
    logger.info(f'RUNNING WITH PARAMETERS: trainFile:{trainFile}, queryTable:{queryTable}, n_fold:{n_fold}, useKFols:{useKFols} and  destBucket:{destBucket}')
    ld = LoadData(tr=trainFile,qt=queryTable)
    print(type(ld))
    #df2Pred = ld.readFiles(predFile)
    X_train, Y_train = ld.prepareTrain()
    logger.info('INIZIALIZATION OF CreateNN')
    cnn = CreateNN(xt=X_train,yt=Y_train,nf=n_fold)
    cnn.modelDefinition()
    cnn.modelCompile()
    if useKFols:
        ret = cnn.modelEval() 
        logger.info('EVALUATION COMPLETED')
        logger.info("FOR THE ACTUAL MODEL THE RESULTS IS: %.2f%%+/-%.2f%%" %(np.mean(ret),np.std(ret)))
    else:
        ret = cnn.modelTrain()
    logger.info('MOVING RESULTS TO GCS')
    cpg = CopyOnGS(ret=ret,uk=useKFols,db=destBucket)
    cpg.prepareFile()
    cpg.copyOnGCS()
        
    
    
if __name__=="__main__":
    main()
    
#python trainModel.py -b cattolica2020 -q 'data-playground-241214.Cattolica_2020_Example.test_promoted'