import pandas as pd
from sdv.evaluation import evaluate


realFiles = ['../data/data_Distinction.csv',
            '../data/data_Pass.csv',
            '../data/data_Fail.csv',
            '../data/data_Withdrawn.csv']

ctgan = ['fakeCTGAN_data_Distinction.csv',
    'fakeCTGAN_data_Pass.csv',
    'fakeCTGAN_data_Fail.csv',
    'fakeCTGAN_data_Withdrawn.csv']

gaussianCopula = ['fakeGaussianCopula_data_Distinction.csv',
    'fakeGaussianCopula_data_Pass.csv',
    'fakeGaussianCopula_data_Fail.csv',
    'fakeGaussianCopula_data_Withdrawn.csv']

copulaGAN = ['fakeCopulaGAN_data_Distinction.csv',
    'fakeCopulaGAN_data_Pass.csv',
    'fakeCopulaGAN_data_Fail.csv',
    'fakeCopulaGAN_data_Withdrawn.csv']

tvae = ['fakeTVAE_data_Distinction.csv',
    'fakeTVAE_data_Pass.csv',
    'fakeTVAE_data_Fail.csv',
    'fakeTVAE_data_Withdrawn.csv']

promp = ['fake_data_Distinction.csv',
    'fake_data_Pass.csv',
    'fake_data_Fail.csv',
    'fake_data_Withdrawn.csv']

d={
    'dataset':['Distinction','Pass','Fail','Withdrawn'],
    'CTGAN':[],
    'gaussianCopula':[],
    'copulaGAN':[],
    'TVAE':[],
    'ProMP':[]
}
for i in range (0,4):
    real = pd.read_csv(realFiles[i])
    fakeCTGAN = pd.read_csv('results/CTGAN/'+ctgan[i])
    fakeGaussianCopula = pd.read_csv('results/gaussianCopula/'+gaussianCopula[i])
    fakeCopulaGAN = pd.read_csv('results/copulaGAN/'+copulaGAN[i])
    fakeTVAE = pd.read_csv('results/TVAE/'+tvae[i])
    fakePROMP = pd.read_csv('results/proMPs/'+promp[i])

    d['CTGAN'].append(str(evaluate(fakeCTGAN,real)))
    d['gaussianCopula'].append(str(evaluate(fakeGaussianCopula,real)))
    d['copulaGAN'].append(str(evaluate(fakeCopulaGAN,real)))
    d['TVAE'].append(str(evaluate(fakeTVAE,real)))
    d['ProMP'].append(str(evaluate(fakePROMP,real)))

df = pd.DataFrame(data=d)

df.to_csv('eval.csv',index=False)
    
    


# realdata = pd.read_csv('../data/GHPrimaire_PASS.csv')

# CTGANdata = pd.read_csv('results/GHPrimaire_PASS_fakeCTGAN.csv')
# gaussianCopula = pd.read_csv('results/GHPrimaire_PASS_fakeGaussianCopula.csv')

# evalCTGAN = evaluate(CTGANdata, realdata)
# evalGaussianCopula = evaluate(gaussianCopula, realdata)


# print("CTGAN : "+ str(evalCTGAN))
# print("Gaussian copula : "+ str(evalGaussianCopula))
