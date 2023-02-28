from ctgan import load_demo
import numpy as np
from utils import ConstrainedBrownianBridge
import pandas as pd
from sdv.metrics.tabular import BinaryMLPClassifier, BinaryAdaBoostClassifier, BinaryLogisticRegression
import argparse

def setup_seed(seed):
    import torch
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--eval_gt', action='store_true', default=False)
args = parser.parse_args()

data = load_demo()
data = data.drop('education-num', axis=1)

print(data.columns)
# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'marital-status',
    'education',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

continuous_constraint_columns = ['age', 'capital-gain', 'capital-loss', 'fnlwgt', 'hours-per-week']

print('Age range:', data['age'].values.min(),'-', data['age'].values.max())
print('Capital gain range:', data['capital-gain'].values.min(), '-', data['capital-gain'].values.max())
print('Capital loss range:', data['capital-loss'].values.min(), '-', data['capital-loss'].values.max())
print('FNLWGT range:', data['fnlwgt'].values.min(), '-', data['fnlwgt'].values.max())
print('Hours-per-week range:', data['hours-per-week'].values.min(), '-', data['hours-per-week'].values.max())

if not args.eval_gt:

    bridge = ConstrainedBrownianBridge(data_shape=(1, 109),
                            N=2000,
                            sigma_min = 1.0,
                            sigma_max = 3.0,
                            noise_type = 'exp',
                            with_noise_decay=True,
                            )
    bridge.fit(data, discrete_columns, continuous_constraint_columns, iterations=5000, batchsize=128)

    # Synthetic copy
    samples = []
    for i in range(1):
        samples.append(bridge.sample(30000))

    samples = pd.concat(samples)
    print('Synthetic Age range:', samples['age'].values.min(), '-', samples['age'].values.max())
    print('Synthetic Capital gain range:', samples['capital-gain'].values.min(), '-', samples['capital-gain'].values.max())
    print('Synthetic Capital loss range:', samples['capital-loss'].values.min(), '-', samples['capital-loss'].values.max())
    print('Synthetic FNLWGT range:', samples['fnlwgt'].values.min(), '-', samples['fnlwgt'].values.max())
    print('Synthetic Hours-per-week range:', samples['hours-per-week'].values.min(), '-', samples['hours-per-week'].values.max())  
    samples['age'].mask(samples['age']<0, 0, inplace=True)
    samples['capital-gain'].mask(samples['capital-gain']<0, 0, inplace=True)
    samples['capital-loss'].mask(samples['capital-loss']<0, 0, inplace=True)
    samples['fnlwgt'].mask(samples['fnlwgt']<0, 0, inplace=True)
    samples['hours-per-week'].mask(samples['hours-per-week']<0, 0, inplace=True)
    print(samples[0:5])

    print('Synthetic Logistic Regression Classification', BinaryLogisticRegression.compute(data, samples, target='income'))
    print('Synthetic Adaboost Classification', BinaryAdaBoostClassifier.compute(data, samples, target='income'))
    print('Synthetic MLP Classification', BinaryMLPClassifier.compute(data, samples, target='income'))

else:

    ### NOTE: fix sdmetrics/single_table/efficacy/base.py L51 to
    ###         train_data = ht.fit_transform(train_data)
    ###         test_data = ht.transform(test_data)
    print('Number of data points in the dataset:', len(data))
    indices = np.random.permutation(len(data))
    train = data.iloc[indices[:30000], :]
    test = data.iloc[indices[30000:], :]
    print('GT Logistic Regression Classification', BinaryLogisticRegression.compute(test, train, target='income'))
    print('GT Adaboost Classification', BinaryAdaBoostClassifier.compute(test, train, target='income'))
    print('GT MLP Classification', BinaryMLPClassifier.compute(test, train, target='income'))
