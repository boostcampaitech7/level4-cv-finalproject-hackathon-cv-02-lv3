import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('/data/ephemeral/home/Jungyeon/level4-cv-finalproject-hackathon-cv-02-lv3/AutoML_Other_Data/data/attrition_tpot.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Attrition', axis=1)
print(type(features))  

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Attrition'], random_state=42)


# Average CV score on the training set was: 0.8096379780825048
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.8500000000000001, min_samples_leaf=8, min_samples_split=10, n_estimators=100, subsample=0.35000000000000003)),
    DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=20, min_samples_split=4)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features.values, training_target)
results = exported_pipeline.predict(testing_features)

print(results)
