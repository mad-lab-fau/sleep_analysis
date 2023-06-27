from sleep_analysis.datasets.mesadataset import MesaDataset
from sleep_analysis.feature_extraction.actigraphy import extract_actigraph_features
from sleep_analysis.feature_extraction.hrv import extract_hrv_features
from sleep_analysis.feature_extraction.rrv import extract_rrv_features
from sleep_analysis.feature_extraction.utils import merge_features
from sleep_analysis.preprocessing.mesa_dataset.edr import extract_edr_features
from sleep_analysis.preprocessing.mesa_dataset.preprocess_mesa import preprocess_mesa
from sleep_analysis.preprocessing.mesa_dataset.utils import check_dataset_validity

print("Extracting RRV features from MESA dataset...")
extract_rrv_features(overwrite=True)
print("Extracting EDR features from MESA dataset...")
extract_edr_features(overwrite=True)

preprocess_mesa()

extract_actigraph_features(overwrite=True)
extract_hrv_features(overwrite=True)
merge_features(overwrite=True)


dataset = MesaDataset()
check_dataset_validity(dataset)
