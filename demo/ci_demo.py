from pipeline import Informer
from salience_basic_util import SEQ_TASK

#  SST-2 (10,000 instances)
d = Informer.confidence_indication_serialized(
        num_classes=2,
        scores_path="serialized_data/SST-tiny-bert/lime/train_10k.json",
        preds_path="serialized_data/SST-tiny-bert/predictions_train_10k.json",
        instance=False, visual=True)

for k, v in d.items():
    print(k, "|", v)
