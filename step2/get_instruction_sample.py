
from streaming import StreamingDataset, MDSWriter

mds_path = "/home/zhliu/database/mds_train_samples"

dataset = StreamingDataset(local=mds_path,split=None)

print("Total number of samples in the dataset:", len(dataset))
for i in range(min(20, len(dataset))):
    sample = dataset[i]
    print(f"Sample {i}:")
    print("Instruction:{}".format(sample['instruction']))
    print("Query:{}".format(sample['queries']))
    print("retrieved_ids:{}".format(sample['retrieved_ids']))
    print(sample['train_output'])
    print()