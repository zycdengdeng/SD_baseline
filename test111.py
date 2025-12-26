from src.dataset.Scannet import Scannetdataset
import yaml

config = yaml.safe_load(open("/mnt/vdb1/lyt/ArbiViewGen-main/configs/depth_generation_train.yaml"))
dataset = Scannetdataset(config, mode="test")

print("Total dataset length:", len(dataset))
print("First few entries in data_list:", dataset.data_list[:5])
print("Scene KFs:", {k: len(v) for k,v in dataset.scene_kfs.items()})
