_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-dcac.py',
    '../_base_/datasets/pascal_voc12_dcac.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_26k.py'
]
model = dict(
    decode_head=dict(num_classes=21))