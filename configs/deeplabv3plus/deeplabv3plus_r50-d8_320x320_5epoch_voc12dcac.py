_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-se.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_5epoch.py'
]
model = dict(
    decode_head=dict(num_classes=21))