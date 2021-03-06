_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-cospcl.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0005)

