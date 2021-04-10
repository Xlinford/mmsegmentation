_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-cospcl.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21,
                     loss_decode=dict(type='KLPatchContrastiveLoss',
                                      use_sigmoid=False,
                                      loss_weight=0.25,
                                      cal_function='COS07',
                                      cal_gate=99)),
    auxiliary_head=dict(num_classes=21),)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)


