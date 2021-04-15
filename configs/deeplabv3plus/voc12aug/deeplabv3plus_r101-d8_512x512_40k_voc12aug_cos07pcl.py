_base_ = '../deeplabv3plus_r50-d8_512x512_40k_voc12aug.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(num_classes=21,
                              loss_decode=dict(type='KLPatchContrastiveLoss',
                                               use_sigmoid=False,
                                               loss_weight=0.2,
                                               cal_function='COS07',
                                               cal_gate=99)))
data = dict(samples_per_gpu=2, workers_per_gpu=2)