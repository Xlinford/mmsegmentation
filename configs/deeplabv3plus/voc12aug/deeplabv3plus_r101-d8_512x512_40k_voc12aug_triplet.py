_base_ = '../deeplabv3plus_r50-d8_512x512_40k_voc12aug.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(num_classes=21,
                              loss_decode=dict(type='KLPatchContrastiveLoss',
                                               use_sigmoid=False,
                                               loss_weight=0.3,
                                               cal_function='triplet',
                                               cal_gate=[49, 199])))
data = dict(samples_per_gpu=2, workers_per_gpu=2)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001)
