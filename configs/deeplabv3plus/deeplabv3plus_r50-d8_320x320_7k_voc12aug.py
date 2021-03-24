_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-e2d.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_7k.py'
]
model = dict(
    decode_head=dict(num_classes=21))

# EncoderDecoder2Inputs debug loss:cross entropy;