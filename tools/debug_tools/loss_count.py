import argparse
import ipdb


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count log file how much minus pwc-loss')
    parser.add_argument('in_file', help='input log filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file):
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    neg_count = 0
    pos_count = 0
    neg = 0
    pos = 0
    with open(in_file, encoding='utf-8') as f:
        loss_line = f.readlines()
    loss_line = loss_line[500:]
    for i in loss_line:
        if '--' in i:
            neg_count += 1
            minus = i.split('-')
            neg += float(minus[-1])
        elif 'pwc_loss' in i:
            pos_count += 1
            plus = i.split('-')
            pos += float(plus[-1])

    print('neg_count:', neg_count)
    print('pos_count:', pos_count)
    print('neg:', neg, neg/neg_count)
    print('pos:', pos, pos/pos_count)


def main():
    args = parse_args()
    process_checkpoint(args.in_file)


if __name__ == '__main__':
    main()
