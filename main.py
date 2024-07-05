import torch
import random
import numpy as np
import argparse
import torch.distributed as dist
from utils.misc import get_rank, init_distributed_mode
from utils.dataset import Dataset_Creator
import utils.misc as utils
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from models import build_model
from sklearn.metrics import average_precision_score, accuracy_score


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # dataset parameters
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    parser.add_argument('--test_selected_subsets', nargs='+', required=True)
    parser.add_argument('--batchsize', type=int, default=32)

    # model
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=8)

    # text encoder
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")

    # frequency
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--clip_vision_width', type=int, default=1024)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)

    # output
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    return parser


@torch.no_grad()
def gather_together(data):
    world_size = dist.get_world_size()
    if world_size < 2:
        return data
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


@torch.no_grad()
def evaluate(model, data_loaders, device, args=None, test=False):
    model.eval()
    if test:
        images = data_loaders.to(device)
        outputs = model(images)
        print(outputs.softmax(dim=1)[:, 1].flatten().tolist())
        return

    test_dataset = []
    test_AP = []
    test_ACC = []
    test_real_ACC = []
    test_fake_ACC = []

    for data_name, data_loader in data_loaders.items():
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        print_freq = args.print_freq

        y_true, y_pred = [], []
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            images, labels = [sample.to(device) for sample in samples]
            outputs = model(images)

            y_pred.extend(outputs.softmax(dim=1)[:, 1].flatten().tolist())
            y_true.extend(labels.flatten().tolist())
        
        merge_y_true = []
        for data in gather_together(y_true):
            merge_y_true.extend(data)
        
        merge_y_pred = []
        for data in gather_together(y_pred):
            merge_y_pred.extend(data)
        
        y_true, y_pred = np.array(merge_y_true), np.array(merge_y_pred)

        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        test_dataset.append(data_name)
        test_AP.append(ap)
        test_ACC.append(acc)
        test_real_ACC.append(r_acc)
        test_fake_ACC.append(f_acc)


    output_strs = []
    for idx, [name, ap, acc, racc, facc] in enumerate(zip(test_dataset + ["mean"], test_AP + [np.mean(test_AP)], test_ACC + [np.mean(test_ACC)], test_real_ACC + [np.mean(test_real_ACC)], test_fake_ACC + [np.mean(test_fake_ACC)])):
        output_str = "({} {:10}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(idx, name, acc*100, ap*100, racc*100, facc*100)
        output_strs.append(output_str)
        print(output_str)
    
    return "; ".join(output_strs), np.mean(test_AP), np.mean(test_ACC)


def main(args):
    init_distributed_mode(args)

    # set device
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    # infer dataset
    dataset_creator = Dataset_Creator(dataset_path=args.dataset_path, batch_size=args.batchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
    dataset_vals, selected_subsets = dataset_creator.build_dataset("test", selected_subsets=args.test_selected_subsets)
    data_loader_vals = {}
    for dataset_val, selected_subset in zip(dataset_vals, selected_subsets):
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
        data_loader_vals[selected_subset] = DataLoader(dataset_val, args.batchsize, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)
    
    # model
    model = build_model(args)
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    if args.eval:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        evaluate(model, data_loader_vals, device,args=args)


if __name__ == "__main__":    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
