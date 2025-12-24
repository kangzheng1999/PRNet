import time
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from dataset import collate_fn, CorrespondencesDataset
from utils import compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E
from config import get_config
from fvcore.nn import FlopCountAnalysis

sys.path.append('../core')
from convmatch import ConvMatch

torch.set_grad_enabled(False)
torch.manual_seed(0)

def inlier_test(config, polar_dis, inlier_mask):
    polar_dis = polar_dis.reshape(inlier_mask.shape).unsqueeze(0)
    inlier_mask = torch.from_numpy(inlier_mask).type(torch.float32)
    is_pos = (polar_dis < config.obj_geod_th).type(inlier_mask.type())
    is_neg = (polar_dis >= config.obj_geod_th).type(inlier_mask.type())
    precision = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        (torch.sum(inlier_mask * (is_pos + is_neg), dim=1) + 1e-15)
    )
    recall = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )
    f_scores = 2 * precision * recall / (precision + recall + 1e-15)

    return precision, recall, f_scores

def count_flops(model, input_tensor):
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     model(input_tensor)
    # return prof.key_averages().total_average().cpu_time_total
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def test(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    test_dataset = CorrespondencesDataset(config.data_te, config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model = ConvMatch(config)

    save_file_best = os.path.join(config.model_file, "model_best.pth")
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting")
        exit(1)
    # Restore model
    checkpoint = torch.load(save_file_best)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    model.to(device)
    model.eval()

    # 获取一个数据批次来推断输入的形状
    first_batch = next(iter(test_loader))
    num_kpts = first_batch['xs'].shape[-1]

    # 创建 dummy_input，用于计算 FLOPs
    dummy_input = {
        'xs': torch.randn(1, 1, 4, num_kpts, device=device)  # 形状调整为 (batch_size, 1, 4, num_kpts)
    }

    # Calculate Flops and Params
    flops = count_flops(model, dummy_input)
    params = count_params(model)

    err_ts, err_Rs = [], []
    precision_all, recall_all, f_scores_all = [], [], []

    total_time = 0.0
    num_images = len(test_loader)
    max_memory = 0.0

    for index, test_data in enumerate(tqdm(test_loader)):
        x = test_data['xs'].to(device)
        y = test_data['ys'].to(device)
        R_gt, t_gt = test_data['Rs'], test_data['ts']

        data = {'xs': x}

        start_time = time.time()

        logits_list, e_hat_list = model(data)
        logits = logits_list[-1]
        e_hat = e_hat_list[-1].cpu().detach().numpy().reshape(3, 3)

        end_time = time.time()
        total_time += (end_time - start_time)

        torch.cuda.synchronize()
        max_memory = max(max_memory, torch.cuda.max_memory_allocated(device) / (1024 ** 2))  # in MB

        mkpts0 = x.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = x.squeeze()[:, 2:].cpu().detach().numpy()
        inlier_mask = logits.squeeze().cpu().detach().numpy() > config.inlier_threshold
        mask_kp0 = mkpts0[inlier_mask]
        mask_kp1 = mkpts1[inlier_mask]

        if config.use_ransac:
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1, conf=config.ransac_prob)
        else:
            if e_hat.shape[0] == 0:
                print("Algorithm has no essential matrix output, can not eval without robust estimator such as RANSAC.")
                print("Try to set use_ransac=True in config file.")
                exit(1)
            ret = estimate_pose_from_E(mkpts0, mkpts1, inlier_mask, e_hat)
        if ret is None:
            err_t, err_R = np.inf, np.inf
            precision_all.append(torch.zeros(1, )[0])
            recall_all.append(torch.zeros(1, )[0])
            f_scores_all.append(torch.zeros(1, )[0])
        else:
            R, t, inlier_mask_new = ret
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)

        precision, recall, f_scores = inlier_test(config, y, inlier_mask)

        precision_all.append(precision.cpu())
        recall_all.append(recall.cpu())
        f_scores_all.append(f_scores.cpu())

    avg_time = total_time / num_images

    out_eval = {'error_t': err_ts,
                'error_R': err_Rs}

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100. * yy for yy in aucs]
    precision, recall, f_scores = np.mean(np.asarray(precision_all)) * 100, np.mean(np.asarray(recall_all)) * 100, np.mean(np.asarray(f_scores_all)) * 100

    print('Evaluation Results (mean over {} pairs):'.format(num_images))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
    print('Prec\t Rec\t F1\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(precision, recall, f_scores))

    print('\nAdditional Metrics:')
    print('Average Inference Time per Image: {:.6f} seconds'.format(avg_time))
    print('Total Flops: {:.2f} GFLOPs'.format(flops / 1e9))
    print('Model Parameters: {:.2f} million'.format(params / 1e6))
    print('Peak Memory Usage: {:.2f} MB'.format(max_memory))

if __name__ == '__main__':
    config, unparsed = get_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    test(config)

