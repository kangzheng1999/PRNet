import numpy as np
import torch
import os
import sys
import cv2
from tqdm import tqdm
from dataset import collate_fn, CorrespondencesDataset
from utils import compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E
from config import get_config

sys.path.append('../core')
from convmatch import ConvMatch

torch.set_grad_enabled(False)
torch.manual_seed(0)


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))
    return ys

def estimate_pose_norm_kpts(kpts0, kpts1, thresh=1e-3, conf=0.99999):
	if len(kpts0) < 5:
		return None

	E, mask = cv2.findEssentialMat(
	kpts0, kpts1, np.eye(3), threshold=thresh, prob=conf,
	method=cv2.RANSAC)

	assert E is not None

	best_num_inliers = 0
	new_mask = mask
	ret = None
	for _E in np.split(E, len(E) / 3):
		n, R, t, mask_ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
		if n > best_num_inliers:
			best_num_inliers = n
			ret = (R, t[:, 0], mask.ravel() > 0)

	return ret

def estimate_pose_from_E(kpts0, kpts1, mask, E):
    assert E is not None
    mask = mask.astype(np.uint8)
    E = E.astype(np.float64)
    kpts0 = kpts0.astype(np.float64)
    kpts1 = kpts1.astype(np.float64)
    I = np.eye(3).astype(np.float64)

    best_num_inliers = 0
    ret = None

    for _E in np.split(E, len(E) / 3):

        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, I, 1e9, mask=mask)

        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def inlier_test(config, polar_dis, inlier_mask):
    polar_dis = polar_dis.reshape(inlier_mask.shape).unsqueeze(0)
    inlier_mask = torch.from_numpy(inlier_mask).type(torch.float32)
    is_pos = (polar_dis < config.obj_geod_th).type(inlier_mask.type())
    is_neg = (polar_dis >= config.obj_geod_th).type(inlier_mask.type())
    precision = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        (torch.sum(inlier_mask * (is_pos + is_neg), dim=1)+1e-15)
    )
    recall = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )
    f_scores = 2*precision*recall/(precision+recall+1e-15)

    return precision, recall, f_scores

def test(config, use_essential=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    test_dataset = CorrespondencesDataset(config.data_te, config)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model = ConvMatch(config)

    save_file_best = os.path.join(config.model_file, "model_best.pth")
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    # Restore model
    checkpoint = torch.load(save_file_best)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    model.cuda()
    model.eval()

    err_ts, err_Rs = [], []
    precision_all, recall_all, f_scores_all = [], [], []

    for index, test_data in enumerate(tqdm(test_loader)):
        # test_data = tocuda(test_data)
        x = test_data['xs'].to(device)
        y = test_data['ys'].to(device)
        R_gt, t_gt = test_data['Rs'], test_data['ts']

        data = {}
        data['xs'] = x
        logits_list, e_hat_list = model(data)
        logits = logits_list[-1]
        e_hat = e_hat_list[-1].cpu().detach().numpy().reshape(3,3)

        mkpts0 = x.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = x.squeeze()[:, 2:].cpu().detach().numpy()
        # use essential matrix
        if use_essential:
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat_list[-1])
            inlier_mask = y_hat.squeeze().cpu().detach().numpy() < config.obj_geod_th
        # use logits
        else:
            inlier_mask = logits.squeeze().cpu().detach().numpy() > config.inlier_threshold
        mask_kp0 = mkpts0[inlier_mask]
        mask_kp1 = mkpts1[inlier_mask]

        if config.use_ransac == True:
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1, conf=config.ransac_prob)
        else:
            if e_hat.shape[0] == 0:
                print("Algorithm has no essential matrix output, can not eval without robust estimator such as RANSAC.")
                print("Try to set use_ransac=True in config file.")
                exit(1)
            ret = estimate_pose_from_E(mkpts0, mkpts1, inlier_mask, e_hat)
        if ret is None:
            err_t, err_R = np.inf, np.inf
            precision_all.append(torch.zeros(1,)[0])
            recall_all.append(torch.zeros(1,)[0])
            f_scores_all.append(torch.zeros(1,)[0])
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

    out_eval = {'error_t': err_ts,
                'error_R': err_Rs}

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    precision, recall, f_scores = np.mean(np.asarray(precision_all))*100, np.mean(np.asarray(recall_all))*100, np.mean(np.asarray(f_scores_all))*100

    print('Evaluation Results {} RANSAC (mean over {} pairs):'
          .format("with" if config.use_ransac else "without", len(test_loader)))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
    print('Prec\t Rec\t F1\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(precision, recall, f_scores))

    return aucs, precision, recall

if __name__ == '__main__':
    config, unparsed = get_config()
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    use_essential = True
    # config.use_ransac = False
    print("use ransac: {}".format(config.use_ransac))
    test(config, use_essential)
    # config.use_ransac = True
    # print("use ransac: {}".format(config.use_ransac))
    # test(config, use_essential)

    # change threshold
    # config.use_ransac = True
    start = -2.5
    end = 3.5
    step = 0.1
    thrs = np.arange(start, end + step, step)

    for thr in thrs:
        config.inlier_threshold = thr
        auc, p, c = test(config)
        with open('aucs_pr.txt', 'a') as f:
            f.write(str(auc) + '\n')
        with open('fs_pr.txt', 'a') as f:
            f.write(str(2 * p * c / (p + c + 1e-15)) + '\n')