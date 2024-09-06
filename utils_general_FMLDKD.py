import sys

import torch
from scipy import special

from utils_dataset import *
from utils_models import *

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_norm = 10


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].named_parameters()))

    param_0 = clnt_models[0].named_parameters()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def train_model_FedAvg_FMLDKD(model, local_model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch,
                              print_per,
                              weight_decay, dataset_name, a, b, alpha, beta, temperature, sch_step=1, sch_gamma=1):
    Softmax = nn.Softmax(dim=1)
    LogSoftmax = nn.LogSoftmax(dim=1)

    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    ######us
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("us Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    model.train()

    #######local
    optimizer_local = torch.optim.SGD(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    local_model.train();
    local_model = local_model.to(device)
    scheduler_local = torch.optim.lr_scheduler.StepLR(optimizer_local, step_size=sch_step, gamma=sch_gamma)
    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn_local, acc_trn_local = get_acc_loss(trn_x, trn_y, local_model, dataset_name, weight_decay)
    if print_test:
        loss_tst_local, acc_tst_local = get_acc_loss(tst_x, tst_y, local_model, dataset_name, 0)
        print("local Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn_local, loss_trn_local, acc_tst_local, loss_tst_local, scheduler_local.get_lr()[0]))
    local_model.train()

    for e in range(epoch):

        # Training
        epoch_loss_local = 0
        epoch_loss_us = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            y_pred_local = local_model(batch_x)

            y_pred_soft = Softmax(y_pred / temperature)
            y_pred_local_soft = Softmax(y_pred_local / temperature)

            #  ######################################改进部分##############################################
            # FML+DKD
            # us
            gt_mask_us = get_gt_mask(y_pred, batch_y)  # Feature extraction of target class
            other_mask_us = get_other_mask(y_pred, batch_y)  # Feature extraction of non-target class

            # local
            gt_mask_local = get_gt_mask(y_pred_local, batch_y)  # Feature extraction of target class

            other_mask_local = get_other_mask(y_pred_local, batch_y)  # Feature extraction of non-target class

            # the KD loss of target class for us private model
            tckd_loss_us = (
                    F.kl_div(torch.log(cat_mask(y_pred_soft, gt_mask_us, other_mask_us)),
                             (cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )
            # the KD loss of non-target class for us private model
            nckd_loss_us = (
                    F.kl_div(LogSoftmax(y_pred / temperature - 1000.0 * gt_mask_us),
                             Softmax((y_pred_local / temperature - 1000.0 * gt_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            # the KD loss of target class for local model
            tckd_loss_local = (
                    F.kl_div(torch.log(cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local)),
                             (cat_mask(y_pred_soft, gt_mask_us, other_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )
            # the KD loss od non-target class for local model
            nckd_loss_local = (
                    F.kl_div(LogSoftmax((y_pred_local / temperature - 1000.0 * gt_mask_local)),
                             Softmax((y_pred / temperature - 1000.0 * gt_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )
            # the kl loss of us private model
            kl_us = a * tckd_loss_us + b * nckd_loss_us
            # the kl loss of local model
            kl_local = a * tckd_loss_local + b * nckd_loss_local

            # CrossEntropyLoss of local model
            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]
            # CrossEntropyLoss of us private model
            ce_us = loss_fn(y_pred, batch_y.reshape(-1).long())
            ce_us = ce_us / list(batch_y.size())[0]

            ##local

            loss_local = ce_local + alpha * kl_local
            loss_local.requires_grad_(True)
            optimizer_local.zero_grad()
            loss_local.backward()
            torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer_local.step()
            epoch_loss_local += loss_local.item() * list(batch_y.size())[0]
            ###us

            loss_us = ce_us + beta * kl_us
            loss_us.requires_grad_(True)
            optimizer.zero_grad()
            loss_us.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss_us += loss_us.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss_local /= n_trn
            print("local Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_local, scheduler_local.get_lr()[0]))

            epoch_loss_us /= n_trn
            print("us Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_us, scheduler.get_lr()[0]))

            local_model.train()
            model.train()
        scheduler_local.step()
        scheduler.step()

    for local_params in local_model.parameters():
        local_params.requires_grad = False
    local_model.eval()

    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model, local_model


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def get_gt_mask(output, batch_y):
    batch_y = batch_y.reshape(-1)
    b = batch_y.unsqueeze(1)
    batch_y_unsqueeze = torch.tensor(b, dtype=torch.int64).detach()
    mask = torch.zeros_like(output).scatter_(1, batch_y_unsqueeze,
                                             1).bool()

    return mask


def get_other_mask(output, batch_y):
    batch_y = batch_y.reshape(-1)
    b = batch_y.unsqueeze(1)
    batch_y_unsqueeze = torch.tensor(b, dtype=torch.int64).detach()

    mask = torch.ones_like(output).scatter_(1, batch_y_unsqueeze, 0).bool()
    return mask


def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_prox_FMLDKD(model, local_model, cld_mdl_param, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size,
                            epoch, print_per,
                            weight_decay, dataset_name, a, b, alpha, beta, temperature, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    KL_Loss = nn.KLDivLoss(reduction='batchmean')
    Softmax = nn.Softmax(dim=1)
    LogSoftmax = nn.LogSoftmax(dim=1)

    ######us
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("us Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    model.train()

    #######local
    optimizer_local = torch.optim.SGD(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    local_model.train();
    local_model = local_model.to(device)
    scheduler_local = torch.optim.lr_scheduler.StepLR(optimizer_local, step_size=sch_step, gamma=sch_gamma)
    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn_local, acc_trn_local = get_acc_loss(trn_x, trn_y, local_model, dataset_name, weight_decay)
    if print_test:
        loss_tst_local, acc_tst_local = get_acc_loss(tst_x, tst_y, local_model, dataset_name, 0)
        print("local Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn_local, loss_trn_local, acc_tst_local, loss_tst_local, scheduler_local.get_lr()[0]))
    local_model.train()

    for e in range(epoch):
        # Training
        epoch_loss_local = 0
        epoch_loss_us = 0

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):

            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            y_pred_local = local_model(batch_x)

            y_pred_soft = Softmax(y_pred / temperature)
            y_pred_local_soft = Softmax(y_pred_local / temperature)

            # us
            gt_mask_us = get_gt_mask(y_pred, batch_y)
            other_mask_us = get_other_mask(y_pred, batch_y)

            # local
            gt_mask_local = get_gt_mask(y_pred_local, batch_y)
            other_mask_local = get_other_mask(y_pred_local, batch_y)

            tckd_loss_us = (
                    F.kl_div(torch.log(cat_mask(y_pred_soft, gt_mask_us, other_mask_us)),
                             (cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )

            nckd_loss_us = (
                    F.kl_div(LogSoftmax(y_pred / temperature - 1000.0 * gt_mask_us),
                             Softmax((y_pred_local / temperature - 1000.0 * gt_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            tckd_loss_local = (
                    F.kl_div(torch.log(cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local)),
                             (cat_mask(y_pred_soft, gt_mask_us, other_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )
            nckd_loss_local = (
                    F.kl_div(LogSoftmax((y_pred_local / temperature - 1000.0 * gt_mask_local)),
                             Softmax((y_pred / temperature - 1000.0 * gt_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            ###us
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = 0.0001 * torch.sum((local_par_list - cld_mdl_param) * (local_par_list - cld_mdl_param))

            loss = loss / list(batch_y.size())[0] + loss_algo
            kl_us = a * tckd_loss_us + b * nckd_loss_us
            loss_us = loss + beta * kl_us

            optimizer.zero_grad()
            loss_us.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss_us += loss_us.item() * list(batch_y.size())[0]

            ####local
            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]

            kl_local = a * tckd_loss_local + b * nckd_loss_local

            loss_local = ce_local + alpha * kl_local
            optimizer_local.zero_grad()
            loss_local.backward()
            torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer_local.step()
            epoch_loss_local += loss_local.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss_us /= n_trn
            epoch_loss_local /= n_trn

            print("us Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_us, scheduler.get_lr()[0]))
            print("local Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_local, scheduler_local.get_lr()[0]))

            local_model.train()
            model.train()
        scheduler_local.step()
        scheduler.step()

        # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    for local_params in local_model.parameters():
        local_params.requires_grad = False
    local_model.eval()

    return model, local_model


def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def train_model_FedDyn_FMLDKD(model, local_model, model_func, model_local_func, alpha_coef, avg_mdl_param,
                              hist_params_diff, trn_x, trn_y,
                              learning_rate, batch_size, epoch, print_per,
                              weight_decay, dataset_name, a, b, alpha, beta, temperature, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    Softmax = nn.Softmax(dim=1)
    LogSoftmax = nn.LogSoftmax(dim=1)

    #####us
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    model.train();
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    n_par = get_mdl_params([model_func()]).shape[1]

    #####local
    optimizer_local = torch.optim.SGD(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    local_model.train();
    local_model = local_model.to(device)
    scheduler_local = torch.optim.lr_scheduler.StepLR(optimizer_local, step_size=sch_step, gamma=sch_gamma)

    n_par_local = get_mdl_params([model_local_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        epoch_loss_local = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            y_pred_local = local_model(batch_x)

            y_pred_soft = Softmax(y_pred / temperature)
            y_pred_local_soft = Softmax(y_pred_local / temperature)

            ######FML+DKD
            # us
            gt_mask_us = get_gt_mask(y_pred, batch_y)
            other_mask_us = get_other_mask(y_pred, batch_y)

            # local
            gt_mask_local = get_gt_mask(y_pred_local, batch_y)
            other_mask_local = get_other_mask(y_pred_local, batch_y)

            tckd_loss_us = (
                    F.kl_div(torch.log(cat_mask(y_pred_soft, gt_mask_us, other_mask_us)),
                             (cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )

            nckd_loss_us = (
                    F.kl_div(LogSoftmax(y_pred / temperature - 1000.0 * gt_mask_us),
                             Softmax((y_pred_local / temperature - 1000.0 * gt_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            tckd_loss_local = (
                    F.kl_div(torch.log(cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local)),
                             (cat_mask(y_pred_soft, gt_mask_us, other_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )
            nckd_loss_local = (
                    F.kl_div(LogSoftmax((y_pred_local / temperature - 1000.0 * gt_mask_local)),
                             Softmax((y_pred / temperature - 1000.0 * gt_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            ####us
            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            kl_us = a * tckd_loss_us + b * nckd_loss_us

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss = loss_f_i + loss_algo

            # loss_us = beta * loss + (1 - beta) * kl_us
            loss_us = loss + beta * kl_us

            ###
            loss_us.requires_grad_(True)
            optimizer.zero_grad()
            loss_us.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss_us.item() * list(batch_y.size())[0]

            ####local
            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]

            kl_local = a * tckd_loss_local + b * nckd_loss_local

            # loss_local = alpha * ce_local + (1 - alpha) * kl_local
            loss_local = ce_local + alpha * kl_local

            loss_local.requires_grad_(True)
            optimizer_local.zero_grad()
            loss_local.backward()
            torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer_local.step()
            epoch_loss_local += loss_local.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            epoch_loss_local /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                params_local = get_mdl_params([local_model], n_par_local)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
                epoch_loss_local += (alpha_coef + weight_decay) / 2 * np.sum(params_local * params_local)

            print("us Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))
            print("local Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_local, scheduler_local.get_lr()[0]))

            local_model.train()
            model.train()
        scheduler_local.step()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    for local_params in local_model.parameters():
        local_params.requires_grad = False
    local_model.eval()

    return model, local_model

def train_model_FedDC_FMLDKD(model, local_model, model_func, model_local_func, alpha_py, local_update_last,
                             global_update_last, global_model_param, hist_i,
                             trn_x, trn_y,
                             learning_rate, batch_size, epoch, print_per,
                             weight_decay, dataset_name, a, b, alpha, beta, temperature, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32,
                                     device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    Softmax = nn.Softmax(dim=1)
    LogSoftmax = nn.LogSoftmax(dim=1)

    #####us
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    n_par = get_mdl_params([model_func()]).shape[1]

    #####local
    optimizer_local = torch.optim.SGD(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    local_model.train();
    local_model = local_model.to(device)
    scheduler_local = torch.optim.lr_scheduler.StepLR(optimizer_local, step_size=sch_step, gamma=sch_gamma)
    local_model.train()
    n_local_par = get_mdl_params([model_local_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss_local = 0
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            y_pred_local = local_model(batch_x)

            y_pred_soft = Softmax(y_pred / temperature)
            y_pred_local_soft = Softmax(y_pred_local / temperature)

            ###FML+DKD
            # us
            gt_mask_us = get_gt_mask(y_pred, batch_y)
            other_mask_us = get_other_mask(y_pred, batch_y)

            # local
            gt_mask_local = get_gt_mask(y_pred_local, batch_y)
            other_mask_local = get_other_mask(y_pred_local, batch_y)

            tckd_loss_us = (
                    F.kl_div(torch.log(cat_mask(y_pred_soft, gt_mask_us, other_mask_us)),
                             (cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )

            nckd_loss_us = (
                    F.kl_div(LogSoftmax(y_pred / temperature - 1000.0 * gt_mask_us),
                             Softmax((y_pred_local / temperature - 1000.0 * gt_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            tckd_loss_local = (
                    F.kl_div(torch.log(cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local)),
                             (cat_mask(y_pred_soft, gt_mask_us, other_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )
            nckd_loss_local = (
                    F.kl_div(LogSoftmax((y_pred_local / temperature - 1000.0 * gt_mask_local)),
                             Softmax((y_pred / temperature - 1000.0 * gt_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            kl_us = a * tckd_loss_us + b * nckd_loss_us
            kl_local = a * tckd_loss_local + b * nckd_loss_local

            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]

            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha_py / 2 * torch.sum((local_parameter - (global_model_param - hist_i)) * (
                    local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)

            loss = loss_f_i + loss_cp + loss_cg

            loss_us = loss + beta * kl_us
            optimizer.zero_grad()
            loss_us.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss_us.item() * list(batch_y.size())[0]

            ####local
            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]

            # loss_local = alpha * ce_local + (1 - alpha) * kl_local
            loss_local = ce_local + alpha * kl_local
            optimizer_local.zero_grad()
            loss_local.backward()
            torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer_local.step()
            epoch_loss_local += loss_local.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            epoch_loss_local /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                params_local = get_mdl_params([local_model], n_local_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)
                epoch_loss_local += (weight_decay) / 2 * np.sum(params_local * params_local)

            print("us Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))
            print("local Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_local, scheduler_local.get_lr()[0]))

            local_model.train()
            model.train()
        scheduler_local.step()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    for local_params in local_model.parameters():
        local_params.requires_grad = False
    local_model.eval()

    return model, local_model


def train_model_Feddisco_FMLDKD(model, local_model, model_func, model_local_func, alpha_py, local_update_last,
                             global_update_last, global_model_param, hist_i,
                             trn_x, trn_y,
                             learning_rate, batch_size, epoch, print_per,
                             weight_decay, dataset_name, a, b, alpha, beta, temperature, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32,
                                     device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    Softmax = nn.Softmax(dim=1)
    LogSoftmax = nn.LogSoftmax(dim=1)

    #####us
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    n_par = get_mdl_params([model_func()]).shape[1]

    #####local
    optimizer_local = torch.optim.SGD(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    local_model.train();
    local_model = local_model.to(device)
    scheduler_local = torch.optim.lr_scheduler.StepLR(optimizer_local, step_size=sch_step, gamma=sch_gamma)
    local_model.train()
    n_local_par = get_mdl_params([model_local_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss_local = 0
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            y_pred_local = local_model(batch_x)

            y_pred_soft = Softmax(y_pred / temperature)
            y_pred_local_soft = Softmax(y_pred_local / temperature)

            ###FML+DKD
            # us
            gt_mask_us = get_gt_mask(y_pred, batch_y)
            other_mask_us = get_other_mask(y_pred, batch_y)

            # local
            gt_mask_local = get_gt_mask(y_pred_local, batch_y)
            other_mask_local = get_other_mask(y_pred_local, batch_y)

            tckd_loss_us = (
                    F.kl_div(torch.log(cat_mask(y_pred_soft, gt_mask_us, other_mask_us)),
                             (cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )

            nckd_loss_us = (
                    F.kl_div(LogSoftmax(y_pred / temperature - 1000.0 * gt_mask_us),
                             Softmax((y_pred_local / temperature - 1000.0 * gt_mask_local).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            tckd_loss_local = (
                    F.kl_div(torch.log(cat_mask(y_pred_local_soft, gt_mask_local, other_mask_local)),
                             (cat_mask(y_pred_soft, gt_mask_us, other_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]

            )
            nckd_loss_local = (
                    F.kl_div(LogSoftmax((y_pred_local / temperature - 1000.0 * gt_mask_local)),
                             Softmax((y_pred / temperature - 1000.0 * gt_mask_us).detach()), reduction="sum")
                    * (temperature ** 2)
                    / batch_y.shape[0]
            )

            kl_us = a * tckd_loss_us + b * nckd_loss_us
            kl_local = a * tckd_loss_local + b * nckd_loss_local

            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]

            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha_py / 2 * torch.sum((local_parameter - (global_model_param - hist_i)) * (
                    local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)

            loss = loss_f_i + loss_cp + loss_cg

            loss_us = loss + beta * kl_us
            optimizer.zero_grad()
            loss_us.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss_us.item() * list(batch_y.size())[0]

            ####local
            ce_local = loss_fn(y_pred_local, batch_y.reshape(-1).long())
            ce_local = ce_local / list(batch_y.size())[0]

            # loss_local = alpha * ce_local + (1 - alpha) * kl_local
            loss_local = ce_local + alpha * kl_local
            optimizer_local.zero_grad()
            loss_local.backward()
            torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer_local.step()
            epoch_loss_local += loss_local.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            epoch_loss_local /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                params_local = get_mdl_params([local_model], n_local_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)
                epoch_loss_local += (weight_decay) / 2 * np.sum(params_local * params_local)

            print("us Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))
            print("local Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss_local, scheduler_local.get_lr()[0]))

            local_model.train()
            model.train()
        scheduler_local.step()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    for local_params in local_model.parameters():
        local_params.requires_grad = False
    local_model.eval()

    return model, local_model


def record_net_data_stats(clnt_y):
    net_cls_counts_npy = np.array([])
    n = np.concatenate([np.array(subset).reshape(-1) for subset in clnt_y])
    num_classes = int(n.max()) + 1

    for i in range(len(clnt_y)):
        unq, unq_cnt = np.unique(clnt_y[i], return_counts=True)
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[int(unq[i])] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
            (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1, num_classes))

    return net_cls_counts_npy


def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    print("effs", np.array(participation_clients).shape)
    local_distributions = client_cls_counts[np.array(participation_clients), :]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:, np.newaxis]

    if metric == 'cosine':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric == 'only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores > 0.9, 0.01, float('inf'))
    elif metric == 'l1':
        difference = np.linalg.norm(local_distributions - hypo_distribution, ord=1, axis=1)
    elif metric == 'l2':
        difference = np.linalg.norm(local_distributions - hypo_distribution, axis=1)
    elif metric == 'kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(
            difference)
    return difference


def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a * distribution_difference + b

    if np.sum(weight_tmp > 0) > 0:
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight < 0.0] = 0.0

    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight
