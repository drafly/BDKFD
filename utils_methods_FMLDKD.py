from utils_general_FMLDKD import *
from utils_libs import *
from utils_dataset import *
from utils_models import *
from tensorboardX import SummaryWriter


### Methods


def train_FedDC_FMLDKD(data_obj, act_prob, n_minibatch,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, model_local_func,init_model, init_local_model,alpha_coef,
                sch_step, sch_gamma, a, b, alpha, beta,
                     temperature,save_period,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):

    suffix = 'FedDC_' + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f_alpha%f_beta%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef,alpha, beta)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)  # 按列进行拼接
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])  # 每个客户的数据长度
    weight_list = weight_list / np.sum(weight_list) * n_clnt  # 每个客户的数据权重比*客户数
    if (not trial) and (not os.path.exists('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)  # 保存实例次数=迭代轮数/保存周期
    avg_ins_mdls = list(range(n_save_instances))
    avg_all_mdls = list(range(n_save_instances))
    avg_cld_mdls = list(range(n_save_instances))

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))

    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])  # 模型中参数的数量
    n_local_par = len(get_mdl_params([model_local_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')  # 给出一个空的所有客户端漂移量参数
    init_par_list = get_mdl_params([init_model], n_par)[0]  # 每个客户模型的初始模型参数
    init_local_par_list = get_mdl_params([init_local_model], n_local_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,-1)  # n_clnt X n_par # 每个客户端的模型参数列表
    clnt_params_list_local = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_local_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    clnt_models_local_old = list(range(n_clnt))
    saved_itr = -1

    ###
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32')  # including cloud state   梯度差异

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%sFMLDKD/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_ins = model_func()
                fed_ins.load_state_dict(
                    torch.load('%sModel/%s/%sFMLDKD/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_ins.eval()
                fed_ins = fed_ins.to(device)

                for params in fed_ins.parameters():
                    params.requires_grad = False

                avg_ins_mdls[saved_itr // save_period] = fed_ins

                ####
                fed_all = model_func()
                fed_all.load_state_dict(
                    torch.load('%sModel/%s/%sFMLDKD/all_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_all.eval()
                fed_all = fed_all.to(device)

                # Freeze model
                for params in fed_all.parameters():
                    params.requires_grad = False

                avg_all_mdls[saved_itr // save_period] = fed_all

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%sFMLDKD/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%sFMLDKD/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (not os.path.exists('%sModel/%sFMLDKD/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        clnt_models_local = list(range(n_clnt))
        # 第一轮本地的联邦平均
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]  # 第一轮服务端的梯度参数

            for clnt in range(0, n_clnt):
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(copy.deepcopy(dict(init_local_model.named_parameters())))

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]  # 第一轮服务端的梯度参数

            for clnt in n_clnt:
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_local/%dclint.pt'
                                                         % (data_path, data_obj.name, suffix, i + 1,clnt)))

        # 客户的参与采样（开始大轮循环）
        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            print("第%d大轮" % i)
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)  # 随机采样一个客户列表
                act_clients = act_list <= act_prob

                selected_clnts = np.sort(np.where(act_clients)[0])  # 采样客户的排序

                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            clnt_models_local_old = list(range(n_clnt))
            delta_g_sum = np.zeros(n_par)
            print("选择的客户数目%2d" % len(selected_clnts))  # *********************
            print(selected_clnts)  # **************************
            # 每个客户端开始第i大轮的训练
            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))  # 获取第一轮的全局模型参数


                clnt_models_local_old[clnt] = model_local_func().to(device)
                model_local = clnt_models_local_old[clnt]
                model_local.load_state_dict(copy.deepcopy(dict(clnt_models_local[clnt].named_parameters()))) # 获取上一轮的私有客户端参数

                for params in model.parameters():
                    params.requires_grad = True
                for params_local in model_local.parameters():
                    params_local.requires_grad = True

                local_update_last = state_gadient_diffs[clnt]  # delta theta_i （单个用户的上一轮跟上上轮梯度差）
                global_update_last = state_gadient_diffs[-1] / weight_list[
                    clnt]  # delta theta         所有客户跟上一轮的梯度差（上一轮 跟上上一轮的梯度差）的平均
                alpha_py = alpha_coef / weight_list[clnt]  # 约束因子（根据客户数据权重）
                hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                clnt_models[clnt],clnt_models_local[clnt] = train_model_FedDC_FMLDKD(model, clnt_models_local[clnt],model_func,model_local_func,alpha_py, local_update_last, global_update_last,
                                                      global_mdl, hist_i,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, print_per, weight_decay, data_obj.dataset,a,b,alpha,beta,temperature,
                                                      sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]  # 得到目前模型参数
                curr_model_local_par = get_mdl_params([clnt_models_local[clnt]], n_local_par)[0]

                delta_param_curr = curr_model_par - cld_mdl_param  ###########   delta S=S-W
                parameter_drifts[clnt] += delta_param_curr  ############  参数漂移量=h+delta
                beta = 1 / n_minibatch / learning_rate

                state_g = local_update_last - global_update_last + beta * (
                    -delta_param_curr)  # 本地梯度变化-全局梯度变化率的平均+模型更新梯度*约束项
                delta_g_cur = (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
                delta_g_sum += delta_g_cur
                state_gadient_diffs[clnt] = state_g
                clnt_params_list[clnt] = curr_model_par  #################### 客户端模型更新好的参数上传到列表
                clnt_params_list_local[clnt] = curr_model_local_par

                clnt_models_local[clnt] = set_client_from_params(model_local_func().to(device), curr_model_local_par)




            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)  # 这一轮参与的客户端更新好的梯度求平均
            delta_g_cur = 1 / n_clnt * delta_g_sum
            state_gadient_diffs[-1] += delta_g_cur

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)  ################  这一大轮的所有客户的梯度平均+漂移量

            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)  # 平均后的参数模型
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis=0))

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_sel_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_all_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_sel_clt_perf[i][0],
                                   'All clients': trn_all_clt_perf[i][0],
                                   'Current cloud': trn_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_sel_clt_perf[i][1],
                                   'All clients': trn_all_clt_perf[i][1],
                                   'Current cloud': trn_cur_cld_perf[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model_sel, data_obj.dataset, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0],
                                   'Current cloud':
                                       get_acc_loss(cent_x, cent_y, cur_cld_model, data_obj.dataset, weight_decay)[0]
                               }, i
                               )

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_sel_clt_perf[i] = [loss_tst, acc_tst]

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_all_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_sel_clt_perf[i][0],
                                   'All clients': tst_all_clt_perf[i][0],
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_sel_clt_perf[i][1],
                                   'All clients': tst_all_clt_perf[i][1],
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model_sel.state_dict(), '%sModel/%sFMLDKD/%s/ins_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%sFMLDKD/%s/all_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(cur_cld_model.state_dict(), '%sModel/%sFMLDKD/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_sel_clt_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_sel_clt_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_all_clt_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_all_clt_perf[:i + 1])

                np.save('%sModel/%sFMLDKD/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_cur_cld_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%sFMLDKD/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_tst_sel_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_trn_all_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_tst_all_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_trn_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%sFMLDKD/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_ins_mdls[i // save_period] = avg_model_sel
                avg_all_mdls[i // save_period] = all_model
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf


def train_FedAvg_FMLDKD(data_obj, act_prob, learning_rate, batch_size, epoch,
                     com_amount, print_per, weight_decay,
                     model_func, model_local_func, init_model, init_local_model, sch_step, sch_gamma, a, b, alpha, beta,
                     temperature,
                     save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedAvg_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_alpha%f_beta%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha, beta)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances));
    fed_mdls_all = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2));
    trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2));
    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    n_local_par = len(get_mdl_params([model_local_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    init_local_par_list = get_mdl_params([init_local_model], n_local_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_params_list_local = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_local_par_list.reshape(1, -1)  # n_clnt X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%sFMLDKD/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_all.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

                ############如果存在该轮的本地模型

                ###
                clnt_models_local = list(range(n_clnt))
                for clnt in n_clnt:

                    clnt_models_local[clnt] = model_local_func()
                    clnt_models_local[clnt].load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_local/%dclint.pt'
                                                                       % (
                                                                       data_path, data_obj.name, suffix, i + 1, clnt)))
                    clnt_models_local[clnt].eval()
                    clnt_models_local[clnt] = clnt_models_local[clnt].to(device)
                    # Freeze model
                    for params_local in clnt_models_local[clnt].parameters():
                        params_local.requires_grad = False

                if os.path.exists('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_perf_sel[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_all[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_all[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load(
                        '%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))
                    # clnt_params_list_local = np.load(
                    #     '%sModel/%s/%s/%d_clnt_params_list_local.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (not os.path.exists('%sModel/%sFMLDKD/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        clnt_models_local = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            for clnt in range(0, n_clnt):
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(copy.deepcopy(dict(init_local_model.named_parameters())))



        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_all.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            for clnt in n_clnt:
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_local/%dclint.pt'
                                                                   % (data_path, data_obj.name, suffix, i + 1, clnt)))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            clnt_models_local_old = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                clnt_models_local_old[clnt] = model_local_func().to(device)
                clnt_models_local_old[clnt].load_state_dict(copy.deepcopy(dict(clnt_models_local[clnt].named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                for params_local in clnt_models_local_old[clnt].parameters():
                    params_local.requires_grad = True

                clnt_models[clnt], clnt_models_local[clnt] = train_model_FedAvg_FMLDKD(clnt_models[clnt], clnt_models_local[clnt],
                                                                             trn_x, trn_y,
                                                                             tst_x, tst_y,
                                                                             learning_rate * (lr_decay_per_round ** i),
                                                                             batch_size, epoch, print_per,
                                                                             weight_decay,
                                                                             data_obj.dataset, a, b, alpha, beta,
                                                                             temperature, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                clnt_params_list_local[clnt] = get_mdl_params([clnt_models_local[clnt]], n_local_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0))
            all_model = set_client_from_params(model_func(),
                                               np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))
            clnt_models_local[clnt] = set_client_from_params(model_local_func().to(device), clnt_params_list_local[clnt])

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model, data_obj.dataset, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             all_model, data_obj.dataset, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                               }, i
                               )

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_perf_sel[i][0],
                                   'All clients': trn_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_perf_sel[i][1],
                                   'All clients': trn_perf_all[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0],
                                   'All clients': tst_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1],
                                   'All clients': tst_perf_all[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%sFMLDKD/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%sFMLDKD/%s/%dcom_all.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_sel[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%sFMLDKD/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_all[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_all[:i + 1])

                np.save('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%sFMLDKD/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%sFMLDKD/%s/%dcom_trn_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%sFMLDKD/%s/%dcom_tst_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all


def train_FedDyn_FMLDKD(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, model_local_func,init_model, init_local_model,alpha_coef,
                 sch_step, sch_gamma,  a, b, alpha, beta,
                     temperature,save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedDyn_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f_alpha%f_beta%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef, alpha, beta)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_ins_mdls = list(range(n_save_instances))  # Avg active clients
    avg_all_mdls = list(range(n_save_instances))  # Avg all clients
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))

    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    n_local_par = len(get_mdl_params([model_local_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')

    init_par_list = get_mdl_params([init_model], n_par)[0]
    init_local_par_list = get_mdl_params([init_local_model], n_local_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_params_list_local = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_local_par_list.reshape(1,-1)  # n_clnt X n_par

    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%sFMLDKD/%s/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_ins = model_func()
                fed_ins.load_state_dict(
                    torch.load('%sModel/%sFMLDKD/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_ins.eval()
                fed_ins = fed_ins.to(device)

                # Freeze model
                for params in fed_ins.parameters():
                    params.requires_grad = False

                avg_ins_mdls[saved_itr // save_period] = fed_ins

                ####
                fed_all = model_func()
                fed_all.load_state_dict(
                    torch.load('%sModel/%sFMLDKD/%s/all_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_all.eval()
                fed_all = fed_all.to(device)

                # Freeze model
                for params in fed_all.parameters():
                    params.requires_grad = False

                avg_all_mdls[saved_itr // save_period] = fed_all

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%sFMLDKD/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)


                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld





                clnt_models_local = list(range(n_clnt))
                for clnt in n_clnt:

                    clnt_models_local[clnt] = model_local_func()
                    clnt_models_local[clnt].load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_local/%dclint.pt'
                                                                       % (
                                                                       data_path, data_obj.name, suffix, i + 1, clnt)))
                    clnt_models_local[clnt].eval()
                    clnt_models_local[clnt] = clnt_models_local[clnt].to(device)


                    # Freeze model
                    for params_local in clnt_models_local[clnt].parameters():
                        params_local.requires_grad = False

                if os.path.exists(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%sFMLDKD/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (not os.path.exists('%sModel/%sFMLDKD/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        clnt_models_local = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]


            for clnt in range(0, n_clnt):
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(copy.deepcopy(dict(init_local_model.named_parameters())))

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

            for clnt in n_clnt:
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_local/%dclint.pt'
                                                         % (data_path, data_obj.name, suffix, i + 1,clnt)))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))
            clnt_models_local_old = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))

                clnt_models_local_old[clnt] = model_local_func().to(device)
                model_local = clnt_models_local_old[clnt]
                model_local.load_state_dict(
                    copy.deepcopy(dict(clnt_models_local[clnt].named_parameters())))  # 获取上一轮的私有客户端参数

                for params in model.parameters():
                    params.requires_grad = True
                for params_local in model_local.parameters():
                    params_local.requires_grad = True

                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt],clnt_models_local[clnt] = train_model_FedDyn_FMLDKD(model, clnt_models_local[clnt],model_func,model_local_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, a,b,alpha,beta,temperature,sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                curr_model_local_par = get_mdl_params([clnt_models_local[clnt]], n_local_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)  ####################

            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis=0))

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            clnt_params_list_local[clnt] = curr_model_local_par

            clnt_models_local[clnt] = set_client_from_params(model_local_func().to(device), curr_model_local_par)

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_sel_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_all_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_sel_clt_perf[i][0],
                                   'All clients': trn_all_clt_perf[i][0],
                                   'Current cloud': trn_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_sel_clt_perf[i][1],
                                   'All clients': trn_all_clt_perf[i][1],
                                   'Current cloud': trn_cur_cld_perf[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model_sel, data_obj.dataset, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0],
                                   'Current cloud':
                                       get_acc_loss(cent_x, cent_y, cur_cld_model, data_obj.dataset, weight_decay)[0]
                               }, i
                               )

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_sel_clt_perf[i] = [loss_tst, acc_tst]

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_all_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_sel_clt_perf[i][0],
                                   'All clients': tst_all_clt_perf[i][0],
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_sel_clt_perf[i][1],
                                   'All clients': tst_all_clt_perf[i][1],
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model_sel.state_dict(), '%sModel/%sFMLDKD/%s/ins_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%sFMLDKD/%s/all_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(cur_cld_model.state_dict(), '%sModel/%sFMLDKD/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_sel_clt_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_sel_clt_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_all_clt_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_all_clt_perf[:i + 1])

                np.save('%sModel/%sFMLDKD/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_cur_cld_perf[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%sFMLDKD/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_trn_sel_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_tst_sel_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_trn_all_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_tst_all_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_trn_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%sFMLDKD/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_ins_mdls[i // save_period] = avg_model_sel
                avg_all_mdls[i // save_period] = all_model
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf
def train_FedProx_FMLDKD(data_obj, act_prob, learning_rate, batch_size, epoch,
                  com_amount, print_per, weight_decay,
                  model_func, model_local_func, init_model,init_local_model, sch_step, sch_gamma, a, b, alpha, beta,
                     temperature,
                  save_period, mu, weight_uniform=False, suffix='', trial=True, data_path='', rand_seed=0,
                  lr_decay_per_round=1):
    suffix = 'FedProx_final_mu0.0001_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_alpha%f_beta%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha, beta)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed
    suffix += '_mu%f_WU%s' % (mu, weight_uniform)

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances));
    fed_mdls_all = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2));
    trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2));
    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    n_local_par = len(get_mdl_params([model_local_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    init_local_par_list = get_mdl_params([init_local_model], n_local_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_params_list_local = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_local_par_list.reshape(1,-1)  # n_clnt X n_par
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%sFMLDKD/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%sFMLDKD/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_all.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_perf_sel[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_all[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_all[:i + 1] = np.load(
                        '%sModel/%sFMLDKD/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load(
                        '%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (not os.path.exists('%sModel/%sFMLDKD/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        clnt_models_local = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([all_model], n_par)[0]

            for clnt in range(0, n_clnt):
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(copy.deepcopy(dict(init_local_model.named_parameters())))


        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_all.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))
            cld_mdl_param = get_mdl_params([all_model], n_par)[0]


            for clnt in n_clnt:
                clnt_models_local[clnt] = model_local_func().to(device)
                clnt_models_local[clnt].load_state_dict(torch.load('%sModel/%sFMLDKD/%s/%dcom_local/%dclint.pt'
                                                         % (data_path, data_obj.name, suffix, i + 1,clnt)))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness

            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            del clnt_models
            clnt_models = list(range(n_clnt))
            clnt_models_local_old = list(range(n_clnt))

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
                # Add regulariser during training
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

                clnt_models_local_old[clnt] = model_local_func().to(device)
                model_local = clnt_models_local_old[clnt]
                model_local.load_state_dict(
                    copy.deepcopy(dict(clnt_models_local[clnt].named_parameters())))  # 获取上一轮的私有客户端参数

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                for params_local in model_local.parameters():
                    params_local.requires_grad = True
                # local train
                clnt_models[clnt],clnt_models_local[clnt] = train_model_prox_FMLDKD(clnt_models[clnt],  clnt_models_local[clnt],cld_mdl_param_tensor, trn_x, trn_y,
                                                     tst_x, tst_y,
                                                     learning_rate * (lr_decay_per_round ** i),
                                                     batch_size, epoch, print_per,
                                                     weight_decay,
                                                     data_obj.dataset,a,b,alpha,beta,temperature, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                curr_model_local_par = get_mdl_params([clnt_models_local[clnt]], n_local_par)[0]
                clnt_params_list_local[clnt] = curr_model_local_par
                clnt_models_local[clnt] = set_client_from_params(model_local_func().to(device), curr_model_local_par)

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0))
            all_model = set_client_from_params(model_func(),
                                               np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))
            cld_mdl_param = np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0)
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model, data_obj.dataset, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             all_model, data_obj.dataset, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                               }, i
                               )

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_perf_sel[i][0],
                                   'All clients': trn_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_perf_sel[i][1],
                                   'All clients': trn_perf_all[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0],
                                   'All clients': tst_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1],
                                   'All clients': tst_perf_all[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%sFMLDKD/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%sFMLDKD/%s/%dcom_all.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_sel[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%sFMLDKD/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_all[:i + 1])
                np.save('%sModel/%sFMLDKD/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_all[:i + 1])

                np.save('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%sFMLDKD/%s/%dcom_trn_perf_sel.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%sFMLDKD/%s/%dcom_tst_perf_sel.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%sFMLDKD/%s/%dcom_trn_perf_all.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%sFMLDKD/%s/%dcom_tst_perf_all.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%sFMLDKD/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all