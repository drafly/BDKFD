from utils_general_FMLDKD import *

from utils_methods_FMLDKD import *
import argparse

parser = argparse.ArgumentParser()

# the setting of dataset
parser.add_argument('--dataset', type=str,
                    default='CIFAR10', help='change to mnist or CIFAR10 or CIFAR100')

parser.add_argument('--n_client', type=int,
                    default=500, help='the number of client')

parser.add_argument('--seed', type=int,
                    default=23, help='change to 20 while the rule is Drichlet')

parser.add_argument('--rule', type=str,
                    default='iid', help='Drichlet or iid')

parser.add_argument('--unbalanced_sgm', type=int,
                    default=0, help='change to 0.3 while the setting is unbalanced')

parser.add_argument('--rule_arg', type=float,
                    default=0, help='we set two Drichlet setting for 0.6 and 0.3')

parser.add_argument('--data_path', type=str,
                    default='Folder/', help='dataset location')

# the setting of model
parser.add_argument('--model_name', type=str,
                    default='cifar10_LeNet', help='mnist_2NN or cifar100_LeNet or cifar10_LeNet')

# the setting of training
parser.add_argument('--com_amount', type=int,
                    default=1000, help='mnist:600, cifar10 and cifar100:1000')

parser.add_argument('--save_period', type=int, default=200)

parser.add_argument('--weight_decay', type=float, default=1e-3)

parser.add_argument('--batch_size', type=int, default=50)

parser.add_argument('--act_prob', type=float, default=1, help='partial participants:0.15, full participants:1')

parser.add_argument('--lr_decay_per_round', type=float, default=0.998)

# some setting for single baseline
parser.add_argument('--epoch', type=int, default=5, help='the epoch of local training')

parser.add_argument('--learning_rate', type=float, default=0.1)

parser.add_argument('--print_per', type=int, default=5, help='avg and prox:5, dyn and dc:2')

parser.add_argument('--a', type=float, default=1.0, help='the coefficient of target class')

parser.add_argument('--b', type=float, default=8.0, help='the coefficient of non-target class')

parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('--beta', type=float, default=0.5)

parser.add_argument('--temperature', type=float, default=4.0)

parser.add_argument('--mu', type=float, default=1e-4, help='prox')

parser.add_argument('--alpha_coef', type=float, default=1e-2, help='dyn and dc and disco, 1e-2, mnist in DC and disco: 0.1')

parser.add_argument('--baselines', type=str, default='Feddisco_FMLDKD',
                    help='FedAvg_FMLDKD or FedProx_FMLDKD or FedDyn_FMLDKD or FedDC_FMLDKD or Feddisco_FMLDKD')

opt = parser.parse_args()

# Dataset
data_obj = DatasetObject(dataset=opt.dataset, n_client=opt.n_client, seed=opt.seed, rule=opt.rule,
                         rule_arg=opt.rule_arg,
                         unbalanced_sgm=opt.unbalanced_sgm, data_path=opt.data_path)

# Model function
model_func = lambda: client_model(opt.model_name)
model_local_func = lambda: client_local_model(opt.model_name)

init_model = model_func()
init_local_model = model_local_func()

# Initalise the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(37)
init_model = model_func()
init_local_model = model_local_func()
if not os.path.exists('%sModel/%sFMLDKD/%s_init_mdl.pt' % (opt.data_path, data_obj.name, opt.model_name)):
    if not os.path.exists('%sModel/%sFMLDKD/' % (opt.data_path, data_obj.name)):
        print("Create a new directory")
        os.mkdir('%sModel/%sFMLDKD/' % (opt.data_path, data_obj.name))
    torch.save(init_model.state_dict(),
               '%sModel/%sFMLDKD/%s_init_mdl.pt' % (opt.data_path, data_obj.name, opt.model_name))
else:
    # Load model
    init_model.load_state_dict(
        torch.load('%sModel/%sFMLDKD/%s_init_mdl.pt' % (opt.data_path, data_obj.name, opt.model_name)))

if not os.path.exists('%sModel/%sFMLDKD/%s_init_local_mdl.pt' % (opt.data_path, data_obj.name, opt.model_name)):
    torch.save(init_local_model.state_dict(),
               '%sModel/%sFMLDKD/%s_init_local_mdl.pt' % (opt.data_path, data_obj.name, opt.model_name))
else:
    # Load model
    init_local_model.load_state_dict(
        torch.load('%sModel/%sFMLDKD/%s_init_local_mdl.pt' % (opt.data_path, data_obj.name, opt.model_name)))

# baselines
if opt.baselines == 'FedAvg_FMLDKD':
    print("*******baseline is FedAvg_FMLDKD*******")
    [fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedAvg_FMLDKD(
        data_obj=data_obj,
        act_prob=opt.act_prob,
        learning_rate=opt.learning_rate,
        batch_size=opt.batch_size,
        epoch=opt.epoch,
        com_amount=opt.com_amount,
        print_per=opt.print_per,
        weight_decay=opt.weight_decay,
        model_func=model_func,
        model_local_func=model_local_func,
        init_model=init_model,
        init_local_model=init_local_model,
        sch_step=1,
        sch_gamma=1,
        save_period=opt.save_period,
        suffix=opt.model_name,
        trial=False,
        data_path=opt.data_path,
        lr_decay_per_round=opt.lr_decay_per_round,
        a=opt.a,
        b=opt.b,
        alpha=opt.alpha,
        beta=opt.beta,
        temperature=opt.temperature)

elif opt.baselines == 'FedProx_FMLDKD':
    print("*******baseline is FedProx_FMLDKD*******")
    [fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedProx_FMLDKD(
        data_obj=data_obj,
        act_prob=opt.act_prob, learning_rate=opt.learning_rate,
        batch_size=opt.batch_size,
        epoch=opt.epoch,
        com_amount=opt.com_amount,
        print_per=opt.print_per,
        weight_decay=opt.weight_decay,
        model_func=model_func,
        model_local_func=model_local_func,
        init_model=init_model,
        init_local_model=init_local_model,
        sch_step=1,
        sch_gamma=1,
        save_period=opt.save_period,
        mu=opt.mu,
        suffix=opt.model_name,
        trial=False,
        data_path=opt.data_path,
        lr_decay_per_round=opt.lr_decay_per_round, a=opt.a,
        b=opt.b,
        alpha=opt.alpha,
        beta=opt.beta,
        temperature=opt.temperature)

elif opt.baselines == 'FedDyn_FMLDKD':
    print("*******baseline is FedDyn_FMLDKD*******")
    [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
     trn_all_clt_perf, tst_all_clt_perf] = train_FedDyn_FMLDKD(data_obj=data_obj, act_prob=opt.act_prob,
                                                               learning_rate=opt.learning_rate,
                                                               batch_size=opt.batch_size,
                                                               epoch=opt.epoch,
                                                               com_amount=opt.com_amount, print_per=opt.print_per,
                                                               weight_decay=opt.weight_decay,
                                                               model_func=model_func, model_local_func=model_local_func,
                                                               init_model=init_model, init_local_model=init_local_model,
                                                               alpha_coef=opt.alpha_coef,
                                                               sch_step=1, sch_gamma=1, save_period=opt.save_period,
                                                               suffix=opt.model_name,
                                                               trial=False,
                                                               data_path=opt.data_path,
                                                               lr_decay_per_round=opt.lr_decay_per_round,
                                                               a=opt.a,
                                                               b=opt.b,
                                                               alpha=opt.alpha,
                                                               beta=opt.beta,
                                                               temperature=opt.temperature)

elif opt.baselines == 'Feddisco_FMLDKD':
    print("*******baseline is Feddisco_FMLDKD*******")
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / opt.n_client
    n_iter_per_epoch = np.ceil(n_data_per_client / opt.batch_size)
    n_minibatch = (opt.epoch * n_iter_per_epoch).astype(np.int64)

    [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
     trn_all_clt_perf, tst_all_clt_perf] = train_Feddisco_FMLDKD(data_obj=data_obj, act_prob=opt.act_prob,
                                                              n_minibatch=n_minibatch,
                                                              learning_rate=opt.learning_rate,
                                                              batch_size=opt.batch_size,
                                                              epoch=opt.epoch,
                                                              com_amount=opt.com_amount, print_per=2,
                                                              weight_decay=opt.weight_decay,
                                                              model_func=model_func, model_local_func=model_local_func,
                                                              init_model=init_model, init_local_model=init_local_model,
                                                              alpha_coef=opt.alpha_coef,
                                                              sch_step=1, sch_gamma=1, save_period=opt.save_period,
                                                              suffix=opt.model_name,
                                                              trial=False,
                                                              data_path=opt.data_path,
                                                              lr_decay_per_round=opt.lr_decay_per_round,
                                                              a=opt.a,
                                                              b=opt.b,
                                                              alpha=opt.alpha,
                                                              beta=opt.beta,
                                                              temperature=opt.temperature)

elif opt.baselines == 'FedDC_FMLDKD':
    print("*******baseline is FedDC_FMLDKD*******")
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / opt.n_client
    n_iter_per_epoch = np.ceil(n_data_per_client / opt.batch_size)
    n_minibatch = (opt.epoch * n_iter_per_epoch).astype(np.int64)

    [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
     trn_all_clt_perf, tst_all_clt_perf] = train_FedDC_FMLDKD(data_obj=data_obj, act_prob=opt.act_prob,
                                                              n_minibatch=n_minibatch,
                                                              learning_rate=opt.learning_rate,
                                                              batch_size=opt.batch_size,
                                                              epoch=opt.epoch,
                                                              com_amount=opt.com_amount, print_per=2,
                                                              weight_decay=opt.weight_decay,
                                                              model_func=model_func, model_local_func=model_local_func,
                                                              init_model=init_model, init_local_model=init_local_model,
                                                              alpha_coef=opt.alpha_coef,
                                                              sch_step=1, sch_gamma=1, save_period=opt.save_period,
                                                              suffix=opt.model_name,
                                                              trial=False,
                                                              data_path=opt.data_path,
                                                              lr_decay_per_round=opt.lr_decay_per_round,
                                                              a=opt.a,
                                                              b=opt.b,
                                                              alpha=opt.alpha,
                                                              beta=opt.beta,
                                                              temperature=opt.temperature)
exit(0)

# Plot results
# plt.figure(figsize=(6, 5))
# plt.plot(np.arange(opt.com_amount) + 1, tst_all_clt_perf[:opt.com_amount, 1], label='FedDyn')
# plt.ylabel('Test Accuracy', fontsize=16)
# plt.xlabel('Communication Rounds', fontsize=16)
# plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
# plt.grid()
# plt.xlim([0, com_amount + 2])
# plt.title(data_obj.name, fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.savefig('%s.pdf' % data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show()
