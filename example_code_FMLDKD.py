

from utils_general_FMLDKD import *

from utils_methods_FMLDKD import *

# Dataset initialization
data_path = 'Folder/'  # The folder to save Data & Model

n_client = 100

######MNIST
#data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
# data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
# data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
# unbalanced
# data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)
# model_name =  'mnist_2NN' # Model type




#####cifar100
# data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
# data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
# data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
# unbalanced
# data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)
# model_name =  'cifar100_LeNet'  # Model type



#####kvasir
#data_obj = DatasetObject(dataset='Kvasir', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
#data_obj = DatasetObject(dataset='Kvasir', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
# data_obj = DatasetObject(dataset='Kvasir', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
# model_name = 'Kvasir_CNN'  # Model type


####cifar10
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
#data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
# unbalanced
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)
model_name = 'cifar10_LeNet' # Model type



###
# Common hyperparameters
# com_amount = 600
com_amount = 1000
save_period = 200
weight_decay = 1e-3
batch_size = 50
# batch_size = 20
#act_prob = 1
act_prob = 0.15
suffix = model_name
lr_decay_per_round = 0.998

# Model function
model_func = lambda: client_model(model_name)
model_local_func = lambda: client_local_model(model_name)

init_model = model_func()
init_local_model = model_local_func()

# Initalise the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(37)
init_model = model_func()
init_local_model = model_local_func()
if not os.path.exists('%sModel/%sFMLDKD/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
    if not os.path.exists('%sModel/%sFMLDKD/' % (data_path, data_obj.name)):
        print("Create a new directory")
        os.mkdir('%sModel/%sFMLDKD/' % (data_path, data_obj.name))
    torch.save(init_model.state_dict(), '%sModel/%sFMLDKD/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

if not os.path.exists('%sModel/%sFMLDKD/%s_init_local_mdl.pt' % (data_path, data_obj.name, model_name)):
    torch.save(init_local_model.state_dict(),
               '%sModel/%sFMLDKD/%s_init_local_mdl.pt' % (data_path, data_obj.name, model_name))
else:
    # Load model
    init_local_model.load_state_dict(torch.load('%sModel/%sFMLDKD/%s_init_local_mdl.pt' % (data_path, data_obj.name, model_name)))


#
print('FedAvg_FMLDKD')

# epoch = 5
#
# learning_rate = 0.1
# print_per = 5
# a=1.0              #the coefficient of target class
# b=8.0            #the coefficient of non-target class
# alpha= 0.5
# beta = 0.5
# temperature=4.0
# [fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedAvg_FMLDKD(data_obj=data_obj,
#                                                                                                     act_prob=act_prob,
#                                                                                                     learning_rate=learning_rate,
#                                                                                                     batch_size=batch_size,
#                                                                                                     epoch=epoch,
#                                                                                                     com_amount=com_amount,
#                                                                                                     print_per=print_per,
#                                                                                                     weight_decay=weight_decay,
#                                                                                                     model_func=model_func,
#                                                                                                     model_local_func=model_local_func,
#                                                                                                     init_model=init_model,
#                                                                                                     init_local_model=init_local_model,
#                                                                                                     sch_step=1,
#                                                                                                     sch_gamma=1,
#                                                                                                     save_period=save_period,
#                                                                                                     suffix=suffix,
#                                                                                                     trial=False,
#                                                                                                     data_path=data_path,
#                                                                                                     lr_decay_per_round=lr_decay_per_round,
#                                                                                                         a=a,
#                                                                                                         b=b,
#                                                                                                         alpha=alpha,
#                                                                                                         beta=beta,
#                                                                                                         temperature=temperature)
#

#
print('FedProx_FMLDKD')

# epoch = 5
# learning_rate = 0.1
# print_per = 5
# mu = 1e-4
#
# a=1.0              #目标类损失参数
# b=8.0            #非目标类损失参数
# alpha= 0.5    #local: the coefficient of kl loss
# beta = 0.5     #us: the coefficient of kl loss
# temperature=4.0
#
# [fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedProx_FMLDKD(data_obj=data_obj,
#                                                                                                      act_prob=act_prob,                                                                                           learning_rate=learning_rate,
#                                                                                                      batch_size=batch_size,
#                                                                                                      epoch=epoch,
#                                                                                                      com_amount=com_amount,
#                                                                                                      print_per=print_per,
#                                                                                                      weight_decay=weight_decay,
#                                                                                                      model_func=model_func,
#                                                                                                      model_local_func = model_local_func,
#                                                                                                      init_model=init_model,
#                                                                                                     init_local_model = init_local_model,
#                                                                                                      sch_step=1,
#                                                                                                      sch_gamma=1,
#                                                                                                      save_period=save_period,
#                                                                                                      mu=mu,
#                                                                                                      suffix=suffix,
#                                                                                                      trial=False,
#                                                                                                      data_path=data_path,
#                                                                                                      lr_decay_per_round=lr_decay_per_round,                                               a=a,
#                                                                                                         b=b,
#                                                                                                         alpha=alpha,
#                                                                                                         beta=beta,
#
#                                                                                                         temperature=temperature)
#

print('FedDyn_FMLDKD')

# epoch = 5
# alpha_coef = 1e-2
# learning_rate = 0.1
# print_per = epoch // 2
#
# a=1.0
# b=8.0
# alpha= 0.5
# beta = 0.5
# temperature=4.0
#
# [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
#  trn_all_clt_perf, tst_all_clt_perf] = train_FedDyn_FMLDKD(data_obj=data_obj, act_prob=act_prob,
#                                                     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
#                                                     com_amount=com_amount, print_per=print_per,
#                                                     weight_decay=weight_decay,
#                                                     model_func=model_func, model_local_func=model_local_func,init_model=init_model, init_local_model=init_local_model,alpha_coef=alpha_coef,
#                                                     sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
#                                                     trial=False,
#                                                     data_path=data_path, lr_decay_per_round=lr_decay_per_round,a=a,
#                                                                                                         b=b,
#                                                                                                         alpha=alpha,
#                                                                                                         beta=beta,
#                                                                                                         temperature=temperature)

#####
print('FedDC_FMLDKD')

epoch = 5
alpha_coef = 1e-2

###500 clients
#alpha_coef = 0.05

#mnist
# alpha_coef = 0.1
learning_rate = 0.1
print_per = epoch // 2
a=1.0
b=8.0
alpha=0.3
beta = 0.3
temperature=4.0

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)

[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
 trn_all_clt_perf, tst_all_clt_perf] = train_FedDC_FMLDKD(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch,
                                                   learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
                                                   com_amount=com_amount, print_per=print_per,
                                                   weight_decay=weight_decay,
                                                   model_func=model_func, model_local_func=model_local_func,init_model=init_model, init_local_model=init_local_model,alpha_coef=alpha_coef,
                                                   sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
                                                   trial=False,
                                                   data_path=data_path, lr_decay_per_round=lr_decay_per_round,
                                                                                                        a=a,
                                                                                                        b=b,
                                                                                                        alpha=alpha,
                                                                                                        beta=beta,
                                                                                                        temperature=temperature)



exit(0)

# Plot results
plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount) + 1, tst_all_clt_perf[:com_amount, 1], label='FedDyn')
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount + 2])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('%s.pdf' % data_obj.name, dpi=1000, bbox_inches='tight')
plt.show()
