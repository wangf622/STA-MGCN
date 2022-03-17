import torch
import numpy as np
import time
import util
from engine import Trainer_1
import os
import json
import random

from get_models import get_stamgcn
import shutil
from tensorboardX import SummaryWriter
from tqdm import tqdm
# model repertition
from util import load_pickle
import argparse

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def train_main(args):
    device = torch.device(args.device)
    dataloader = util.load_dataset(dataset_dir=args.data_dir, batch_size=args.batch_size, test_batch_size=args.batch_size,
                                   valid_batch_size=args.batch_size, out_seq=2, in_seq=args.seq_length, days=args.days)
    scaler = dataloader['scaler']
    print(args)

    model_name = args.model_name

    if model_name ==  "STA_MGCN":
        model = get_stamgcn(args.config)

    else:
        raise SystemExit('Wrong name of model!')
    engine = Trainer_1(device, model, model_name=model_name, lrate=args.learning_rate, wdecay=args.weight_decay, decay=args.decay)

    print("start training...", flush=True)

    start_epoch = args.start_epoch
    params_path = args.save+"/"+args.model_name + "/" + "experiments_" + str(args.expid)


    if (start_epoch == 0) and (not os.path.exists(params_path)):  # 判断存放数据的地址是否存在
        os.makedirs(params_path)  # 创建相对应的地址
        print('create params directory %s' % params_path)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % params_path)
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % params_path)
    else:
        raise SystemExit('Wrong type of model!')

    sw = SummaryWriter(logdir=params_path, flush_secs=2)
    print(engine.model)  # 打印模型的相关信息

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in engine.model.state_dict():
        print(param_tensor, '\t', engine.model.state_dict()[param_tensor].size())
        total_param += np.prod(engine.model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)  # 模型的参数总数

    best_epoch = 0
    best_val_loss = np.inf
    his_loss = []
    val_time = []
    train_time = []
    count = 0
    if start_epoch > 0:
        param_filename = os.path.join(params_path + "/" + args.model_name + "_epoch_" + str(start_epoch) + "_" + ".pth")
        checkpoint = torch.load(param_filename)
        engine.model.load_state_dict(checkpoint['model'])
        engine.optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['min_val_loss']
        # engine.model.load_state_dict(torch.load(param_filename))
        engine.scheduler.load_state_dict(checkpoint['scheduler'])
        print('start epoch:', start_epoch)
        print('load weight from: ', param_filename)

    for i in range(start_epoch+1, args.epochs+1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        train_mae = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, _) in enumerate(tqdm(dataloader['train_loader'].get_iterator())):

            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)

            trainy = torch.Tensor(y).to(device)
            trainy = torch.squeeze(trainy, dim=1)  # B 1 N F -> B N F


            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            if iter % 100 == 0:
                log = 'Iter: {:03d}, ,Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()

        for iter, (x, y, _) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)  # B F N T
            testy = torch.Tensor(y).to(device)
            testy = torch.squeeze(testy, dim=1)   # B n 2

            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        engine.scheduler.step()  # epoch 不是必要的，pytorch 文档建议省略
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)


        sw.add_scalars('loss', {'train_loss_epoch': mtrain_loss, 'val_loss_epoch': mvalid_loss}, i)
        sw.add_scalars('train_metrics', {'train_mae': mtrain_mae, 'train_mape': mtrain_mape, 'train_rmse': mtrain_rmse}, i)
        sw.add_scalars('train_metrics', {'val_mae': mvalid_mae, 'val_mape': mvalid_mape, 'val_rmse': mvalid_rmse}, i)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, (t2 - t1)), flush=True)
        if mvalid_loss < best_val_loss:
            count = 0
            best_epoch = i
            best_val_loss = mvalid_loss
            # 保存 val_loss最小的结果
            state = {'model': engine.model.state_dict(), 'optimizer': engine.optimizer.state_dict(), 'min_val_loss': best_val_loss, 'epoch': i, 'scheduler': engine.scheduler.state_dict()}
            # state = {'model': engine.model.state_dict(), 'optimizer': engine.optimizer.state_dict(), 'epoch': i}

            print("save current best model's params to : %s " % params_path + "/"  + args.model_name + "_epoch_" + str(i) + "_" + ".pth")
            torch.save(state, params_path + "/" + args.model_name + "_epoch_" + str(i) + "_" + ".pth")

        else:
            count += 1
            print(f"no improve for {count} epochs")
        if count >= 50:
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


def test_main(args):
    device = torch.device(args.device)
    dataloader = util.load_dataset(dataset_dir=args.data_dir, batch_size=args.batch_size, test_batch_size=args.batch_size,
                                   valid_batch_size=args.batch_size, out_seq=2, in_seq=args.seq_length, days=args.days)
    scaler = dataloader['scaler']
    print(args)
    model_name = args.model_name
    if model_name == "STA_MGCN":
        model = get_stamgcn(args.config)

    else:
        raise SystemExit('Wrong name of model!')

    engine = Trainer_1(device, model, model_name=model_name, lrate=args.learning_rate, wdecay=args.weight_decay, decay=args.decay)

    print("start testing...", flush=True)  # flush = True 实现动画效果

    test_epoch = args.test_epoch

    # params_path = args.save + "/" + 'quarter_3/'+ args.model_name + "/" + "experiments_" + str(args.expid)
    params_path = args.save + "/" + args.model_name + "/" + "experiments_" + str(args.expid)

    checkpoint = torch.load(params_path + "/" + args.model_name + "_epoch_" + str(test_epoch) + "_" + ".pth")
    engine.model.load_state_dict(checkpoint['model'])
    engine.optimizer.load_state_dict(checkpoint['optimizer'])
    engine.scheduler.load_state_dict(checkpoint['scheduler'])

    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = torch.squeeze(realy, dim=1)  # b n 2

    for iter, (x, y, _) in enumerate(tqdm(dataloader['test_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)

        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = {'inflow': [], 'outflow': []}
    amape = {'inflow': [], 'outflow': []}
    armse = {'inflow': [], 'outflow': []}

    predictions = yhat
    targets = realy
    for i, category in enumerate(['inflow', 'outflow']):
        predictions_category = predictions[:, :, i]
        targets_category = targets[:, :, i]
        metrics_total = util.metric(predictions_category, targets_category)
        log = '{:s} On average over 2 horizons, Test MAE: {:.5f}, Test MAPE: {:.5f}, Test RMSE: {:.5f}'
        print(log.format(category, metrics_total[0], metrics_total[1], metrics_total[2]))
        amae[category].append(metrics_total[0])
        amape[category].append(metrics_total[1])
        armse[category].append(metrics_total[2])

    evaluate = {'mae': amae, "mape": amape, "RMSE": armse}
    np.savez(params_path + '\\epoch_' + str(test_epoch) + '_test_results', predictions.cpu().numpy())
    np.savez(params_path + '\\epoch_' + str(test_epoch) + '_evaluate_results', evaluate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="config/NYC_fhvtrip_STA-MGCN.json",
                        type=str, help="configuration file path")

    args = parser.parse_args()

    # os.makedirs(args.save, exist_ok=True)
    config = json.load(open(args.config, 'r'))['Training']
    data_config = json.load(open(args.config, 'r'))['Data']
    model_config = json.load(open(args.config, 'r'))['Model']

    args.device = model_config['device']
    args.start_epoch = config['start_epoch']
    args.test_epoch = config['test_epoch']
    args.expid = config['expid']
    args.data_dir = config['data_dir']
    args.od_data_dir = config['od_data_dir']
    args.epochs = config['epochs']
    args.model_name = config['model_name']

    args.adj_filename = data_config['adj_matrix_filename']
    args.seq_length = data_config['input_length']
    args.data_name = config['dataset_name']
    args.batch_size = config['batch_size']
    args.save = os.path.join('save_models/', args.data_name)
    args.days = 24
    args.learning_rate = config['learning_rate']
    args.dropout = config['dropout']
    args.weight_decay = config['weight_decay']
    args.decay = 0.97
    train_main(args)
    test_main(args)

