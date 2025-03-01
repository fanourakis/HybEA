import os
import time
import torch
import numpy as np
from structure_model.reader.kg_reader import KGDataReader
from structure_model.utils.loss_func import cross_entropy
from structure_model.Param import CSLS
from structure_model.reader.batching import prepare_batch_data
from structure_model.utils.tools import device
from structure_model.valid.evaluate_kbc import kbc_predict
from structure_model.utils.swa import swa
from structure_model.valid.test_task import entity_alignment_test
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from contiguous_params import ContiguousParams
import torch.nn.functional as F

def entity_alignment_train(args, my_model, logger):

    train_data_reader = KGDataReader(
        vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
        data_path=os.path.join(
            args.dataset_root_path, args.dataset, args.ea_train_triples_file
        ),
        batch_size=args.batch_size,
        is_training=True,
    )

    criterion = cross_entropy
    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)

    max_hits1, times = 0, 0
    is_relation = None
    one_hot_labels = None
    hits_1_list = []
    epoch_loss_list = []
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        epoch_loss = list()
        epoch_another_loss = list()
        my_model.train()
        # eval_hits1_performance, _ = entity_alignment_test(args, my_model, logger, csls=CSLS, valid=True)
        # _, _ = entity_alignment_test(args, my_model, logger, csls=0)
        # exit()

        for batch in train_data_reader.data_generator():
            mask_index = -1
            batch_data = prepare_batch_data(batch, -1, train_data_reader.mask_id)
            
            src_ids, input_mask, mask_label, mask_pos, mask_pos_2, r_flag = batch_data
            src_ids = torch.LongTensor(src_ids).to(device)
            input_mask = torch.LongTensor(input_mask).to(device)
            mask_label = torch.LongTensor(mask_label).to(device)
            mask_pos = torch.LongTensor(mask_pos).to(device)
            if mask_pos_2 is not None:
                mask_pos_2 = torch.LongTensor(mask_pos_2).to(device)
            if r_flag is not None:
                r_flag = torch.LongTensor(r_flag).to(device)

            fc_out, fc_out_other = my_model(
                src_ids,
                input_mask,
                mask_pos,
                mask_index=mask_index,
                mask_pos_2=mask_pos_2,
                r_flag=r_flag,
            )

            if one_hot_labels is None or one_hot_labels.shape[0] != mask_label.shape[0]:
                one_hot_labels = (
                    torch.zeros(mask_label.shape[0], args.vocab_size)
                    .to(device)
                    .scatter_(1, mask_label, 1)
                )
            else:
                one_hot_labels.fill_(0).scatter_(1, mask_label, 1)

            if is_relation is None or is_relation.shape[0] != mask_label.shape[0]:
                entity_indicator = torch.zeros(
                    mask_label.shape[0], args.vocab_size - args.num_relations
                ).to(device)
                relation_indicator = torch.ones(
                    mask_label.shape[0], args.num_relations
                ).to(device)
                is_relation = torch.cat((entity_indicator, relation_indicator), dim=-1)

            soft_labels = one_hot_labels * args.soft_label + (
                1.0 - one_hot_labels - is_relation
            ) * ((1.0 - args.soft_label) / (args.vocab_size - 1 - args.num_relations))
            soft_labels.requires_grad = False

            loss = criterion(fc_out, soft_labels)
            epoch_loss.append(loss.item())

            if fc_out_other is not None:
                loss_other = criterion(fc_out_other, soft_labels)
                loss_other *= args.addition_loss_w
                epoch_another_loss.append(loss_other.item())
                loss += loss_other

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        msg = "epoch: %d, epoch loss: %f, epoch another loss: %f, training time: %f" % (
            epoch,
            np.float(np.mean(epoch_loss)),
            np.float(np.mean(epoch_another_loss)),
            time.time() - start_time,
        )
        epoch_loss_list.append(np.mean(epoch_loss))
        print(msg)
        if epoch % args.eval_freq == 0:
            print("do valid")
            my_model.eval()
            with torch.no_grad():
                # test on validation
                eval_hits1_performance, _ = entity_alignment_test(args, my_model, logger, csls=CSLS, valid=True)
                hits_1_list.append(eval_hits1_performance)
            
                if eval_hits1_performance > max_hits1:
                    max_hits1 = eval_hits1_performance
                    times = 0
                else:
                    times += 1
                if times >= args.early_stop_max_times and epoch >= args.min_epochs:
                    print("early stop at this epoch")
                    break    
                
    _, _ = entity_alignment_test(args, my_model, logger, csls=0)
    eval_hits1_performance, export_sim_mat = entity_alignment_test(args, my_model, logger, csls=CSLS)
    return export_sim_mat[0], export_sim_mat[1], export_sim_mat[2], export_sim_mat[3], export_sim_mat[4], epoch_loss_list, hits_1_list
    logger.info("Finish training")
