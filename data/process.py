import torch
import os
from torch_geometric.data import Data
from tqdm import tqdm, trange
from torch_scatter import scatter_add, scatter_mean

client_num = 13
mode = 'sum'
# mode = 'mean'
# mode = 'mean_sum'

for client_idx in range(1, client_num + 1):
    # 客户端数据的路径 
    data_path = os.path.join("CIKM22Competition", str(client_idx))
    train_path = os.path.join(data_path, "train.pt")
    val_path = os.path.join(data_path, "val.pt")
    test_path = os.path.join(data_path, "test.pt")

    # 载入数据
    print(f"client {client_idx}================================")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)

    num_train = len(train_data)
    num_val = len(val_data)
    num_test = len(test_data)

    # 输出数据统计量
    print('train samples: ', num_train)
    print('val samples: ', num_val)
    print('test samples: ', num_test)


    if train_data[0].edge_attr is not None:
        # 处理训练数据
        print("process train data ...")
        for i in range(num_train):
            edge_attr = train_data[i].edge_attr

            if mode == 'sum':
                node_cat_feat = scatter_add(edge_attr, train_data[i].edge_index[0], dim=0, dim_size=train_data[i].x.size(0))
            elif mode == 'mean':
                node_cat_feat = scatter_mean(edge_attr, train_data[i].edge_index[0], dim=0, dim_size=train_data[i].x.size(0))
            elif mode == 'mean_sum':
                node_cat_feat_1 = scatter_mean(edge_attr, train_data[i].edge_index[0], dim=0, dim_size=train_data[i].x.size(0))
                node_cat_feat_2 = scatter_add(edge_attr, train_data[i].edge_index[0], dim=0, dim_size=train_data[i].x.size(0))
                node_cat_feat = torch.cat([node_cat_feat_1, node_cat_feat_2], dim=-1)
            
            node_cat_feat = node_cat_feat.float()
            train_data[i].x = torch.cat([train_data[i].x, node_cat_feat], dim=-1)

        # 处理验证集数据
        print("process val data ...")
        for i in range(num_val):
            edge_attr = val_data[i].edge_attr

            if mode == 'sum':
                node_cat_feat = scatter_add(edge_attr, val_data[i].edge_index[0], dim=0, dim_size=val_data[i].x.size(0))
            elif mode == 'mean':
                node_cat_feat = scatter_mean(edge_attr, val_data[i].edge_index[0], dim=0, dim_size=val_data[i].x.size(0))
            elif mode == 'mean_sum':
                node_cat_feat_1 = scatter_mean(edge_attr, val_data[i].edge_index[0], dim=0, dim_size=val_data[i].x.size(0))
                node_cat_feat_2 = scatter_add(edge_attr, val_data[i].edge_index[0], dim=0, dim_size=val_data[i].x.size(0))
                node_cat_feat = torch.cat([node_cat_feat_1, node_cat_feat_2], dim=-1)
            
            node_cat_feat = node_cat_feat.float()
            val_data[i].x = torch.cat([val_data[i].x, node_cat_feat], dim=-1)

        # 处理测试集数据
        print("process test data ...")
        for i in range(num_test):
            edge_attr = test_data[i].edge_attr

            if mode == 'sum':
                node_cat_feat = scatter_add(edge_attr, test_data[i].edge_index[0], dim=0, dim_size=test_data[i].x.size(0))
            elif mode == 'mean':
                node_cat_feat = scatter_mean(edge_attr, test_data[i].edge_index[0], dim=0, dim_size=test_data[i].x.size(0))
            elif mode == 'mean_sum':
                node_cat_feat_1 = scatter_mean(edge_attr, test_data[i].edge_index[0], dim=0, dim_size=test_data[i].x.size(0))
                node_cat_feat_2 = scatter_add(edge_attr, test_data[i].edge_index[0], dim=0, dim_size=test_data[i].x.size(0))
                node_cat_feat = torch.cat([node_cat_feat_1, node_cat_feat_2], dim=-1)
            
            node_cat_feat = node_cat_feat.float()
            test_data[i].x = torch.cat([test_data[i].x, node_cat_feat], dim=-1)
        
        print("processed")

    if mode == 'sum':
        output_path = os.path.join("CIKM22Competition"+'_add', str(client_idx)+'_add')
    elif mode == 'mean':
        output_path = os.path.join("CIKM22Competition"+'_mean', str(client_idx)+'_mean')
    elif mode == 'mean_sum':
        output_path = os.path.join("CIKM22Competition"+'_mean_add', str(client_idx)+'_mean_add')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.save(train_data, os.path.join(output_path, "train.pt"))
    torch.save(val_data, os.path.join(output_path, "val.pt"))
    torch.save(test_data, os.path.join(output_path, "test.pt"))
