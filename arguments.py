import argparse

arg_p = argparse.ArgumentParser()
"args for MKL"
arg_p.add_argument('--data_path', type=str, default='./data/', help='path to data')
arg_p.add_argument('--OPname', type=str, default='./Points/optest.txt', help='name of origin point data file')
arg_p.add_argument('--SPname', type=str, default='./Points/sptest.txt', help='name of super point file')
arg_p.add_argument('--INpaint', type=str, default='./Points/inpaint.txt', help='name of inpaint point file')
arg_p.add_argument('--OUTpaint', type=str, default='./Points/outpaint.txt', help='name of outpaint point file')
arg_p.add_argument('--num_points', type=int, default=2000, help='number of points')
arg_p.add_argument('--num_neighbors', type=int, default=256, help='number of points')
arg_p.add_argument('--num_kernels', type=int, default=10, help='number of kernels')
arg_p.add_argument('--SpectralGraph', action='store_true', default=False, help='Whether to use spectral composition, otherwise use spatial coordinates')
arg_p.add_argument('--Spatialweight', type=float, default=0.5, help='Spatial weight')
arg_p.add_argument('--percenttrain', type=float, default=0.6, help='percent of train data')
arg_p.add_argument('--percentval', type=float, default=0.2, help='percent of val data')
arg_p.add_argument('--percenttest', type=float, default=0.2, help='percent of test data')
arg_p.add_argument('--sigmalist', type=list, default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], help='sigma list')
arg_p.add_argument('--Visualisation', action='store_true', default=False, help='Visualisation')
arg_p.add_argument('--Plot', action='store_true', default=True, help='Plot')

"args for GCN"
arg_p.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
arg_p.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
arg_p.add_argument('--seed', type=int, default=24, help='Random seed.')
arg_p.add_argument('--epochsMKL', type=int, default=2000,help='Number of epochs to train.')
arg_p.add_argument('--lr', type=float, default=0.001,help='Initial learning rate.')
arg_p.add_argument('--templr', type=float, default=0.001,help='Initial learning rate.')
arg_p.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
arg_p.add_argument('--hidden', type=int, default=1024,help='Number of hidden units.')
arg_p.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')


"args for PROGNN"
arg_p.add_argument('--debug', action='store_true',default=True, help='debug mode')
arg_p.add_argument('--only_gcn', action='store_true',default=False, help='test the performance of gcn without other components')
arg_p.add_argument('--attack', type=str, default='meta',choices=['no', 'meta', 'random', 'nettack'])
arg_p.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
arg_p.add_argument('--epochs', type=int,  default=100, help='Number of epochs to train.')
arg_p.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
arg_p.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
arg_p.add_argument('--gamma', type=float, default=0.1, help='weight of l2 norm')
arg_p.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
arg_p.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
arg_p.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
arg_p.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
arg_p.add_argument('--lr_adj', type=float, default=0.001, help='lr for training adj')
arg_p.add_argument('--symmetric', action='store_true', default=False,help='whether use symmetric matrix')




args = arg_p.parse_args()





