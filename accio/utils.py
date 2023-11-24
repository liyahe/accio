
import datetime
import logging
import os
import pickle
import random
import sys
import time
from typing import Union, Dict
import copy
# can you see this?
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding

from .pca_torch import PCA_Torch

# logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def add_one(x):
    return x + 1



def indirect_calls(
    model,
    func_name: str,
    dataloader=None,
    dataset=None,
    prepare_inputs: callable = None,
    training=False,
    copy_model=False,
    func_kwargs=None,
    **kwargs,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if not hasattr(model, func_name):
        raise NotImplementedError
    if dataloader is None:
        assert dataset is not None
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=kwargs.get('batch_size', 8))
    if func_kwargs is None:
        func_kwargs = {}
    outputs, dict_outputs = [], {}
    if copy_model:  # sometimes, use copy model to avoid changing the original model
        model = copy.deepcopy(model)
    original_mode = model.training
    model.train(training)
    with torch.no_grad():
        for step, batch_inputs in enumerate(dataloader):
            # batch_x = {k: t.to(self.device) for k, t in batch_inputs.items() if k != 'y'}
            if prepare_inputs is not None:
                batch_inputs = prepare_inputs(batch_inputs)
            # batch_inputs.pop('labels')  # 从batch_inputs中删除labels
            batch_outputs = model.__getattribute__(func_name)(**batch_inputs, **func_kwargs)  # tensor or dict[tensor]
            if isinstance(batch_outputs, dict):
                for k_, v_ in batch_outputs.items():
                    if k_ not in dict_outputs:
                        dict_outputs[k_] = []
                    dict_outputs[k_].append(v_)
            else:
                outputs.append(batch_outputs)
    model.train(original_mode)  # restore model mode
    if len(dict_outputs) > 0:
        for k_, v_ in dict_outputs.items():
            dict_outputs[k_] = torch.cat(v_, dim=0)
        return dict_outputs
    outputs = torch.cat(outputs, dim=0)
    return outputs


def parse_arg_from_str(s: str, key: str, type_=float):
    """a simple func to parse arg from str, only support specific format, e.g. local_acc_knn30_KR0.5"""
    if key in s:
        res = s.split(key)[-1].split('_')[0]
        try:
            res = float(res)  # when res is a number
            return type_(res)
        except Exception as e:
            return None
    return None


def kernel_regression(x, x_support, y_support, h=1.0, kernel='gaussian'):
    """

    Args:
        x: (n, d)
        x_support: (m, d)
        y_support: (m, ) or (m, any_dim)
        h: the bandwidth of the kernel
        kernel: the kernel function, currently only support 'gaussian'

    Returns:

    """
    eps = 1e-7
    if kernel == 'gaussian':
        # if h == 'adaptive':
        #     h = (normed_euclidean_distance_func(x_support, x_support) ** 2).mean().sqrt() / 3.0 + eps  # (1, ), std
        K = lambda distance_mat: torch.exp(-distance_mat ** 2 / (2 * (h ** 2)))
    else:
        raise ValueError(f'Unknown kernel {kernel}')
    dist_mat = normed_euclidean_distance(x, x_support)  # (n, m)
    kernel_score = K(dist_mat)  # (n, m)
    kernel_score = kernel_score / (kernel_score.sum(dim=-1, keepdim=True) + eps)  # (n, m)
    not_valid = (kernel_score.sum(-1) - 1.0).abs() > 1e-3  # (n, )
    if not_valid.any():  # kernel score is too small and sum to 0
        logging.info(color(f'warning: kernel score is too small and sum to 0, '
                          f'{not_valid.sum()} / {not_valid.shape[0]}', 'red'))
        # this means each support point is too far from x, so we just use the nearest support point
        nearest_support_idx = torch.argmin(dist_mat, dim=1)  # (n, )
        for i in torch.arange(len(kernel_score), device=kernel_score.device)[not_valid]:
            kernel_score[i, :] = 0.0
            kernel_score[i, nearest_support_idx[i]] = 1.0
    assert ((kernel_score.sum(-1) - 1.0).abs() < 1e-3).all()  # check the kernel weight is valid
    return torch.matmul(kernel_score, y_support)  # (n, ) or (n, any_dim)


def get_result_dir(experiment_cfg: str, root_result_dir: str = 'result'):
    """
    Get the result directory, create it if not exist. Make sure the result directory is unique and will not
    cover the previous result.
    Args:
        root_result_dir: the root result directory, a str.
        experiment_cfg: the experiment configuration, a str, this will be used to generate the result directory name.

    Returns:

    """
    now_time = datetime.datetime.now().strftime('%Y-%m-%d')  # avoid the result dir name conflict
    if not os.path.exists(f'{root_result_dir}/{now_time}/{experiment_cfg}'):
        return f'{root_result_dir}/{now_time}/{experiment_cfg}'
    for i in range(1, 101):
        if not os.path.exists(f'{root_result_dir}/{now_time}#{i}/{experiment_cfg}'):
            return f'{root_result_dir}/{now_time}#{i}/{experiment_cfg}'
    raise ValueError(f'Failed to create result directory for {experiment_cfg}')  # 100个结果目录都已经存在，说明实验太多了


def config_logging(file_name: str, console_level: int = logging.DEBUG, file_level: int = logging.DEBUG,
                   output_to_file=True):  # 配置日志输出
    file_handler = logging.FileHandler(file_name, mode='a', encoding="utf8")
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(module)s.%(lineno)d:\t%(message)s',
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(message)s',  # [%(levelname)s]:
    ))
    console_handler.setLevel(console_level)
    
    if output_to_file:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler] if output_to_file else [console_handler],
    )


def cache_result(cache_path, overwrite=False, logger=None):
    """
    A decorator generator that tries to load the result from the cache file, if the cache file does not exist, run the
    function and save the result to the cache file.
    Args:
        cache_path: the path of the cache file.
        overwrite: whether to overwrite the cache file if it exists. default: False.
        logger: the logger to log the information, if None, print the information to the console.
    """

    def wrapper_generator(func):
        def wrapper(*args, **kwargs):
            success = False
            info, result = '', None
            if os.path.exists(cache_path) and not overwrite:
                start = time.time()
                try:
                    result = pickle.load(open(cache_path, 'rb'))
                    info = color(f"Load result of {func.__name__} from {cache_path} [took {time.time() - start:.2f} s]",
                                 'blue')
                    success = True
                except Exception as e:
                    info = color(f'Failed to load result of {func.__name__} from {cache_path}, Exception: {e}', 'red')
                    logging.info(info)
            if not success:
                start = time.time()
                result = func(*args, **kwargs)
                pickle.dump(result, open(cache_path, 'wb'))
                info = color(
                    f'Compute and save result of {func.__name__} at {cache_path}, [took {time.time() - start:.2f} s]',
                    'blue')
            logging.info(info)
            return result

        return wrapper

    return wrapper_generator


def allow_exception(logger=None):
    """
    A decorator that allows the decorated function to raise exception without interrupting the program.
    Args:
        logger: the logger to log the exception information, if None, print the information to the console.
    """

    def wrapper_generator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                info = color(f'Failed to execute function {func.__name__}: raise Exception \'{e}\' and ignore it.',
                             'red')
                logging.warning(info)

        return wrapper

    return wrapper_generator


def auto_data_type_convert(target_type):
    """
    A decorator that automatically converts the input data type to the target type. And convert the output of
    the decorated function to the input data type(and device).
    Note that is input args has multi data type, the first data type will be used as output data type.
    Args:
        target_type: must be in supported_data_type.
    """
    supported_data_type = (torch.Tensor, np.ndarray, list)
    assert target_type in supported_data_type

    def wrapper_generator(func):
        def wrapper(*args, **kwargs):
            args = list(args)
            dt, device = None, None
            for i in range(len(args)):
                if isinstance(args[i], supported_data_type):
                    dt = type(args[i]) if dt is None else dt
                    if isinstance(args[i], torch.Tensor) and device is None:
                        device = args[i].device
                    args[i] = convert_type(args[i], target_type)
            for k, v in kwargs.items():
                if isinstance(v, supported_data_type):
                    dt = type(v) if dt is None else dt
                    if isinstance(v, torch.Tensor) and device is None:
                        device = v.device
                    kwargs[k] = convert_type(v, target_type)
            output = func(*args, **kwargs)
            if device is None:
                device = torch.device('cpu')  # default device
            if isinstance(output, tuple):
                return tuple(convert_type(o, dt, device, strict=False) for o in output)
            return convert_type(output, dt, device, strict=False)

        return wrapper

    return wrapper_generator


def convert_type(data, dt, device=torch.device('cpu'), strict=True):  # 转换数据类型，以适应不同的输入输出
    """
    Convert the data type to the target type.
    Args:
        data: the data to be converted.
        dt: the target data type, must be in (torch.Tensor, np.ndarray, list).
        device: the device to store the data, only used when dt is torch.Tensor.
        strict: whether to strictly check the data type. default: True.

    Returns: the converted data.

    """
    if not strict:
        if dt not in (torch.Tensor, np.ndarray, list) or not isinstance(data, (torch.Tensor, np.ndarray, list)):
            return data  # 如果不严格检查，不支持的数据类型直接返回
    assert dt in (torch.Tensor, np.ndarray, list), f'unsupported data type: {dt}'
    assert isinstance(data, (torch.Tensor, np.ndarray, list)), f'unsupported data type: {type(data)}'
    if dt == np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)
    elif dt == torch.Tensor:
        if isinstance(data, list):
            data = torch.tensor(data).to(device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(device)
        # elif isinstance(data, torch.Tensor):
        #     data = data.to(device)  # note that tensor may be on different device
    elif dt == list:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            data = data.tolist()
    return data


def knn_graph(embeddings, k, metric='euclidean', include_self=True) -> torch.LongTensor:
    """计算knn图, 输入（num_examples, dim）的embeddings, 输出（num_examples, k）的knn indices"""
    device = embeddings.device
    k = k + 1 if not include_self else k
    num_examples = len(embeddings)
    if k > num_examples:
        raise ValueError(f'k({k}) must be equal or less than num_examples({num_examples})')
    if metric == 'cosine':
        distance_matrix = 1.0 - cosine_similarity(embeddings, embeddings)
    elif metric == 'dot':
        distance_matrix = -torch.mm(embeddings, embeddings.t())
    elif metric == 'euclidean':
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    else:
        raise ValueError(f'unsupported metric: {metric}')
    _, indices = topk_with_batch(distance_matrix, k=k, batch_size=10000, dim=-1, largest=False)
    # embeddings = embeddings.cpu().numpy()
    # nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=metric).fit(embeddings)
    # distances, indices = nbrs.kneighbors(embeddings)
    if not include_self:
        indices = indices[:, 1:]
    # indices = torch.LongTensor(indices).to(device)  # [num_examples, k]
    return indices.to(device)


def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def get_nonlinearity(act: str) -> torch.nn.Module:
    """get non-linearity function"""
    if act in ['relu', 'ReLU']:
        return torch.nn.ReLU()
    elif act == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act == 'tanh':
        return torch.nn.Tanh()
    elif act == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act == 'elu':
        return torch.nn.ELU()
    elif act == 'selu':
        return torch.nn.SELU()
    elif act == 'gelu':
        return torch.nn.GELU()
    elif act == 'none':
        return torch.nn.Identity()
    else:
        raise ValueError(f'unsupported activation function: {act}')


def get_cls_boundary_projection(cls_embeddings, cls_weight, cls_bias=None, normalize_boundary=True,
                                return_boundary=False):
    """计算分类边界法向量上的投影"""
    device = cls_embeddings.device
    if cls_bias is not None:
        cls_weight = torch.cat((cls_weight, cls_bias.view(-1, 1)), dim=-1)  # [num_classes, dim+1]
        cls_embeddings = torch.cat((cls_embeddings, torch.ones(len(cls_embeddings), 1, device=device)), dim=-1)
    num_classes, dim = cls_weight.shape
    cls_boundary = torch.zeros((int(num_classes * (num_classes - 1) / 2), dim)).to(device)  # 分类边界法向量
    t = 0
    for i in range(num_classes):
        for k in range(i + 1, num_classes):
            cls_boundary[t] = cls_weight[i] - cls_weight[k]
            t += 1
    if normalize_boundary:
        cls_boundary = cls_boundary / torch.norm(cls_boundary, dim=-1, keepdim=True)
    if return_boundary:
        return torch.matmul(cls_embeddings, cls_boundary.t()), cls_boundary
    return torch.matmul(cls_embeddings, cls_boundary.t())  # [num_examples, num_classes*(num_classes-1)/2]


def get_label_distribution_entropy(labels, num_classes):
    """计算分布的熵"""
    p = torch.bincount(labels, minlength=num_classes).float()
    p = p / p.sum()
    return -torch.sum(p * torch.log(p + 1e-7))


# set random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# def get_logger(name): # deprecated
#     """Return a logger with a default handler."""
#     logger = logging.getLogger(name)
#     logger.setLevel('DEBUG')  # 设置了这个才会把debug以上的输出
#     file_handler = logging.FileHandler(f'{name}.log', 'a', encoding='utf-8')  # 输出到文件(追加)
#     console_handler = logging.StreamHandler()  # 输出到控制台
#     file_handler.setLevel(logging.INFO)
#     console_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#     return logger


def k_means_clustering(embedding, num_clusters, device=torch.device('cpu')):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(embedding.cpu().numpy())
    return torch.from_numpy(kmeans.labels_).to(device), torch.from_numpy(kmeans.cluster_centers_).to(device)


@auto_data_type_convert(target_type=np.ndarray)  # the function take ndarray as input
def clustering(embeddings, num_clusters=2, method='kmeans', eps=0.5, min_samples=5,
               affinity='euclidean', linkage='ward', n_init=10):
    print(f'clustering {embeddings.shape} shape data to {num_clusters} clusters with {method} method...')
    # device = embeddings.device if isinstance(embeddings, torch.Tensor) else torch.device('cpu')
    # dt = type(embeddings)
    # embeddings = convert_type(embeddings, np.ndarray)
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=n_init).fit(embeddings)
        return kmeans.labels_, kmeans.cluster_centers_
    elif method == 'agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=num_clusters, affinity=affinity, linkage=linkage).fit(
            embeddings)
        return agglomerative.labels_, None
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        return dbscan.labels_, None
    else:
        raise NotImplementedError


def k_means_plus_plus(embeddings, num_clusters):
    num_u = len(embeddings)
    if num_u < num_clusters:
        raise ValueError(f'num_clusters ({num_clusters}) should be less than num_embeddings ({num_u})')
    cluster_centers = []
    cluster_idx = []
    while len(cluster_idx) < num_clusters:
        if len(cluster_centers) == 0:
            idx = torch.multinomial(torch.ones(num_u), 1, replacement=False)[0].item()
            cluster_centers.append(embeddings[idx])
            cluster_idx.append(idx)
        else:
            dist = torch.cdist(embeddings, torch.stack(cluster_centers, dim=0))
            dist = torch.min(dist, dim=1)[0]
            dist = dist ** 2
            dist = dist / dist.sum()
            idx = torch.multinomial(dist, 1, replacement=False)[0].item()
            del dist
            cluster_centers.append(embeddings[idx])
            cluster_idx.append(idx)
        cluster_idx = list(set(cluster_idx))
    cluster_idx = list(set(cluster_idx))
    if not len(cluster_idx) == num_clusters:
        raise ValueError('k-means++ error: cluster center idx len({}) mismatch with num_cluster({})'.format(
            len(cluster_idx), num_clusters))
    return torch.tensor(cluster_idx, device=embeddings.device)


def k_center_greedy(embeddings, num_clusters):
    cluster_centers = []
    cluster_idx = []
    while (len(set(cluster_idx)) < num_clusters):
        if len(cluster_idx) == 0:
            idx = torch.multinomial(torch.ones(len(embeddings)), 1, replacement=False)[0].item()
            cluster_centers.append(embeddings[idx])
            cluster_idx.append(idx)
        else:
            dist = torch.cdist(embeddings, torch.stack(cluster_centers, dim=0))
            dist = torch.min(dist, dim=1)[0]
            idx = torch.argmax(dist).item()
            cluster_centers.append(embeddings[idx])
            cluster_idx.append(idx)
    cluster_idx = torch.tensor(cluster_idx, device=embeddings.device)
    assert len(set(cluster_idx)) == num_clusters
    return cluster_idx


def pca(x, n_components, return_pca=False, torch_version=False):
    device = x.device
    if torch_version:  # torch version, in order to use GPU
        pca_ = PCA_Torch(n_components=n_components)
        pca_.fit(x)
        # x_ = (x - x.mean(0)) @ pca_.components_.T  # 与pca_.transform(x)等价
        x = pca_.transform(x)
        if return_pca:
            return x, pca_
        return x
    x = x.cpu().numpy()
    pca_ = PCA(n_components=n_components)
    pca_.fit(x)
    # x_ = (x - x.mean(0)) @ pca_.components_.T  # 与pca_.transform(x)等价
    x = torch.tensor(pca_.transform(x), device=device)
    if return_pca:
        return x, pca_
    return x


@auto_data_type_convert(target_type=np.ndarray)  # the function take ndarray as input
def manifold_map(x, algorithm='tsne', n_components=2):
    print('{} mapping from ({},{}) to ({},{})...'.format(algorithm, x.shape[0], x.shape[1], x.shape[0], n_components))
    if algorithm == 'tsne':
        mani_map = TSNE(n_components=n_components)
    elif algorithm == 'isomap':
        mani_map = Isomap(n_components=n_components)
    elif algorithm == 'mds':
        mani_map = MDS(n_components=n_components)
    elif algorithm == 'lle':
        mani_map = LocallyLinearEmbedding(n_components=n_components)
    elif algorithm == 'spectralEmbedding':
        mani_map = SpectralEmbedding(n_components=n_components)
    else:
        raise ValueError(f'unknown algorithm {algorithm}')
    # device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
    # dt = type(x)
    # if isinstance(x, torch.Tensor):
    #     x = x.detach().cpu().numpy()
    # x = convert_type(x, np.ndarray)
    # x = convert_type(mani_map.fit_transform(x), dt, device)  # (n_samples, n_components) 返回原本的数据类型和设备
    return mani_map.fit_transform(x)


def normed_euclidean_distance(feat1, feat2):
    # Normalized Euclidean Distance
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Euclidean Distance
    feat1, feat2 = F.normalize(feat1), F.normalize(feat2)
    return torch.cdist(feat1, feat2, p=2)
    # feat_matmul = torch.matmul(feat1, feat2.t())
    # distance = torch.ones_like(feat_matmul) - feat_matmul
    # distance = distance * 2
    # return distance.clamp(1e-10).sqrt()


def mahalanobis_distance(x, y, cov):
    # mahalanobis_distance
    # x: N * Dim
    # y: M * Dim
    # cov: Dim * Dim # 注意，这里的cov不是协方差矩阵
    # out:   N * M mahalanobis_distance Distance
    cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6
    return ((x.mm(cov) * x).sum(-1, keepdims=True) + (y.mm(cov) * y).sum(-1, keepdims=True).T - 2 * x.mm(cov).mm(
        y.t())).clamp(1e-6).sqrt()


def jaccard_similarity(set_list):
    """计算多个集合的jaccard相似度"""
    if len(set_list) == 1:
        return 1
    else:
        xs, xt = set(set_list[0]), set(set_list[0])
        for (i, s) in enumerate(set_list):
            if i != 0:
                ss = set(s)
                xs = xs & ss
                xt = xt | ss
        return len(xs) / len(xt)


# from transformers import set_seed
#
# set_seed(1)
# x = torch.rand(15, 1024)
# y = torch.rand(25, 1024)
# wd = wasserstein_distance(x, y)
# print(wd)


def visualize(embeddings, labels, title='default', stress='zero'):
    xx = manifold_map(embeddings, algorithm='tsne')
    # pickle.dump(xx, open('result_saved/mds-48-19-maskEmb.pkl', 'wb'))
    plt.clf()
    if stress == 'zero':
        plt.scatter(xx[:, 0], xx[:, 1], c=labels, cmap=plt.cm.Spectral, s=[20 if i == 0 else 0.5 for i in labels])
    else:
        plt.scatter(xx[:, 0], xx[:, 1], c=labels, cmap=plt.cm.Spectral, s=[10 if i != 0 else 0.5 for i in labels])
    plt.title(title)
    print('title:', title)
    plt.savefig(f'figs/visualize/{title}.png')
    plt.show()


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def topk_with_batch(dist_mat, k: int, batch_size: int, **kwargs):  # in order to save memory
    n = dist_mat.shape[0]
    device = dist_mat.device
    top_k_dist, top_k_indices = torch.zeros((n, k), device=device), torch.zeros((n, k), device=device, dtype=torch.long)
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        top_k_dist[start:end], top_k_indices[start:end] = torch.topk(dist_mat[start:end], k, **kwargs)
        start = end
    return top_k_dist, top_k_indices


def cosine_similarity(x, y):
    # x: (N1, d), y: (N2, d), return: (N1, N2)
    x = x.view(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])
    return x.mm(y.t()) / (x.norm(dim=1).unsqueeze(1) * y.norm(dim=1).unsqueeze(0))


@auto_data_type_convert(target_type=torch.Tensor)
def kl_divergence(p, q):
    # p: (N1, c), q: (N2, c), return: (N1, N2)
    p = p.view(-1, p.shape[-1])
    q = q.view(-1, q.shape[-1])
    return (p * (p + 1e-7).log()).sum(-1, keepdims=True) - p.mm((q + 1e-7).log().t())


def get_one_hot(labels, num_classes=None):
    """Convert an iterable of indices to one-hot encoded labels."""
    if not isinstance(labels, torch.Tensor):
        raise ValueError('labels must be a torch.Tensor')
    if num_classes is None:
        num_classes = max(labels) + 1
    labels = labels.view(-1, 1)
    return torch.zeros(len(labels), num_classes, dtype=torch.float32, device=labels.device).scatter_(1, labels, 1)


def describe_model(model, input_size):
    from torchsummary import summary
    assert isinstance(model, torch.nn.Module)
    print(f'------------------------------------{model.__class__.__name__}------------------------------------')
    summary(model, input_size=input_size, device='cpu', batch_size=1)


@auto_data_type_convert(target_type=torch.Tensor)  # convert input args into torch.Tensor
def describe_statistic_info(data, name='data'):
    # if not isinstance(data, torch.Tensor):
    #     if isinstance(data, np.ndarray):
    #         data = torch.from_numpy(data)
    #     else:
    #         raise ValueError('data must be torch.Tensor')
    data = data.float()
    print(color('------------------------------------{}------------------------------------'.format(name)))
    print(f'shape: {data.shape}')
    data = data.reshape(-1)
    print(f'min: {torch.min(data).item() :.4f} | max: {torch.max(data).item() :.4f}')
    print(
        f'mean: {torch.mean(data).item() :.4f} | std: {torch.std(data).item() :.4f} | median: {torch.median(data).item() :.4f}')
    distribution = torch.histc(data, bins=10)
    distribution = (distribution / torch.sum(distribution) * 100).tolist()
    key = torch.linspace(torch.min(data).item(), torch.max(data).item(), steps=11).tolist()
    res = ''.join([f' {key[i] :.2f} : {distribution[i] :.2f}% |' for i in range(len(key) - 1)])
    print(f'distribution: {res}')
    print(color('------------------------------------------------------------------------' + '-' * len(name)))
    del data, distribution, key, res


def color(text, color='purple'):  # or \033[32m
    color2code = {'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m', 'blue': '\033[34m',
                  'purple': '\033[35m', 'cyan': '\033[36m', 'white': '\033[37m', 'black': '\033[30m'}
    return color2code[color] + text + "\033[0m"


def plot_func_deprecated(x, y, xlabel='x', ylabel='y', title='default', label=None, type='line', c=None, save=False,
                         y_std=None):
    # print(color('calling plot_func', 'blue') + f': algorithm={algorithm}, title={title}, save={save}')
    plt.clf()
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape == y.shape, f'x.shape={x.shape}, y.shape={y.shape}'
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        c = c.reshape(1, -1) if c is not None else None
        y_std = y_std.reshape(1, -1) if y_std is not None else None
    if type == 'line':
        for r in range(y.shape[0]):
            plt.plot(x[r], y[r], label=label[r] if label is not None else None)
            if y_std is not None:
                plt.fill_between(x[r], y[r] - y_std[r], y[r] + y_std[r], alpha=0.2)
    elif type == 'scatter':
        for r in range(y.shape[0]):
            plt.scatter(x[r], y[r], c=c[r] if c is not None else None, cmap=plt.cm.Spectral)
    else:
        raise ValueError('algorithm must be line or scatter')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=4)
    if save:
        plt.savefig(f'figs/analyse/{title}.png')
        print(color('Save as figs/analyse/{}.png'.format(title), 'blue'))
    else:
        plt.show()


def any_to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        try:
            data = np.array(data)
        except Exception as _:
            raise ValueError('unknown data algorithm', type(data))
    return data


# better version
def plot_func(x: dict, y: dict = None, title=None, type='line', c=None, y_std=None, save=False):
    # print(color('calling plot_func', 'blue') + f': algorithm={algorithm}, title={title}, save={save}')
    assert isinstance(x, dict) and (y is None or isinstance(y, dict))
    plt.clf()
    assert len(x.keys()) == 1, 'x只能有一个key'
    xlabel = list(x.keys())[0]  # 横轴的key
    x_ = any_to_numpy(x[xlabel])

    if type == 'line':
        assert y is not None and len(y.keys()) > 0, 'line时纵轴至少有一个key'
        for ylable in y.keys():
            y_ = any_to_numpy(y[ylable])
            plt.plot(x_, y_, label=ylable)
            if y_std is not None and ylable in y_std.keys():
                y_std_ = any_to_numpy(y_std[ylable])
                plt.fill_between(x_, y_ - y_std_, y_ + y_std_, alpha=0.2)
    elif type == 'scatter':
        if y is not None:
            assert len(y.keys()) == 1, 'scatter时纵轴至多有一个key'
            y_ = any_to_numpy(y[list(y.keys())[0]])
            x_ = np.stack([x_, y_], axis=-1)
        assert x_.shape[-1] == 2, 'scatter只支持二维散点图，x的最后一维必须为2'
        c_ = any_to_numpy(c) if c is not None else None  # 不同颜色的点
        psc = plt.scatter(x_[:, 0], x_[:, 1], c=c_ if c_ is not None else None, cmap=plt.cm.Spectral, s=0.5, )
    else:
        raise ValueError('algorithm must be line or scatter')
    if title is None:
        title = f'{xlabel} vs {list(y.keys())[0]}' if y is not None else f'{xlabel}'
    plt.title(title)
    plt.xlabel(xlabel)
    if type == 'scatter':
        if c is not None:
            plt.legend(*psc.legend_elements(), loc=4)
    else:
        plt.legend(loc=4)
    if save:
        os.makedirs('figs/analyse', exist_ok=True)
        plt.savefig(f'figs/analyse/{title}.png')
        print(color('Save as figs/analyse/{}.png'.format(title), 'blue'))
    plt.show()


def power_law_distribution(size, p=2, lower=0., upper=1.):  # 与np.random.power(p)等价，但是可以指定范围，power默认[0,1]
    # 生成power law分布的随机数，p越大，分布越集中向 upper
    # p(x) = x^p / Z, 其中 Z = int_{l}^{u} x^p dx = (u ^ (1 + p) - l ^ (1 + p)) / (1 + p)
    # 则 pdf(x) =  (x^(p+1) - l^(p+1)) / (Z * (1 + p))
    # 则 pdf-1(x) = (x * Z * (p + 1) + l^(p + 1)) ^ (1 / (1 + p))
    # = (x * (u ^ (p + 1) - l ^ (p + 1)) + l ^ (p + 1)) ^ (1 / (1 + p))
    assert p > 0 and upper > lower >= 0, f'p={p}, upper={upper}, lower={lower}'
    r = np.random.random(size=size)
    return ((upper ** (p + 1) - lower ** (p + 1)) * r + lower ** (p + 1)) ** (1.0 / (p + 1))

# describe_statistic_info(torch.from_numpy(power_law_distribution(100000, p=4, lower=0., upper=1.)))
# describe_statistic_info(torch.from_numpy(np.random.power(4, 100000)))

# # test
# import numpy as np
#
# x = np.array([1, 2, 3, 4, 5] * 3).reshape(3, 5)
# y = np.random.randn(x.shape[0], x.shape[1])
# plot_func(x, y, 'x', 'y', 'test', save=True)
