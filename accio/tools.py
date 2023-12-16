import copy
from typing import Dict, Union
import torch
from torch.utils.data import DataLoader, SequentialSampler, Sampler
import logging
from tqdm import tqdm


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        logging.info("%s cost time: %.3f s" % (func.__name__, time_spend))
        return result

    return func_wrapper


def get_inputs_process_func(args):
    def inner_func(batch):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if len(batch) == 5:
            if args.local_params.get("InfoRetention", False):
                inputs["new_input_ids"] = batch[4]
            else:
                inputs["inputs_embeds_noise"] = batch[4]
        return inputs

    return inner_func


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
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=kwargs.get("batch_size", 8), num_workers=kwargs.get("num_workers", 0))
    if func_kwargs is None:
        func_kwargs = {}
    outputs, dict_outputs = [], {}
    if copy_model:  # sometimes, use copy model to avoid changing the original model
        model = copy.deepcopy(model)
    original_mode = model.training
    model.train(training)
    with torch.no_grad():
        for step, batch_inputs in enumerate(tqdm(dataloader)):
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


def whiting(vecs: torch.Tensor, eps=1e-8):
    """进行白化处理
    x.shape = [num_samples, embedding_size]，
    最后的变换: y = (x - mu).dot( W^T )
    """
    mu = vecs.mean(dim=0, keepdims=True)  # [1, embedding_size]
    cov = torch.cov(vecs.T)  # [embedding_size, embedding_size]
    u, s, vh = torch.linalg.svd(cov)
    W = torch.mm(u, torch.diag(1.0 / (torch.sqrt(s) + eps)))  # [embedding_size, embedding_size]
    return (vecs - mu).mm(W)  # [num_samples, embedding_size]
