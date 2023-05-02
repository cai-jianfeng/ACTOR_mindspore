# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: accuracy.py
@Date: 2023/4/27 20:27
@Author: caijianfeng
"""
import mindspore as ms


def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = ms.ops.zeros(num_labels, num_labels, dtype=ms.int64)
    with torch.no_grad():
    # TODO: mindspore 中如何显式不计算梯度
        for batch in motion_loader:
            batch_prob = classifier(batch["output_xyz"], lengths=batch["lengths"])
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1

    accuracy = ms.Tensor.trace(confusion)/ms.ops.reduce_sum(confusion)
    return accuracy.item(), confusion
