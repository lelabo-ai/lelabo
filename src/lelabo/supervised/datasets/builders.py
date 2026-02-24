from __future__ import annotations

from .registry import register_dataset


@register_dataset("iris")
def build_iris_dataset(**kwargs):
    from .tabular.iris import make_iris_dataset

    return make_iris_dataset(**kwargs)


@register_dataset("breast_cancer")
def build_breast_cancer_dataset(**kwargs):
    from .tabular.breast_cancer import make_breast_cancer_dataset

    return make_breast_cancer_dataset(**kwargs)


@register_dataset("mnist")
def build_mnist_dataset(**kwargs):
    from .vision.mnist import make_mnist_dataset

    return make_mnist_dataset(**kwargs)


@register_dataset("cifar10")
def build_cifar10_dataset(**kwargs):
    from .vision.cifar import make_cifar10_dataset

    return make_cifar10_dataset(**kwargs)


@register_dataset("cifar100")
def build_cifar100_dataset(**kwargs):
    from .vision.cifar import make_cifar100_dataset

    return make_cifar100_dataset(**kwargs)


@register_dataset("glue")
def build_glue_dataset(**kwargs):
    from .nlp.glue import make_glue_dataset

    return make_glue_dataset(**kwargs)
