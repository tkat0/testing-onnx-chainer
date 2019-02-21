import numpy
from onnx import checker
import chainer
from chainer import links as L
from chainer import functions as F
import onnx_chainer


class Model(chainer.Chain):

    def __init__(self, use_bn=False):
        super(Model, self).__init__()

        self._use_bn = use_bn

        with self.init_scope():
            self.conv = L.Convolution2D(None, 32, ksize=3, stride=1)
            if self._use_bn:
                self.bn = L.BatchNormalization(32)

    def forward(self, x):
        h = self.conv(x)

        if self._use_bn:
            h = self.bn(h)

        return {
            'out1': F.sigmoid(h),
            'out2': F.sigmoid(h)
        }


def main():
    x = numpy.zeros((1, 3, 32, 32), dtype=numpy.float32)

    onnx_chainer.export(Model(use_bn=True), x, filename='output/C_.onnx')
    onnx_chainer.export(Model(use_bn=False), x, filename='output/D_.onnx')


if __name__ == '__main__':
    main()
