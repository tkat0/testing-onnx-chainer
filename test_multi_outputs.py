import numpy
from onnx import checker
import chainer
from chainer import links as L
from chainer import functions as F
import onnx_chainer
# import onnx_chainer.export
from export import export

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

    # disable rename_tensors version
    export(Model(use_bn=True), x, filename='output/A.onnx')
    export(Model(use_bn=False), x, filename='output/B.onnx')

    # disable check model in onnx_chainer.export
    checker.check_model = lambda x: None

    onnx_chainer.export(Model(use_bn=True), x, filename='output/C.onnx')
    onnx_chainer.export(Model(use_bn=False), x, filename='output/D.onnx')


if __name__ == '__main__':
    main()
