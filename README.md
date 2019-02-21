# testing-onnx-chainer

fixed in onnx-chainer@v1.3.2 :tada:

```bash
$ docker-compose run fixed python test_multi_outputs_fixed.py
```

output/C_.onnx

![](./img/C_.png)

output/D_.onnx

![](./img/D_.png)

---

old

```bash
$ docker-compose run dev python test_multi_outputs.py
```

output/A.onnx

![](./img/A.png)

output/B.onnx

![](./img/B.png)

output/C.onnx

![](./img/C.png)

output/D.onnx

![](./img/D.png)
