# HW3 Report
## System info
- Windows 11 Pro 22H2
- CPU - Intel i5-12400F
- vCPU - 4, RAM - 8gb

## Task info
Image classification for dogs breeds based on [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset
Using ResNet18

## Repo info
```
📦model_repository
 ┣ 📂ensemble-onnx
 ┃ ┣ 📂1
 ┃ ┃ ┗ 📜.gitkeep
 ┃ ┗ 📜config.pbtxt
 ┣ 📂image-preproc
 ┃ ┣ 📂1
 ┃ ┃ ┗ 📜model.py
 ┃ ┗ 📜config.pbtxt
 ┗ 📂onnx-resnet18
 ┃ ┣ 📂1
 ┃ ┃ ┣ 📜.gitkeep
 ┃ ┃ ┣ 📜model.onnx
 ┃ ┃ ┗ 📜model.onnx.dvc
 ┃ ┗ 📜config.pbtxt
```

## Performance metrics
### Optimization
| instaces | delay | throughput, infer/sec | latency, usec |
| -------- | ----- | ---------- | ----------- |
| 1        | No    | 1353       | 23600       |
| 1        | 1000  | 1300       | 24600       |
| 1        | 2000  | 1290       | 24700       |
| 1        | 4000  | 1327       | 24000       |
| 1        | 500   | 1275       | 25100       |
| 2        | 1000  | 1395       | 22300       |
| 2        | 2000  | 1372       | 23300       |
| 2        | 500   | 1358       | 23500       |

Best params - instances=2, delay=1000
