from torch import cuda

class Config:
    data_dir='../input/data/ICDAR17_Korean'
    model_dir='trained_models'
    device='cuda' if cuda.is_available() else 'cpu'
    num_workers=4
    image_size=1024
    input_size=512
    batch_size=12
    learning_rate=1e-3
    max_epoch=200
    save_interval=5
    optimizer='Adam'
    expr_name='config test'
    resume_from='/opt/ml/code/trained_models/latest.pth' # 이어서 학습 할 .pth 경로 없으면 ''