import argparse
from torchvision import transforms
from torchvision.utils import save_image
import models
from utils import *
from dataloader import make_dataloader

'''
argparse는 터미널 창에서 입력을 파싱할 수 있게 도와줍니다.
예를 들어 main 함수에서 설정하는 변수인 model을 직접 코드를 수정하지 않고 터미널에서 코드 실행시에
python main.py --model XXX 형태로 실행하면 model을 XXX로 결정할 수 있습니다.

model: 사용할 뉴럴 네트워크의 구조
data_dir: digit_data의 경로
batch_size: 한 번의 iteration에 사용할 data instance의 수
load_path: 학습된 뉴럴 네트워크의 weight parameter가 저장된 경로
'''
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet20', help='network architecture, resnet is only implemented')
parser.add_argument('--data_dir', default='../../digit_data', type=str, help='digit data directory')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--load_path', default='checkpoints/resnet20.pth', type=str, help='path of model weight')
args = parser.parse_args()

model = args.model
data_dir = args.data_dir
batch_size = args.batch_size
load_path = args.load_path

# model 생성 및 학습된 parameter load
model = models.__dict__[model](num_classes=10)
model.load_state_dict(torch.load(load_path))
if torch.cuda.is_available(): # GPU 사용이 가능한 경우 사용
    model.cuda()

# data loader 생성 (dataloader.py 참고)
train_loader, valid_loader = make_dataloader(data_dir, 64, batch_size)
# 뉴럴 네트워크를 학습하기 위해 normalization한 이미지를 다시 사람이 알아볼 수 있도록 RGB 값으로 변환
unnorm = transforms.Compose([transforms.Normalize(mean=[0.,0.,0.],
                                              std=[1/0.22327253, 1/0.29523788, 1/0.24583565]),
                         transforms.Normalize(mean=[-0.80048384, -0.44734452, -0.50106468],
                                              std=[1.,1.,1.])])
error_num = 0 # 잘못 분류한 imange의 개수 기록
# validation loader를 사용하여 결과 분석
for img, tar in valid_loader:
    if torch.cuda.is_available(): # GPU 사용이 가능한 경우 GPU 사용
        img = img.cuda()
    _, pred = torch.max(model(img), 1) # 모델의 category 예측

    # unnorm 함수를 통해
    img = torch.stack([unnorm(x) for x in img], dim=0) # visualization을 위해 unnrom transformation 적용

    for idx, (p, t) in enumerate(zip(pred, tar)):
        # 잘못 분류된 이미지를 저장하여 error case 분석을 위한 데이터 확립
        if not p == t:
            save_image(img[idx], 'error_img/error_%s_%s_%s.png' %(error_num, p.item(), t.item()))
            error_num += 1
