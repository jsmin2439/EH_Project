import os
import torch
import logging


def train(model, data_loader, criterion, optimizer, cuda):
    '''
    모델의 학습을 위한 함수
    model: 학습에 사용되는 뉴럴 네트워크 모델
    data_loader: training data를 불러오는 객체
    criterion: loss function으로 cross entropy 사용
    optimizer: loss function에 따라 model의 parameter를 업데이트
    cuda: GPU 사용 여부
    '''
    model.train() # model을 training setting으로 교체

    # loss, accuracy 기록 (AvgMeter class 참고)
    train_loss = AvgMeter()
    train_acc = AvgMeter()
    for imgs, targets in data_loader:
        if cuda: # GPU를 사용하는 경우 data_loader의 출력을 CPU에서 GPU로 옮겨줌
            imgs, targets = imgs.cuda(), targets.cuda()
        outputs = model(imgs) # model 예측
        loss = criterion(outputs, targets) # loss 계산

        # model parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss, accuracy 기록
        train_loss.update(loss.item(), targets.size(0))
        train_acc.update(cal_acc(outputs, targets), targets.size(0))
    return train_loss.avg, train_acc.avg

def valid(model, data_loader, saver, cuda):
    '''
    모델의 평가를 위한 함수
    model: 평가에 사용되는 뉴럴 네트워크 모델
    saver: 모델의 parameter를 저장
    cuda: GPU 사용 여부
    '''
    model.eval() # model을 validation을 위한 상태로 변경

    valid_acc = AvgMeter() # accuracy 기록 (AvgMeter class 참고)

    for imgs, targets in data_loader:
        if cuda: # GPU를 사용하는 경우 data_loader의 출력을 CPU에서 GPU로 옮겨줌
            imgs, targets = imgs.cuda(), targets.cuda()
        with torch.no_grad(): # validation 시에 gradient 계산 불필요
            outputs = model(imgs) # model 예측
        valid_acc.update(cal_acc(outputs, targets), targets.size(0)) # accuracy 기록
    saver.save_checkpoint(model, valid_acc.avg) # 모델 저장
    return valid_acc.avg

def cal_acc(outputs, targets):
    # accuracy를 계산하기 위한 함
    _, pred = torch.max(outputs.data, 1)
    correct = pred.eq(targets).sum().item()
    acc = correct / targets.size(0)
    return acc


class AvgMeter(object):
    '''
    기록을 위함 함수
    reset: 초기화
    update: 매 iteration마다 기록을 update
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Saver:
    '''
    학습 기록 및 학습된 모델 parameter를 저장
    '''
    def __init__(self, args, checkname):
        '''
        args: 파싱한 argument
        checkname: 저장할 기록 파일의 이름
        '''
        # 기록을 위한 폴더, 파일 및 logger 생성
        self.log_dir = 'logs/%s' % checkname
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = self.create_logging()

        # setting 표기
        for param in sorted(vars(args).keys()):
            self.logger.info('--{0} {1}'.format(param, vars(args)[param]))
        self.best = 0

    def save_checkpoint(self, model, acc):
        '''
        모델이 가장 좋은 validation accuracy를 기록한 경우 모델 저
        '''
        is_best = acc > self.best
        if is_best:
            self.best = acc
            torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth'))

    def create_logging(self):
        '''
        학습에 사용한 세팅 및 accuracy 등을 기록하기 위한 logger 생성
        '''
        logger = logging.getLogger('Result_log')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'logs.txt'))
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
        return logger

