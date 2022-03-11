import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
import random
import time
import datetime
import itertools

"""
    purpose : BERT 모델에 대한 Tokenizer 세팅

    output :
        - data_dataloader
"""
def tokenizer_setting():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    return tokenizer

"""
    purpose : BERT 모델에 대한 전처리(임베딩)
    
    input :
        - data : train/test 데이터
        
    output :
        - data_dataloader
"""
def preprocessing(X_data, y_data, tokenizer, process_type='test'):
    sentences = ["[CLS] " + str(document) + " [SEP]" for document in X_data]
    labels = y_data.values

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
    MAX_LEN = 128
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
        
    BATCH_SIZE = 32
    
    if(process_type == 'train'):
        data_inputs, validation_inputs, data_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels, 
                                                                                    random_state=42, 
                                                                                    test_size=0.1)
        # 어텐션 마스크를 train과 valid로 분리
        data_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                               input_ids,
                                                               random_state=42, 
                                                               test_size=0.1)

        data_inputs = torch.tensor(data_inputs) # 시퀀스를 토큰 ID로 표현
        data_labels = torch.tensor(data_labels) # 긍/부정
        data_masks = torch.tensor(data_masks) # 패딩 마스크 (attention mask)
        
        validation_inputs = torch.tensor(validation_inputs)
        validation_labels = torch.tensor(validation_labels)
        validation_masks = torch.tensor(validation_masks)    
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)
        
    else:
        data_inputs = torch.tensor(input_ids) # 시퀀스를 토큰 ID로 표현
        data_labels = torch.tensor(labels) # 긍/부정
        data_masks = torch.tensor(attention_masks) # 패딩 마스크 (attention mask)
        
    data_data = TensorDataset(data_inputs, data_masks, data_labels)
    data_sampler = RandomSampler(data_data)
    data_dataloader = DataLoader(data_data, sampler=data_sampler, batch_size=BATCH_SIZE)
    
    
    if(process_type == 'train'):
        return data_dataloader, validation_dataloader
    
    else:
        return data_dataloader


def GPU_setting():
    n_devices = torch.cuda.device_count()

    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        for i in range(n_devices):
            print("Device",i,":", torch.cuda.get_device_name(i))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')
    
    return device
        

"""
    purpose : hyperparameter 세팅
    
    input :
        - model, train_dataloader, lr, eps, epochs
        
    output :
        - optimizer, epochs, total_steps, scheduler
"""        
def hyperparmeter_setting(model, train_dataloader, lr=2e-5, eps=1e-8, epochs=2):
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(),
                      lr = lr, # 학습률
                      eps = eps # 0으로 나누는 것을 방지하기 위한 epsilon 값
                    )

    # 에폭수
    epochs = epochs

    # 총 훈련 스텝 = 배치반복 횟수 * 에폭
    total_steps = len(train_dataloader) * epochs

    # lr 조금씩 감소시키는 스케줄러
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    return optimizer, epochs, total_steps, scheduler

# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


def initial_setting(model, seed_val=42):
    # 재현을 위해 랜덤시드 고정
    seed_val = seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 그래디언트 초기화
    model.zero_grad()
    
    return model


"""
    purpose : train 진행
    
    input :
        - model, epochs, train_dataloader, validation_dataloader, optimizer, scheduler, device
        
    output :
        - model
"""  
def run_train(model, epochs, train_dataloader, validation_dataloader, optimizer, scheduler, device, path):
    first_start_time = time.time()
    
    # 에폭 수만큼 반복
    for epoch in range(epochs):

        # ========================================
        #               1. Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        # 시작 시간 설정
        start_time = time.time()

        # 로스 초기화
        total_loss = 0

        # 훈련모드로 변경
        model.train()

        for step, batch in enumerate(train_dataloader):
            # 경과 정보 표시 (step 500번마다 출력)
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - start_time)
                print('Batch {:>5,}  of  {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 배치를 GPU에 올림
            batch = tuple(b.to(device) for b in batch)

            # 배치에서 데이터 추출 (input, mask, label 순으로 넣었었음)
            b_input_ids, b_input_mask, b_labels = batch

            # forward 수행
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                           token_type_ids=None,
                            labels=b_labels)

            # 로스 구함
            loss = outputs.loss # outputs[0]

            # 총 로스 계산
            total_loss += loss.item()

            # Backward 수행으로 그래디언트 계산 (Back-propagation)
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # 예제 코드에서는 1.0이었음

            # 그래디언트를 이용해 가중치 파라미터를 lr만큼 업데이트
            optimizer.step()

            # 스케줄러로 학습률 감소
            scheduler.step()

            # 그래디언트 초기화
            ## (호출시 경사값을 0으로 설정. 이유 : 반복 때마다 기울기를 새로 계산하기 때문)
            model.zero_grad()

        # 1 에폭이 끝나면 평균 train 로스 계산 (전체 loss / 배치 수)
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - start_time)))
        
        # 체크포인트 저장
        print("  Model Checkpoint Save")
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }, path)
        
        # ========================================
        #               2. Validation
        # ========================================

        # 1 에폭이 끝나면 validation 시행
        print("")
        print("Running Validation...")

        # 시작 시간 설정
        start_time = time.time()

        # 평가 모드로 변경
        model.eval()

        # 변수 초기화
        total_valid_accuracy = 0
        nb_valid_steps = 0

        # valid 데이터로더에서 배치만큼 반복하여 가져옴
        for batch in validation_dataloader:

            # 배치를 GPU에 넣음
            batch = tuple(b.to(device) for b in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # 그래디언트 계산 안함!
            with torch.no_grad():
                # Forward 수행
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)

            # 로스 구함 (train할 때는 loss, validation할 때는 logits)
            ## logits은 softmax를 거치기 전의 classification score를 반환합니다. shape: (batch_size, config.num_labels)
            logits = outputs.logits

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            valid_accuracy = flat_accuracy(logits, label_ids)
            total_valid_accuracy += valid_accuracy

        print("  Accuracy: {0:.2f}".format(total_valid_accuracy/len(validation_dataloader)))
        print("  Validation took: {:}".format(format_time(time.time() - start_time)))

    print("")
    print("Total took: {:}".format(format_time(time.time() - first_start_time)))
    print("Training complete!")
    
    return model


"""
    purpose : test 진행
    
    input :
        - model, test_dataloader, device
"""  
def run_test(model, test_dataloader, device):
    #시작 시간 설정
    start_time = time.time()

    # 평가모드로 변경
    model.eval()
    
    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(test_dataloader):
        # 경과 정보 표시
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - start_time)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(b.to(device) for b in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데TensorDataset이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

    print("")
    print("Accuracy: {0:.2f}".format(eval_accuracy/len(test_dataloader)))
    print("Test took: {:}".format(format_time(time.time() - start_time)))
    

# 입력 데이터 변환
def convert_input_data(tokenizer, sentences):

    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks



"""
    purpose : test 진행
    
    input :
        - model, sentence
    output :
        - logits
""" 
def test_sentence_unit(model, device, tokenizer, sentence):
        
    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(tokenizer, sentence)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits


def test_sentence_many(model, device, tokenizer, sentences):
    start_time = time.time()
    
    # 출력된 label 리스트
    label_list = list() 
    
    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(tokenizer, sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    
    test_data = TensorDataset(b_input_ids, b_input_mask)
    test_dataloader = DataLoader(test_data, batch_size=32)
    
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(test_dataloader):
        # 경과 정보 표시
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - start_time)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(b.to(device) for b in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask = batch
    
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        preds = logits.detach().cpu().numpy()
        pred_flat = np.argmax(preds, axis=1).flatten()
        label_list.append(list(pred_flat))
    
    # 이중 리스트를 단일 리스트로 변경
    result = list(itertools.chain.from_iterable(label_list))    
    return result


def save_checkpoint(state, path, file_name='checkpoint.pth.tar'):
    file_path = path + file_name
    print(f"file_path: {file_path}")
    torch.save(state, file_path)