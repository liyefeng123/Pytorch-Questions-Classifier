import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim  # 模型优化器模块
import torch.nn.functional as F
import sys
import numpy as np

# handle command line code
print('program in')
##############################################处理配置文件#####################################
print('Processing config file and data .......')
with open(sys.argv[3], 'r') as file1:
    idx = 0
    config_title = []  #配置项目名称
    config_content = [] #配置项目内容
    for line in file1.readlines():
        row = line.split('\n')
        row_title = row[0].split(':')
        if len(row_title) > 1:
            config_title.append(row_title[0])
            if len(row_title[1].split('#'))>1:
                row_content = row_title[1].split('#')
                config_content.append(row_content[0])
            else:
                config_content.append(row_title[1])
    New_list = []
    for i in config_title:
        New_list.append(i.replace(" ", ""))
    config_title = New_list
    New_list2 = []
    for i in config_content:
        New_list2.append(i.replace(" ", ""))
    config_content = New_list2
    #配置文件用字典形式储存
    config_dictionary = dict(zip(config_title, config_content))
    #print(config_dictionary)
###############################处理glove(预训练)###############################################
def embLayer():
    with open(config_dictionary['path_pre_emb'], 'r') as file1:
        idx = 0
        # 对word做一个索引 为了之后embedding
        word2idx = {}
        emb = []
        vocab = []
        for line in file1.readlines():
            row = line.split('\t')
            row_emb = row[1].split(' ')
            word = row[0]
            vocab.append(word)
            word2idx[word] = idx
            idx += 1
            emb.append(row_emb)

    # 将结构化数据转换为ndarray
    emb = np.asarray(emb, dtype="float32")
    vocab = np.asarray(vocab)
    return word2idx, emb, vocab
word2idx, emb, vocab = embLayer();
weight = torch.FloatTensor(emb)
##########################################处理训练集得到word2ix################################
def data_input(path):
    with open(path, 'r') as file2:
        label = []
        questions = []
        for line in file2.readlines():
            row = line.split(' ')
            label.append(row[0])
            questions.append(row[1:-1])
        return label, questions

label_train, questions_train = data_input(config_dictionary['path_train'])
questions = questions_train #+ questions_test
label = label_train #+ label_test
if config_dictionary['lowercase'] == 'true':
    new_ques = []
    new_ques_list = []
    for ques in questions:
        for word in ques:
            word = word.lower()
            new_ques.append(word)
        new_ques_list.append(new_ques)
        new_ques = []
    questions = new_ques_list
word2ix = {}  # 单词的索引字典
i = 0
for ques in questions:
    for word in ques:
        if word not in word2ix:
            word2ix[word] = i
            i += 1
############################################计算标签数量###########################################
def data_input():
    with open(config_dictionary['path_train'], 'r') as file2:
        label = []
        questions = []
        for line in file2.readlines():
            row = line.split(' ')
            label.append(row[0])
            questions.append(row[1:-1])
        return label, questions
label, questions = data_input()
# 处理标签，建立标签字典
label_dictionary = list(set(label))
##############################################定义神经网络#########################################
#bag of word model using pre train weight
class bow_pre(torch.nn.Module):
    def __init__(self, hidden_dim, tagset_size):
        super(bow_pre, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.word_embeddings = nn.Embedding.from_pretrained(weight,freeze=config_dictionary['freeze'])
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        sentence_vector = sum(embeds) / len(embeds)
        tag_space = self.hidden2tag(sentence_vector)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores
#bilstm model using pre train weight
class BiLSTM_pre(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, weight, tagset_size):
        super(BiLSTM_pre, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding.from_pretrained(weight,freeze=config_dictionary['freeze'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        h = torch.cat((_[0][0], _[0][1]), 1)
        tag_space = self.hidden2tag(h)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores
#bag of word model using random vector
class bow_random(torch.nn.Module):
    def __init__(self,embedding_dim, tagset_size, vocab_size):
        super(bow_random, self).__init__()
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,)
        self.hidden2tag = torch.nn.Linear(embedding_dim, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        sentence_vector = sum(embeds) / len(embeds)
        tag_space = self.hidden2tag(sentence_vector)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores
#bag of word model using random vector
class BiLSTM_random(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTM_random, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        h = torch.cat((_[0][0], _[0][1]), 1)
        tag_space = self.hidden2tag(h)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores
#Ensemble
class Ensemble(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, weight, tagset_size):
        super(Ensemble, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        self.word_embeddings = nn.Embedding.from_pretrained(weight,freeze=config_dictionary['freeze'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm_hidden = nn.Linear(hidden_dim * 2, tagset_size)
        self.bow_hidden = torch.nn.Linear(300, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        h = torch.cat((_[0][0], _[0][1]), 1)
        lstm_space = self.lstm_hidden(h)
        lstm_scores = F.log_softmax(lstm_space.view(1, -1), dim=1)
        sentence_vector = sum(embeds) / len(embeds)
        bow_space = self.bow_hidden(sentence_vector)
        bow_scores = F.log_softmax(bow_space.view(1, -1), dim=1)
        tag_scores = torch.mean(torch.stack([bow_scores, lstm_scores]), 0)
        return tag_scores
#################################################定义训练函数###################################################
def train_bow_random(model,word2ix):
    def data_input():
        with open(config_dictionary['path_train'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')
                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    # 处理标签，建立标签字典
    label_dictionary = list(set(label))
    label_dictionary.sort()
    #修正大小写
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    # 更新标签列表，用数字替换标签
    lab2ix = dict()
    for i in label:
        for j in label_dictionary:
            if i == j:
                ind = label_dictionary.index(i)
                lab2ix[j] = ind
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)
    label_dictionary = np.array(label_dictionary)
    np.save(config_dictionary['path_label_dictionary'], label_dictionary)
    # 准备数据集
    train_data = questions[0:]
    train_label = label[0:]
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(config_dictionary['lr_param']))
    print('Start training .......')
    model.train()
    for epoch in range(int(config_dictionary['epoch'])):
        print('Neural network ' + str(epoch) + ' iteration')
        for i in range(len(train_data)):
            # Step 1. Remember that Pytorch accumulates gradients.tgg
            # We need to clear them out before each instance
            sentence = train_data[i]
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word2ix)
            m = train_label[i]
            b = [lab2ix[train_label[i]]]
            # Step 2. Run our forward pass.
            tag_scores = model(sentence_in)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            #print('tag_scores', torch.tensor(b))
            loss = loss_function(tag_scores, torch.tensor(b))
            loss.backward()
            optimizer.step()
    torch.save(model, config_dictionary['path_model'])
    print('the model bow_random has been trained and saved')

def train_bilstm_random(model,word2ix):
    def data_input():
        with open(config_dictionary['path_train'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')

                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    # 处理标签，建立标签字典
    label_dictionary = list(set(label))
    label_dictionary.sort()
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    # 更新标签列表，用数字替换标签
    lab2ix = dict()
    for i in label:
        for j in label_dictionary:
            if i == j:
                ind = label_dictionary.index(i)
                lab2ix[j] = ind
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)
    label_dictionary = np.array(label_dictionary)
    np.save(config_dictionary['path_label_dictionary'], label_dictionary)
    # 准备数据集
    train_data = questions[0:]
    train_label = label[0:]
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(config_dictionary['lr_param']))
    print('Start training .......')
    model.train()
    for epoch in range(int(config_dictionary['epoch'])):
        print('Neural network ' + str(epoch) + ' iteration')
        for i in range(len(train_data)):
            model.zero_grad()
            sentence = train_data[i]
            sentence_in = prepare_sequence(sentence, word2ix)
            tag_scores = model(sentence_in)
            b = [lab2ix[train_label[i]]]
            should = torch.tensor(b)
            loss = loss_function(tag_scores, should)
            loss.backward()
            optimizer.step()
    torch.save(model, config_dictionary['path_model'])
    print('the model bilstm_random has been trained and saved')

def train_bilstm_pre(model,word2idx,vocab,weight):
    # 获取train test data
    # label存在数字表示的标签里 总共50个标签
    # 把问题和label分开
    def data_input():
        with open(config_dictionary['path_train'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')
                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    ques = questions[0:]
    # 映射标签
    # vec存储每个问题的vectors
    sen_vec_list = []
    # x：每一个具体的问题
    for x in ques:
        vec_list = []
        # n: 每一个问题中的单词
        for n in x:
            if n in vocab:
                index = word2idx[n]
                vec_list.append(index)
                # vec_list.append(word_emb)
        sen_vec_list.append(vec_list)
    # 处理标签，建立标签字典
    label_dictionary = set(label)
    label_dictionary = list(label_dictionary)
    label_dictionary.sort()
    label_dictionary = np.array(label_dictionary)
    np.save(config_dictionary['path_label_dictionary'], label_dictionary)
    # 更新标签列表，用数字替换标签
    new_label = []
    for i in label:
        n = 0
        for j in label_dictionary:
            if i == j:
                new_label.append(n)
            n += 1
    # 准备数据集
    for i in range(len(sen_vec_list)):
        sen_vec_list[i] = torch.Tensor(sen_vec_list[i]).long()
    new_label = torch.tensor(new_label).long()
    train_data = sen_vec_list[0:]
    train_label = new_label[0:]

    #定义参数
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(config_dictionary['lr_param']))
    print('Start training .......')
    model.train()
    for epoch in range(int(config_dictionary['epoch'])):
        print('Neural network ' + str(epoch) + ' iteration')
        for i in range(len(train_data)):
            model.zero_grad()
            # Step 2. Run our forward pass.
            tag_scores = model(train_data[i])
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, train_label[i].unsqueeze_(dim=0))
            loss.backward()
            optimizer.step()
    torch.save(model,config_dictionary['path_model'])
    print('the model bilstm_pre has been trained and saved')

def train_bow_pre(model,word2idx,vocab,weight):
    # 获取train test data
    # label存在数字表示的标签里 总共50个标签
    # 把问题和label分开
    def data_input():
        with open(config_dictionary['path_train'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')

                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    ques = questions[0:]
    # 映射标签
    # vec存储每个问题的vectors
    sen_vec_list = []
    # x：每一个具体的问题
    for x in ques:
        vec_list = []
        # n: 每一个问题中的单词
        for n in x:
            if n in vocab:
                index = word2idx[n]
                vec_list.append(index)
                # vec_list.append(word_emb)
        sen_vec_list.append(vec_list)
    # 处理标签，建立标签字典
    label_dictionary = set(label)
    label_dictionary = list(label_dictionary)
    label_dictionary.sort()
    label_dictionary = np.array(label_dictionary)
    np.save(config_dictionary['path_label_dictionary'], label_dictionary)
    # 更新标签列表，用数字替换标签
    new_label = []
    for i in label:
        n = 0
        for j in label_dictionary:
            if i == j:
                new_label.append(n)
            n += 1
    # 准备数据集
    for i in range(len(sen_vec_list)):
        sen_vec_list[i] = torch.Tensor(sen_vec_list[i]).long()
    new_label = torch.tensor(new_label).long()
    train_data = sen_vec_list[0:]
    train_label = new_label[0:]
    # 训练模型
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr= float(config_dictionary['lr_param']))
    # 训练
    print('Start training .......')
    model.train()
    for epoch in range(int(config_dictionary['epoch'])):
        print('Neural network ' + str(epoch) + ' iteration')
        for i in range(len(train_data)):
            model.zero_grad()
            tag_scores = model(train_data[i])
            loss = loss_function(tag_scores, train_label[i].unsqueeze_(dim=0))
            loss.backward()
            optimizer.step()
    torch.save(model, config_dictionary['path_model'])
    print('the model bow_pre has been trained and saved')
def train_ensemble_pre(model,word2idx,vocab):
    # label存在数字表示的标签里 总共50个标签
    # 把问题和label分开
    def data_input():
        with open(config_dictionary['path_train'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')
                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    ques = questions[0:]
    # 映射标签
    # vec存储每个问题的vectors
    sen_vec_list = []
    # x：每一个具体的问题
    for x in ques:
        vec_list = []
        # n: 每一个问题中的单词
        for n in x:
            if n in vocab:
                index = word2idx[n]
                vec_list.append(index)
                # vec_list.append(word_emb)
        sen_vec_list.append(vec_list)
    # 处理标签，建立标签字典
    label_dictionary = set(label)
    label_dictionary = list(label_dictionary)
    label_dictionary.sort()
    label_dictionary = np.array(label_dictionary)
    np.save(config_dictionary['path_label_dictionary'], label_dictionary)
    # 更新标签列表，用数字替换标签
    new_label = []
    for i in label:
        n = 0
        for j in label_dictionary:
            if i == j:
                new_label.append(n)
            n += 1
    # 准备数据集
    for i in range(len(sen_vec_list)):
        sen_vec_list[i] = torch.Tensor(sen_vec_list[i]).long()
    new_label = torch.tensor(new_label).long()
    train_data = sen_vec_list[0:]
    train_label = new_label[0:]
    # 训练模型
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr= float(config_dictionary['lr_param']))
    # 训练
    print('Start training .......')
    model.train()
    for epoch in range(int(config_dictionary['epoch'])):
        print('Neural network ' + str(epoch) + ' iteration')
        for i in range(len(train_data)):
            model.zero_grad()
            tag_scores = model(train_data[i])
            loss = loss_function(tag_scores, train_label[i].unsqueeze_(dim=0))
            loss.backward()
            optimizer.step()
    torch.save(model, config_dictionary['path_model'])
    print('the model ensemble_pre has been trained and saved')
############################################预处理的测试#####################################（新加的)
def test_pre(model,label_dictionary,word2idx,vocab):
    #处理数据
    def data_input():
        with open(config_dictionary['path_test'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')
                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    ques = questions[0:]
    # 映射标签
    # vec存储每个问题的vectors
    sen_vec_list = []
    # x：每一个具体的问题
    for x in ques:
        vec_list = []
        # n: 每一个问题中的单词
        for n in x:
            if n in vocab:
                index = word2idx[n]
                vec_list.append(index)
        sen_vec_list.append(vec_list)
    # 处理标签，建立标签字典
    # 更新标签列表，用数字替换标签
    new_label = []
    for i in label:
        n = 0
        for j in label_dictionary:
            if i == j:
                new_label.append(n)
            n += 1
    # 准备数据集
    for i in range(len(sen_vec_list)):
        sen_vec_list[i] = torch.Tensor(sen_vec_list[i]).long()
    new_label = torch.tensor(new_label).long()
    test_data = sen_vec_list[0:]
    test_label = new_label[0:]
    #模型测试
    model.eval()
    right_num = 0
    output = []
    for i in range(len(test_data)):
        pre_tag = model(test_data[i])
        pre_tag = pre_tag.detach().numpy()
        print('Predictive label:', label_dictionary[np.argmax(pre_tag)], ',', 'Real label', label_dictionary[test_label[i]])
        output.append('Predictive label:'+ label_dictionary[np.argmax(pre_tag)] + ',' + 'Real label:' + label_dictionary[test_label[i]] )
        if np.argmax(pre_tag) == test_label[i]:
            right_num += 1
    output.append('right_rate: ' + str(right_num / len(test_data)))
    print('right_rate', right_num / len(test_data))
    file = open(config_dictionary['path_eval_result'], 'w')
    for i in range(len(output)):
        file.write(str(output[i]));
        file.write('\n');
    file.close()
############################################随机的测试###########################################
def test_random(model,label_dictionary,word2ix):
    def data_input():
        with open(config_dictionary['path_test'], 'r') as file2:
            label = []
            questions = []
            for line in file2.readlines():
                row = line.split(' ')
                label.append(row[0])
                questions.append(row[1:-1])
            return label, questions
    label, questions = data_input()
    if config_dictionary['lowercase'] == 'true':
        new_ques = []
        new_ques_list = []
        for ques in questions:
            for word in ques:
                word = word.lower()
                new_ques.append(word)
            new_ques_list.append(new_ques)
            new_ques = []
        questions = new_ques_list
    lab2ix = dict()
    for i in label:
        for j in label_dictionary:
            if i == j:
                ind = label_dictionary.index(i)
                lab2ix[j] = ind
    def prepare_sequence(seq, to_ix):
        idxs = []
        for w in seq:
            if w in to_ix:
                idx = to_ix[w]
                idxs.append(idx)
        return torch.tensor(idxs, dtype=torch.long)
    test_data = questions[0:]
    test_label = label[0:]
    right = 0
    output = []

    for g in range(len(test_data)):
        with torch.no_grad():
            inputs = prepare_sequence(test_data[g], word2ix)
            tag_scores = model(inputs)
            pre = torch.argmax(tag_scores)
            shou = lab2ix[test_label[g]]
            print('Predictive label:', label_dictionary[pre], ',', 'Real label:', label_dictionary[shou])
            output.append('Predictive label:'+ label_dictionary[pre] + ',' + 'Real label: ' + label_dictionary[shou])
            if pre == shou:
                right += 1
    output.append('right_rate: ' + str(float(right / len(test_data))))
    print('right_rate', float(right / len(test_data)))
    file = open(config_dictionary['path_eval_result'], 'w')
    for i in range(len(output)):
        file.write(str(output[i]));
        file.write('\n');
    file.close()
#############################################根据命令行调用函数#############################################
if sys.argv[1] == 'train':
    if config_dictionary['model'] == 'bow_pre_train':
        model = bow_pre(int(config_dictionary['word_embedding_dim']), len(label_dictionary))
        train_bow_pre(model,word2idx,vocab,weight)
    if config_dictionary['model'] == 'bow_random':
        model = bow_random(int(config_dictionary['word_embedding_dim']),len(label_dictionary),len(word2ix))
        train_bow_random(model,word2ix)
    if config_dictionary['model'] == 'bilstm_pre_train':
        model = BiLSTM_pre(int(config_dictionary['word_embedding_dim']), int(config_dictionary['hidden_dim']),weight,len(label_dictionary))
        train_bilstm_pre(model,word2idx,vocab,weight)
    if config_dictionary['model'] == 'bilstm_random':
        model = BiLSTM_random(int(config_dictionary['word_embedding_dim']), int(config_dictionary['hidden_dim']), len(word2ix), len(label_dictionary))
        train_bilstm_random(model,word2ix)
    if config_dictionary['model'] == 'ensemble_pre_train':
        model = Ensemble(int(config_dictionary['word_embedding_dim']), int(config_dictionary['hidden_dim']), weight, len(label_dictionary))
        train_ensemble_pre(model,word2idx,vocab)

if sys.argv[1] == 'test':
    if config_dictionary['model'] == 'bow_pre_train':
        model = torch.load(config_dictionary['path_model'])
        label_dictionary = []
        label_dictionary = np.load(config_dictionary['path_label_dictionary'])
        label_dictionary = label_dictionary.tolist()
        print('Start testing......')
        test_pre(model,label_dictionary,word2idx,vocab)
        print(config_dictionary['model'] + ' test finished')
    if config_dictionary['model'] == 'bilstm_pre_train':
        model = torch.load(config_dictionary['path_model'])
        label_dictionary = []
        label_dictionary = np.load(config_dictionary['path_label_dictionary'])
        label_dictionary = label_dictionary.tolist()
        print('Start testing......')
        test_pre(model, label_dictionary, word2idx, vocab)
        print(config_dictionary['model'] + ' test finished')
    if config_dictionary['model'] == 'ensemble_pre_train':
        model = torch.load(config_dictionary['path_model'])
        label_dictionary = []
        label_dictionary = np.load(config_dictionary['path_label_dictionary'])
        label_dictionary = label_dictionary.tolist()
        print('Start testing......')
        test_pre(model, label_dictionary, word2idx, vocab)
        print(config_dictionary['model'] + ' test finished')
    if config_dictionary['model'] == 'bow_random':
        model = torch.load(config_dictionary['path_model'])
        label_dictionary = []
        label_dictionary = np.load(config_dictionary['path_label_dictionary'])
        print('Start testing......')
        label_dictionary = label_dictionary.tolist()
        test_random(model,label_dictionary,word2ix)
        print(config_dictionary['model'] + ' test finished')
    if config_dictionary['model'] == 'bilstm_random':
        print('start to test bilstm_random model')
        model = torch.load(config_dictionary['path_model'])
        label_dictionary = []
        label_dictionary = np.load(config_dictionary['path_label_dictionary'])
        label_dictionary = label_dictionary.tolist()
        print('Start testing......')
        test_random(model, label_dictionary, word2ix)
        print(config_dictionary['model'] + ' test finished')
