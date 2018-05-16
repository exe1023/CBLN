import logging
import random
import os

from data_provider.image_loader import get_img_builder
from data_provider.nlp_utils import GloveEmbeddings

from vqa_preprocess.vqa_tokenizer import VQATokenizer
from vqa_preprocess.vqa_dataset import VQADataset

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import numpy as np

logger = logging.getLogger()

'''
DataLoader : to load the 
                - Train, Validation and Test data
                - Tokenizer
                - Glove Embeddings
'''
class DataEngine(Dataset):

    def __init__(self, 
                 config, 
                 data_dir, 
                 img_dir, 
                 year, 
                 test_set, 
                 data_type):

        self.total_len = None # total size of the dataset being used
        self.use_glove = config['model']['glove']
        self.data_type = data_type

        # Load images
        logger.info('Loading images..')
        self.image_builder = get_img_builder(config['model']['image'], img_dir)
        self.use_resnet = self.image_builder.is_raw_image()
        self.require_multiprocess = self.image_builder.require_multiprocess()

        # Load dictionary
        logger.info('Loading dictionary..')
        self.tokenizer = VQATokenizer(os.path.join(data_dir, config["dico_name"]))

        # Load data
        logger.info('Loading data..')
        if data_type == 'train':
            self.dataset = VQADataset(data_dir, year=year, which_set="train", image_builder=self.image_builder, 
                                                        preprocess_answers=self.tokenizer.preprocess_answers)
        elif data_type == 'val':
            self.dataset = VQADataset(data_dir, year=year, which_set="val", image_builder=self.image_builder, 
                                                       preprocess_answers=self.tokenizer.preprocess_answers)
        else:
            self.dataset = VQADataset(data_dir, year=year, which_set=test_set, image_builder=self.image_builder)
        self.preprocess()
        # Load glove
        self.glove = None
        if self.use_glove:
            logger.info('Loading glove..')
            self.glove = GloveEmbeddings(os.path.join(data_dir, config["glove_name"]))
    
    def __len__(self):
        return len(self.dataset.games)

    def preprocess(self):
        que = [x.question for x in self.dataset.games] 
        tokens = [self.tokenizer.encode_question(x)[0] for x in que]
        self.max_len = max([len(x) for x in tokens]) # max length of the question

        
    '''
    Arguments:
        ind : current iteration which is converted to required indices to be loaded
        data_type: specifies the train('train'), validation('val') and test('test') partition

    Returns:
        image : image in the form of torch.autograd.Varibale - (batch, channels, height, width) format
        tokens : question tokens
        glove_emb : glove embedding of the question
        answer : ground truth tokens
    '''
    def __getitem__(self, ind):

        dataset = self.dataset.games # just for simplified
        data = dataset[ind]

        # get the images from the dataset and convert them into torch.autograd.Variable
        image = torch.Tensor(data.image.get_image())
        # reshape the image to (batch, channels, height, width) format
        image = image.permute(2,0,1).contiguous()

        # get the questions from the dataset, tokenize them and convert them into torch.autograd.Variable
        tokens = self.tokenizer.encode_question(data.question)[0]
        words = self.tokenizer.encode_question(data.question)[1]
        # pad the additional length with unknown token '<unk>'
        for i in range(self.max_len-len(tokens)):
            tokens.append(self.tokenizer.word2i['<unk>'])

        for i in range(self.max_len-len(words)):
            words.append('<unk>')
        tokens = torch.LongTensor(tokens)
        
        # get the ground truth answer, tokenize them and convert them into torch.autograd.Variable
        answer = self.tokenizer.encode_answer(data.majority_answer)
        answer = torch.LongTensor([answer])
        
        # get the glove embeddings of the question token and convert them into torch.autograd.Variable
        glove_emb = self.glove.get_embeddings(words)
        glove_emb = torch.Tensor(glove_emb)

        return image, tokens, glove_emb, answer