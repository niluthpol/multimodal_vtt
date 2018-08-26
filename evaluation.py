from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from data_resnet import get_test_loader as get_test_loader1
from data_i3d_audio import get_test_loader as get_test_loader2
from model import VSE
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""

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
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (videos, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(videos, captions, lengths,
                                             volatile=True)
											 
        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del videos, captions

    return img_embs, cap_embs


def evalrank(model_path1, model_path2, data_path=None, split='dev', fold5=False, shared_space='both'):
    """
    Evaluate a trained model.
    """
    # load model and options
    checkpoint = torch.load(model_path1)
    opt = checkpoint['opt']
    print(opt)

    if data_path is not None:
        opt.data_path = data_path
    opt.vocab_path = "./vocab/"
    # load vocabulary used by the model				   
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, 'coco_vocab.pkl'), 'rb'))
        
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader1(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs1, cap_embs1 = encode_data(model, data_loader)
	
    # load second model and options
    checkpoint2 = torch.load(model_path2)
    opt = checkpoint2['opt']
    print(opt)

    if data_path is not None:
        opt.data_path = data_path
    opt.vocab_path = "./vocab/"
    # load vocabulary used by the model			   
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, 'coco_vocab.pkl'), 'rb'))
        
    opt.vocab_size = len(vocab)

    # construct model
    model2 = VSE(opt)

    # load model state
    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    data_loader = get_test_loader2(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs2, cap_embs2 = encode_data(model2, data_loader)	
	
    print('Images: %d, Captions: %d' %
          (img_embs2.shape[0] / 20, cap_embs2.shape[0]))

    # no cross-validation, full evaluation
    r, rt = i2t(img_embs1, cap_embs1, img_embs2, cap_embs2, shared_space, measure=opt.measure, return_ranks=True)
    ri, rti = t2i(img_embs1, cap_embs1, img_embs2, cap_embs2, shared_space, measure=opt.measure, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(videos, captions, videos2, captions2, shared_space='both', measure='cosine', return_ranks=False):
    """
    Videos->Text (Video Annotation)
    Videos: (20N, K) matrix of videos
    Captions: (20N, K) matrix of captions
    """

    
    npts = videos.shape[0] / 20
    index_list = []
    print(npts)
	
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):
        # Get query image
        im = videos[20 * index].reshape(1, videos.shape[1])
        im2 = videos2[20 * index].reshape(1, videos2.shape[1])
        # Compute scores
        if 'both' == shared_space:
            d1 = numpy.dot(im, captions.T).flatten()
            d2 = numpy.dot(im2, captions2.T).flatten()
            d= d1+d2
        elif 'object_text' == shared_space:
            d = numpy.dot(im, captions.T).flatten()
        elif 'activity_text' == shared_space:
            d = numpy.dot(im2, captions2.T).flatten()		
			
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])
        # Score
        rank = 1e20
        for i in range(20 * index, 20 * index + 20, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
                flag=i-20 * index
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


		
def t2i(videos, captions, videos2, captions2, shared_space='both', measure='cosine', return_ranks=False):
    """
    Text->Videos (Video Search)
    Videos: (20N, K) matrix of videos
    Captions: (20N, K) matrix of captions
    """
    
    npts = videos.shape[0] / 20
    ims = numpy.array([videos[i] for i in range(0, len(videos), 20)])
    ims2 = numpy.array([videos2[i] for i in range(0, len(videos2), 20)])
	
    ranks = numpy.zeros(20 * npts)
    top1 = numpy.zeros(20 * npts)
    for index in range(npts):
        # Get query captions
        queries = captions[20 * index:20 * index + 20]
        queries2 = captions2[20 * index:20 * index + 20]

        if 'both' == shared_space:
            d1 = numpy.dot(queries, ims.T)
            d2 = numpy.dot(queries2, ims2.T)		
            d = d1+d2
        elif 'object_text' == shared_space:
            d = numpy.dot(queries, ims.T)
        elif 'activity_text' == shared_space:
            d = numpy.dot(queries2, ims2.T)			
		
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[20 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[20 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
