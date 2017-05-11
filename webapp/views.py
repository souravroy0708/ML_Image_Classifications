from __future__ import division
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.core import urlresolvers
from django.contrib import messages
from django.contrib.auth import authenticate, login as login_auth,logout
from django.core.urlresolvers import reverse
from django.db.models import Q
from django.contrib.auth.decorators import login_required
import json
from datetime import datetime, timedelta,date
import socket
from robobrowser import RoboBrowser
from bs4 import BeautifulSoup
import re
import urllib
import random
import time
import requests
import os 

import tensorflow as tf, sys

from webapp.models import ImageClassification


import logging
logger = logging.getLogger(__name__)




def flower_classification(image_path=""):
    final_score_list = []
    current_directory = os.getcwd()
    labelfile = "%s/%s" %(current_directory,"tf_files/retrained_labels_flowers.txt")
    model_file = "%s/%s" %(current_directory,"tf_files/retrained_graph_flowers.pb")
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile(labelfile)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            #final_score = '%s (score = %.5f)' % (human_string, score)
            final_score = {}
            final_score["category"] = human_string
            final_score["score"] = "%.2f" %(score*100)
            final_score_list.append(final_score)
    return final_score_list


def image_classification(request):
    """
    Image Classification
    """
    image_classifier_result = {}
    image_file = ""
    latest_result = ImageClassification.objects.filter().order_by("-created_date")[:5]

    if request.method == 'POST':
        image = request.FILES['input_file']
        strore_image = ImageClassification.objects.create(image_file=image)

        image_file = str(strore_image.image_file)
        current_directory = os.getcwd()
        stoted_image_file = "%s/%s" %(current_directory,image_file)

        image_classifier_result = flower_classification(stoted_image_file)
        strore_image.result=image_classifier_result
        strore_image.save()

    return render_to_response('googleSearch/image_classification.html',{"image_classifier_result":image_classifier_result,"image_file":image_file,"latest_result":latest_result},context_instance=RequestContext(request))

