# -*- coding: utf-8 -*-
import  sys

from PIL import Image, ImageDraw
import numpy as np

from shape import TextBox

def draw_box(path, vertices):
  print(vertices)
  im = Image.open(path)
  # create rectangle image 
  new_im = ImageDraw.Draw(im)
  for v in vertices: 
    new_im.rectangle(v, outline ="red")
  im.show()

def remove_parents(text_boxes):
  to_remove = set()
  f_text_boxes = []
  for i in range(0, len(text_boxes)):
    for j in range(i+1, len(text_boxes)):
      a = text_boxes[i]
      b = text_boxes[j]
      if a.contains(b):
        to_remove.add(a)
      if b.contains(a):
        to_remove.add(b)
  f_text_boxes = set(text_boxes) - to_remove
  return f_text_boxes

def combine(text_boxes):
  ''' combine text_boxes, given are already sorted from left to right
  '''
  text_boxes.sort(key=lambda x: x.center()[0])
  texts = ''
  lm = sys.maxsize
  rm = 0
  um = sys.maxsize
  dm = 0
  for t in text_boxes:
    texts += t.text
    for v in t.vertices:
      if v[0] < lm:
        lm = v[0]
      if v[0] > rm:
        rm = v[0]
      if v[1] < um:
        um = v[1]
      if v[1] > dm:
        dm = v[1]
  t_combined = TextBox(texts, [(lm, um), (rm, um), (rm, dm), (lm, dm)])
  return t_combined

def cluster_texts(text_boxes):
  # remove box a if b belongs to a
  f_text_boxes = remove_parents(text_boxes)
  heights = [x.height() for x in f_text_boxes]
  threshold = np.percentile(np.array(heights), 0.3)
  cluster = []
  while len(f_text_boxes) > 0:
    t = f_text_boxes.pop()
    added = False
    for c in cluster:
      _t = c[0]
      if abs(t.center()[1] - _t.center()[1]) <= threshold:
        c.append(t)
        added = True
        break
    if not added:
      cluster.append([t])
  combined_cluster = [combine(c) for c in cluster]
  return combined_cluster
