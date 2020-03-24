# -*- coding: utf-8 -*-
import  sys
import html
import logging
import os

from PIL import Image, ImageDraw, ImageFont, ImageDraw
import numpy as np
from scipy import stats
from google.cloud import vision, translate_v2

from shape import TextBox

dirname = os.path.dirname(__file__)
LOG_FILE = os.path.join(dirname, '../log/image-tranlsation.log')

logger = logging.getLogger('image-tranlsation')

level = logging.WARNING

logger.setLevel(level)
# create file handler which logs even debug messages
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(level)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(message)s [%(filename)s:%(lineno)s %(funcName)s]', datefmt='%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def draw_box(im, vertices, outline=None, fill=None):
  # create rectangle image 
  new_im = ImageDraw.Draw(im)
  new_im.rectangle(vertices, fill=fill, outline =outline)

def draw_clusters(im, clusters):
  for c in clusters:
    draw_box(im, [c.vertices[0], c.vertices[2]], outline="red")
  im.show()

def get_background_color(im, x_min, x_max, y_min, y_max):
  rgb_im = im.convert('RGB')
  rt = []
  gt = []
  bt = []
  for i in range(x_min, x_max+1):
    for j in range(y_min,y_max+1):
      r, g, b = rgb_im.getpixel((i, j))
      rt.append(r)
      gt.append(g)
      bt.append(b)
  r_mean = int(stats.mode(np.array(rt))[0][0])
  g_mean = int(stats.mode(np.array(gt))[0][0])
  b_mean = int(stats.mode(np.array(bt))[0][0])
  return (r_mean, g_mean, b_mean)

def get_text_color(im, x_min, x_max, y_min, y_max):
  ''' get text color by computing the mode without the background color
  '''
  threshold = 40
  (br,bg,bb) = get_background_color(im, x_min, x_max, y_min, y_max)
  rgb_im = im.convert('RGB')
  rt = []
  gt = []
  bt = []
  for i in range(x_min, x_max+1):
    for j in range(y_min,y_max+1):
      r, g, b = rgb_im.getpixel((i, j))
      # if (r,g,b) != (br, bg, bb):
      if (r < br-threshold or r>br+threshold) or (g < bg-threshold or g>bg+threshold) or (b < bb-threshold or b>bb+threshold):
        rt.append(r)
        gt.append(g)
        bt.append(b)
  r_mean = int(stats.mode(np.array(rt))[0][0])
  g_mean = int(stats.mode(np.array(gt))[0][0])
  b_mean = int(stats.mode(np.array(bt))[0][0])
  # r_mean = 128 if len(rt) == 0 else int(np.mean(np.array(rt)))
  # g_mean = 0 if len(gt) == 0 else int(np.mean(np.array(gt)))
  # b_mean = 0 if len(bt) == 0 else int(np.mean(np.array(bt)))
  return (r_mean, g_mean, b_mean) 

def find_fit_font(im, text, font_type, m_width, m_height, max_scale=0.95, min_scale=0.8):
  draw_txt = ImageDraw.Draw(im.copy())
  success = False
  font_size = 10
  last_step = font_size
  font = ImageFont.truetype(font_type, font_size)
  width, height = draw_txt.textsize(text, font=font)
  while not success and font_size >= 1:
    logger.debug("Trying font size {}. size is {} {} comparing to {} {}".format(font_size, width, height, m_width, m_height))
    if width < m_width * max_scale and height < m_height * max_scale and width > m_width * min_scale and height > m_height * min_scale:
      success = True
      break
    if width > m_width * max_scale or height > m_height * max_scale:
      if font_size - 1 == last_step:
        break
      last_step = font_size
      font_size -= 1
      font = ImageFont.truetype(font_type, font_size)
      width, height = draw_txt.textsize(text, font=font)
      continue
    if width < m_width * min_scale or height < m_height * min_scale:
      if font_size + 1 == last_step:
        break
      last_step = font_size
      font_size += 1
      font = ImageFont.truetype(font_type, font_size)
      width, height = draw_txt.textsize(text, font=font)
      continue
    logger.warning("Cannot find proper font. Using size {}".format(font_size))
    break
  if font_size < 1:
    font_size = 1
  return font_size

def draw_text(im, text, color, font_type, x_min, x_max, y_min, y_max, max_scale=0.95, min_scale=0.8):
  m_width = x_max - x_min
  m_height = y_max - y_min
  font_size = find_fit_font(im, text, font_type, m_width, m_height, max_scale, min_scale)
  font = ImageFont.truetype(font_type, font_size)
  draw = ImageDraw.Draw(im)
  try:
    draw.text((x_min, y_min), text, color ,font=font)
  except OSError as e:
    logger.error("{} not drawed due to OS error".format(text))
    logger.error(e)
  return


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
  # vertcle groups
  heights = [x.height() for x in f_text_boxes]
  v_threshold = np.percentile(np.array(heights), 0.3)
  clusters = []
  while len(f_text_boxes) > 0:
    t = f_text_boxes.pop()
    added = False
    for c in clusters:
      _t = c[0]
      if abs(t.center()[1] - _t.center()[1]) <= v_threshold:
        c.append(t)
        added = True
        break
    if not added:
      clusters.append([t])
  # horizontal groups
  n_clusters = []
  for c in clusters:
    if len(c) == 1:
      n_clusters.append(c)
      continue
    # get cluster member width and std
    widths = np.array([x.width() for x in c])
    mean = np.average(widths)
    std = np.std(widths)
    # i don't know; need to try
    w_threshold = mean + 2*std
    logger.debug("threshold {} mean {} std {}".format(w_threshold, mean, std))
    # sort from left to right
    c.sort(key=lambda x: x.center()[0])
    tmp = [[c[0]]]
    for idx in range(1, len(c)):
      if c[idx].left() - c[idx-1].right() < w_threshold:
        tmp[-1].append(c[idx])
      else:
        tmp.append([c[idx]])
    for t in tmp:
      n_clusters.append(t)
  combined_cluster = [combine(c) for c in n_clusters]
  return combined_cluster


def detect_text(path):
  """Detects text in the file."""

  client = vision.ImageAnnotatorClient()
  with open(path, 'rb') as image_file:
    content = image_file.read()
  image = vision.types.Image(content=content)
  response = client.text_detection(image=image)
  texts = response.text_annotations

  results = []

  print('Texts:')

  for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
    text_box = TextBox(text.description, vertices)

    results.append(text_box)

    print('bounds: {}'.format(vertices))

  print(results)
  dump(results)

  if response.error.message:
    raise Exception(
      '{}\nFor more info on error messages, check: '
      'https://cloud.google.com/apis/design/errors'.format(
        response.error.message))


def translate(text, source_language='zh-CN', target_language="en"):
  """Translates text into the target language.
  Make sure your project is whitelisted.

  Target must be an ISO 639-1 language code.
  See https://g.co/cloud/translate/v2/translate-reference#supported_languages
  """
  assert type(text) is str or type(text) is list
  translate_client = translate_v2.Client()

  # Text can also be a sequence of strings, in which case this method
  # will return a sequence of results for each text.
  raw_result = translate_client.translate(
    text, source_language=source_language, target_language=target_language, model="nmt")

  if type(text) is str:
    result = html.unescape(raw_result['translatedText'])
  elif type(text) is list:
    result = [html.unescape(x['translatedText']) for x in raw_result]
  else:
    raise Exception("Unknown return type from Google translate {} {}".format(type(t_text), t_text))

  return result
