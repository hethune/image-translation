# -*- coding: utf-8 -*-
import  sys
import html
import logging
import os
import textwrap
import pickle
from math import ceil

from PIL import Image, ImageDraw, ImageFont, ImageDraw, ImageFilter
import numpy as np
from scipy import stats
from google.cloud import vision, translate_v2

from shape import TextBox

DEBUG = True

dirname = os.path.dirname(__file__)
LOG_FILE = os.path.join(dirname, '../log/image-tranlsation.log')

logger = logging.getLogger('image-tranlsation')

level = logging.DEBUG if DEBUG else logging.INFO

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


def dump(data):
  with open('data/data.dump', 'wb') as f:
    pickle.dump(data, f)

def load():
  with open('data/data.dump', 'rb') as f:
    data = pickle.load(f)
    return data

def draw_box(im, vertices, outline=None, fill=None):
  # create rectangle image 
  new_im = ImageDraw.Draw(im)
  new_im.rectangle(vertices, fill=fill, outline =outline)

def draw_clusters(im, clusters):
  im = im.copy()
  for c in clusters:
    draw_box(im, [c.vertices[0], c.vertices[2]], outline="red")
  im.show()

def low_pass_image(im):
  km = np.array((
        (1/9, 1/9, 1/9),
        (1/9, 1/9, 1/9),
        (1/9, 1/9, 1/9),
      )) 
  k = ImageFilter.Kernel(
      size=km.shape,
      kernel=km.flatten(),
      scale=np.sum(km),
    )
  rgb_im = im.convert('RGB').filter(k)
  return rgb_im

def high_pass_image(im):
  km = np.array((
      (-1/9, -1/9, -1/9),
      (-1/9, 8/9, -1/9),
      (-1/9, -1/9, -1/9),
    )) 
  k = ImageFilter.Kernel(
      size=km.shape,
      kernel=km.flatten(),
      scale=np.sum(km),
    )
  rgb_im = im.convert('RGB').filter(k)
  return rgb_im

def get_average_color(im, x_min, x_max, y_min, y_max, margin=None, background_color=None, threshold=0.2):
  ''' reverse: True get edge; False get backgroud
      bucket rgb by 10s and get most common bucket
      margin: calculate the outside margin
      background_color: minus_background color
  '''
  rt = []
  gt = []
  bt = []
  im = im.convert('RGB')
  x_start = x_min
  y_start = y_min
  x_end = x_max
  y_end = y_max
  if margin:
    x_start = max(0, x_min - margin)
    y_start = max(0, y_min - margin)
    x_end = min(im.size[0], x_max + margin)
    y_end = min(im.size[1], y_max + margin)
  for i in range(x_start, x_end+1):
    for j in range(y_start,y_end+1):
      if margin:
        if x_min <= i and i <= x_max and y_min <= j and j <= y_max:
          continue
      r, g, b = im.getpixel((i, j))
      is_background = True
      tmp = [r,g,b]
      if background_color:
        for idx in range(0, len(background_color)):
          if tmp[idx] < background_color[idx] * (1-threshold) or background_color[idx]*(1+threshold) < tmp[idx]:
            is_background = False
      if background_color and is_background:
        continue
      rt.append(r)
      gt.append(g)
      bt.append(b)
  # hro = np.histogram([x[0] for x in original], bins=16)
  # logger.debug("Original bins: {}".format(hro))
  for i in [rt, gt, bt]:
   hr = np.histogram(i, bins=8)
   logger.debug("Bins: {}".format(hr))
  r_mean = int(np.average(np.array(rt)))
  g_mean = int(np.average(np.array(gt)))
  b_mean = int(np.average(np.array(bt)))
  return (r_mean, g_mean, b_mean)

def find_fit_font(im, text, font_type, m_width, m_height, w_min_scale, h_min_scale, w_overflow, h_overflow):
  draw_txt = ImageDraw.Draw(im.copy())
  success = False
  font_size = 1
  last_step = font_size
  wrap_length = len(text)
  font = ImageFont.truetype(font_type, font_size)
  lines = textwrap.wrap(text, width=wrap_length)
  width, height = draw_txt.textsize(lines[0], font=font)
  while not success:
    # logger.debug("Trying font size {}. size is {} {} lines {} comparing to {} {}".format(font_size, width, height, len(lines), m_width, m_height))

    width_satisfied =  width > m_width * w_min_scale
    height_satisfied =  height * len(lines) > m_height * h_min_scale
    if width_satisfied and height_satisfied:
      success = True
      break

    font_size += 1
    font = ImageFont.truetype(font_type, font_size)
    width, height = draw_txt.textsize(lines[0], font=font)

    # check if width is maxmized 
    if width > m_width*w_overflow:
      # split lines
      wrap_length = ceil(wrap_length/2)
      lines = textwrap.wrap(text, width=wrap_length)
      width, height = draw_txt.textsize(lines[0], font=font)
      # check if height is maximized
      if height * len(lines) > m_height*h_overflow:
        logger.warning("Overflowing; break")
        wrap_length = wrap_length * 2
        success = True
        break
    
  return font_size, wrap_length

def draw_text(im, text, color, font_type, x_min, x_max, y_min, y_max, w_min_scale=0.7, h_min_scale=0.7, w_overflow=1.2, h_overflow=1.5):
  m_width = x_max - x_min
  m_height = y_max - y_min
  font_size, wrap_length = find_fit_font(im, text, font_type, m_width, m_height, w_min_scale, h_min_scale, w_overflow, h_overflow)
  font = ImageFont.truetype(font_type, font_size)
  draw = ImageDraw.Draw(im)
  lines= textwrap.wrap(text, width=wrap_length)
  y_text = y_min
  try:
    for line in lines:
      width, height = font.getsize(line)
      draw.text(((x_min+x_max-width)/2, y_text), line, color ,font=font)
      y_text += height
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
    texts += ' ' + t.text
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
    mean_char_width = sum([x.width() for x in c])/sum([len(x.text) for x in c])
    texts = ' '.join([x.text for x in c])
    # i don't know; need to try
    w_threshold = 12
    logger.debug("threshold {} mean {} std {} mean char width {} texts {}".format(w_threshold, mean, std, mean_char_width, texts))
    # sort from left to right
    c.sort(key=lambda x: x.center()[0])
    tmp = [[c[0]]]
    msg = "Disances "
    for idx in range(1, len(c)):
      distance = c[idx].left() - c[idx-1].right()
      msg += "{}-{}: {}||".format(c[idx-1].text, c[idx].text, distance)
      if distance < w_threshold:
        tmp[-1].append(c[idx])
      elif distance < mean_char_width * 2:
        tmp[-1].append(c[idx])
      else:
        tmp.append([c[idx]])
    logger.debug(msg)
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

  logger.info('Texts:')

  for text in texts:
    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
    text_box = TextBox(text.description, vertices)

    results.append(text_box)

    logger.debug('bounds: {}'.format(vertices))

  dump(results)

  if response.error.message:
    raise Exception(
      '{}\nFor more info on error messages, check: '
      'https://cloud.google.com/apis/design/errors'.format(
        response.error.message))
  return results

def translate(text, source_language='zh-CN', target_language="en"):
  def _split(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
  """Translates text into the target language.
  Make sure your project is whitelisted.

  Target must be an ISO 639-1 language code.
  See https://g.co/cloud/translate/v2/translate-reference#supported_languages
  """
  assert type(text) is list
  translate_client = translate_v2.Client()

  # Text can also be a sequence of strings, in which case this method
  # will return a sequence of results for each text.
  raw_result = []
  for ts in _split(text, 100):
    raw_result += translate_client.translate(
      ts, source_language=source_language, target_language=target_language, model="nmt")

  result = [html.unescape(x['translatedText']) for x in raw_result]
  return result
