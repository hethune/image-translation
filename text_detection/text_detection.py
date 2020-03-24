# -*- coding: utf-8 -*-
import argparse
import pickle
import html

from google.cloud import vision, translate_v2
from PIL import Image
import six

from utils import draw_box, cluster_texts, get_background_color, get_text_color, draw_text
from shape import TextBox

FONT_TYPE = 'font/LucidaGrande.ttc'

def dump(data):
  with open('data/data.dump', 'wb') as f:
    pickle.dump(data, f)

def load():
  with open('data/data.dump', 'rb') as f:
    data = pickle.load(f)
    return data

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
  translate_client = translate_v2.Client()

  # Text can also be a sequence of strings, in which case this method
  # will return a sequence of results for each text.
  result = translate_client.translate(
    text, source_language=source_language, target_language=target_language, model="nmt")

  return html.unescape(result['translatedText'])

def wipe_out_and_translate(img_path, texts):
  print("Clustering")
  clusters = cluster_texts(texts)
  print("Translating")
  for c in clusters:
    translated = translate(c.text)
    print("Translated {} to {}".format(c.text, translated))
    c.text=translated
  im = Image.open(img_path)
  print("Calcuating text color and bg color")
  for c in clusters:
    vertice = [c.vertices[0], c.vertices[2]]
    r,g,b = get_background_color(im, vertice[0][0], vertice[1][0], vertice[0][1], vertice[1][1])
    tr,tg,tb = get_text_color(im, vertice[0][0], vertice[1][0], vertice[0][1], vertice[1][1])
    c.bg_color = (r,g,b)
    c.text_color = (tr, tg, tb)
  # wipe out background
  print("Wiping out original text")
  for c in clusters:
    # vertices = [[x.vertices[0], x.vertices[2]] for x in clusters]
    draw_box(im, [c.vertices[0], c.vertices[2]], fill=c.bg_color)
  # draw text:
  print("Draw new text")
  for c in clusters:
    draw_text(im, c.text, c.text_color, FONT_TYPE, c.vertices[0][0], c.vertices[2][0], c.vertices[0][1], c.vertices[2][1])
  im.show(0)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process Images.')
  parser.add_argument('--path', type=str, default=None, help='path to input image file')
  args = parser.parse_args()

  # texts = detect_text(args.path)  
  texts = load()
  wipe_out_and_translate(args.path, texts)
