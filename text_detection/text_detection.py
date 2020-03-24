# -*- coding: utf-8 -*-
import argparse
import pickle

from PIL import Image

from utils import draw_box, cluster_texts, get_background_color, get_text_color, draw_text, detect_text, translate, logger
from shape import TextBox

FONT_TYPE = 'font/LucidaGrande.ttc'

def dump(data):
  with open('data/data.dump', 'wb') as f:
    pickle.dump(data, f)

def load():
  with open('data/data.dump', 'rb') as f:
    data = pickle.load(f)
    return data

def wipe_out_and_translate(img_path, texts):
  print("Clustering")
  clusters = cluster_texts(texts)
  print("Translating")
  original = [x.text for x in clusters]
  translated = translate(original)
  assert len(original) == len(translated)
  for i in range(0, len(translated)):
    clusters[i].text = translated[i]
  print("Translated {} to {}".format(original, translated))
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
