# -*- coding: utf-8 -*-
import argparse

from PIL import Image

from utils import draw_box, cluster_texts, low_pass_image, high_pass_image, get_average_color, draw_text, detect_text, translate, logger, draw_clusters, DEBUG, load
from shape import TextBox

FONT_TYPE = 'text_detection/font/PingFang-SC-Bold.ttf'

def wipe_out_and_translate(img_path, texts, theme=None, target_language="en"):
  if theme == "dark":
    BG_COLOR = (43, 43, 43)
    TEXT_COLOR = (255, 255, 255)
  if theme == "light":
    BG_COLOR = (211, 211, 211)
    TEXT_COLOR = (255, 255, 255)
  if theme == "white":
    BG_COLOR = (255, 255, 255)
    TEXT_COLOR = (43, 43, 43)

  im = Image.open(img_path)
  logger.info("Clustering")
  if DEBUG:
    draw_clusters(im, texts)
  clusters = cluster_texts(texts)
  if DEBUG:
    draw_clusters(im, clusters)
  logger.info("Translating")
  original = [x.text for x in clusters]
  translated = translate(original, target_language=target_language)
  assert len(original) == len(translated)
  for i in range(0, len(translated)):
    clusters[i].text = translated[i]
  logger.info("Translated {} to {}".format(original, translated))
  logger.info("Calcuating text color and bg color")
  # high_pass_im = high_pass_image(im)
  # if DEBUG:
  #   high_pass_im.convert('L').show()

  for c in clusters:
    vertice = [c.vertices[0], c.vertices[2]]
    r,g,b = get_average_color(im, vertice[0][0], vertice[1][0], vertice[0][1], vertice[1][1], margin=5)
    tr,tg,tb = get_average_color(im, vertice[0][0], vertice[1][0], vertice[0][1], vertice[1][1], background_color=(r,g,b))
    logger.debug("background color is {}".format((r,g,b)))
    logger.debug("text color is {}".format((tr,tg,tb)))
    c.bg_color = (r,g,b)
    c.text_color = (tr,tg,tb)
  # wipe out background
  logger.info("Wiping out original text")
  for c in clusters:
    draw_box(im, [c.vertices[0], c.vertices[2]], fill=c.bg_color)
  # draw text:
  logger.info("Draw new text")
  for c in clusters:
    draw_text(im, c.text, c.text_color, FONT_TYPE, c.vertices[0][0], c.vertices[2][0], c.vertices[0][1], c.vertices[2][1])
  im.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process Images.')
  parser.add_argument('--path', '-p',type=str, default=None, help='path to input image file')
  parser.add_argument('--theme', '-t', type=str, default=None)
  parser.add_argument('--target_language', '-tl', type=str, default='en', help="target langguage")
  args = parser.parse_args()

  assert args.theme in [None, 'dark', 'light', 'white']

  texts = detect_text(args.path)  
  # texts = load()
  wipe_out_and_translate(args.path, texts, theme=args.theme, target_language=args.target_language)
